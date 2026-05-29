"""
Regression Gate

WHAT THIS DOES:
    Compares the most recent evaluation results against a stored baseline.
    If faithfulness drops more than REGRESSION_THRESHOLD (5%) from baseline,
    exits with code 1 — which fails a CI pipeline.

    This is the quality gate. It answers the question:
    "Did my last code change make the RAG system worse?"

USAGE:
    # Set a new baseline (after a known-good evaluation):
    uv run python evaluation/regression_gate.py --set-baseline recursive

    # Check current results against baseline (in CI):
    uv run python evaluation/regression_gate.py --check recursive

    # Check all three strategies:
    uv run python evaluation/regression_gate.py --check-all

HOW BASELINE WORKS:
    The baseline is stored in evaluation/results/baseline.json.
    It maps strategy → metrics dict.
You set it once after establishing a known-good evaluation run.
    After that, every eval_runner.py run can be gated against it.

    baseline.json is committed to git — it's the reference point.
    results/*.json files are also committed — they're the audit trail.

WHY EXIT CODE 1 (not just a warning):
    A warning that nobody reads is not a gate.
    Exit code 1 fails the CI job. The merge is blocked.
    A drop in faithfulness means the system is hallucinating more —
    that's a regression that must be explicitly reviewed and accepted,
    not silently merged.

THE REGRESSION THRESHOLD:
    5% is the default. Why not stricter?
    RAGAS metrics have natural variance between runs because:
    - LLM outputs are non-deterministic (even at temperature=0, minor variance exists)
    - Retrieval order can shift slightly with Qdrant version changes
    5% absorbs this noise. If your baseline faithfulness is 0.85,
    a drop to 0.80 is a real signal. A drop to 0.81 is likely noise.
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR   = Path(__file__).parent / "results"
BASELINE_PATH = RESULTS_DIR / "baseline.json"

# The primary metric we gate on. From the original project spec.
PRIMARY_METRIC = "faithfulness"
REGRESSION_THRESHOLD = 0.05  # 5% relative drop triggers failure

# All metrics we report on (even if we only gate on faithfulness).
ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_result(strategy: str) -> dict | None:
    """
    Find and load the most recent results JSON for a given strategy.
    Results are named {strategy}_{timestamp}.json — sort by name descending.
    """
    pattern = f"{strategy}_*.json"
    candidates = sorted(RESULTS_DIR.glob(pattern), reverse=True)

    if not candidates:
        print(f"  No results found for strategy '{strategy}' in {RESULTS_DIR}")
        return None

    latest = candidates[0]
    print(f"  Loading latest result: {latest.name}")
    with open(latest) as f:
        return json.load(f)


def _load_baseline() -> dict:
    """Load baseline.json. Returns empty dict if not set yet."""
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


def _save_baseline(baseline: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Core operations
# ─────────────────────────────────────────────────────────────────────────────

def set_baseline(strategy: str) -> None:
    """
    Set the baseline from the latest results for a strategy.
    Call this after a known-good evaluation run.

    Example:
        After running eval_runner.py --strategy recursive and being satisfied
        with the results, run:
            python evaluation/regression_gate.py --set-baseline recursive
    """
    result = _load_latest_result(strategy)
    if not result:
        print(f"Cannot set baseline: no results found for '{strategy}'")
        sys.exit(1)

    baseline = _load_baseline()
    baseline[strategy] = {
        "metrics":    result["metrics"],
        "timestamp":  result["timestamp"],
        "git_commit": result["git_commit"],
        "sample_count": result["valid_sample_count"],
    }
    _save_baseline(baseline)

    print(f"\n✓ Baseline set for strategy='{strategy}'")
    print(f"  faithfulness      : {result['metrics']['faithfulness']}")
    print(f"  answer_relevancy  : {result['metrics']['answer_relevancy']}")
    print(f"  context_precision : {result['metrics']['context_precision']}")
    print(f"  context_recall    : {result['metrics']['context_recall']}")
    print(f"  Saved to          : {BASELINE_PATH}")


def check_regression(strategy: str) -> bool:
    """
    Compare latest results for a strategy against its baseline.

    Returns:
        True  — no regression detected (gate passes)
        False — regression detected (gate fails)

    Prints a detailed comparison table regardless of outcome.
    """
    result = _load_latest_result(strategy)
    if not result:
        print(f"SKIP: No results to check for '{strategy}'")
        return True  # No results = not a failure; it just hasn't been run

    baseline = _load_baseline()
    if strategy not in baseline:
        print(f"  No baseline set for '{strategy}'. Run --set-baseline {strategy} first.")
        print(f"  Skipping regression check (not a failure).")
        return True

    current_metrics  = result["metrics"]
    baseline_metrics = baseline[strategy]["metrics"]

    print(f"\nRegression check | strategy={strategy}")
    print(f"  Baseline commit : {baseline[strategy]['git_commit']}")
    print(f"  Current commit  : {result['git_commit']}")
    print()
    print(f"  {'Metric':<22} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>8}")
    print(f"  {'-'*62}")

    regression_found = False

    for metric in ALL_METRICS:
        baseline_val = baseline_metrics.get(metric, 0.0)
        current_val  = current_metrics.get(metric, 0.0)
        delta        = current_val - baseline_val
        delta_pct    = delta / baseline_val if baseline_val > 0 else 0.0

        # Gate only on PRIMARY_METRIC (faithfulness).
        # Other metrics are reported but do not fail the gate.
        # This is a deliberate design choice — faithfulness is the
        # hallucination metric. The others matter but are secondary.
        if metric == PRIMARY_METRIC and delta_pct < -REGRESSION_THRESHOLD:
            status = "FAIL ✗"
            regression_found = True
        elif delta_pct < -REGRESSION_THRESHOLD:
            status = "WARN ⚠"
        elif delta_pct >= 0:
            status = "OK   ✓"
        else:
            status = "OK   ✓"

        delta_str = f"{delta:+.4f} ({delta_pct:+.1%})"
        print(f"  {metric:<22} {baseline_val:>10.4f} {current_val:>10.4f} {delta_str:>10} {status:>8}")

    print()
    if regression_found:
        print(
            f"  REGRESSION DETECTED: {PRIMARY_METRIC} dropped more than "
            f"{REGRESSION_THRESHOLD:.0%} from baseline."
        )
        print(f"  Review changes before merging.")
    else:
        print(f"  All checks passed. No regression detected.")

    return not regression_found


def check_all_strategies() -> bool:
    """
    Run regression check for all three strategies.
    Returns True only if ALL pass.
    """
    strategies = ["recursive", "semantic", "proposition"]
    results = []

    for strategy in strategies:
        result_files = list(RESULTS_DIR.glob(f"{strategy}_*.json"))
        if not result_files:
            print(f"\nSKIP: No results for '{strategy}' — run eval_runner.py first")
            continue
        passed = check_regression(strategy)
        results.append((strategy, passed))

    print("\n" + "="*60)
    print("REGRESSION GATE SUMMARY")
    print("="*60)
    all_passed = True
    for strategy, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {strategy:<15} {status}")
        if not passed:
            all_passed = False

    print("="*60)
    return all_passed


def print_comparison_report() -> None:
    """
    Print a side-by-side comparison of all strategies' latest results.
    Useful for the portfolio README — shows chunking strategy tradeoffs.
    """
    strategies = ["recursive", "semantic", "proposition"]
    results = {}

    for strategy in strategies:
        result = _load_latest_result(strategy)
        if result:
            results[strategy] = result["metrics"]

    if not results:
        print("No results found. Run eval_runner.py for at least one strategy first.")
        return

    print("\n" + "="*70)
    print("CHUNKING STRATEGY COMPARISON REPORT")
    print("="*70)
    print(f"  {'Metric':<22}", end="")
    for strategy in strategies:
        if strategy in results:
            print(f"  {strategy:>12}", end="")
    print()
    print(f"  {'-'*60}")

    for metric in ALL_METRICS:
        print(f"  {metric:<22}", end="")
        for strategy in strategies:
            if strategy in results:
                val = results[strategy].get(metric, 0.0)
                print(f"  {val:>12.4f}", end="")
        print()

    print("="*70)
    print()
    print("Interpretation guide:")
    print("  faithfulness      → hallucination rate (higher = less hallucination)")
    print("  answer_relevancy  → answer addresses the question (higher = more on-topic)")
    print("  context_precision → retrieval noise (higher = less irrelevant chunks)")
    print("  context_recall    → retrieval completeness (higher = less missed info)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression gate for RAG evaluation metrics.")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--set-baseline",
        metavar="STRATEGY",
        choices=["recursive", "semantic", "proposition"],
        help="Set baseline from latest results for a strategy.",
    )
    group.add_argument(
        "--check",
        metavar="STRATEGY",
        choices=["recursive", "semantic", "proposition"],
        help="Check latest results against baseline for a strategy.",
    )
    group.add_argument(
        "--check-all",
        action="store_true",
        help="Check all three strategies against their baselines.",
    )
    group.add_argument(
        "--report",
        action="store_true",
        help="Print side-by-side comparison of all strategies.",
    )

    args = parser.parse_args()

    if args.set_baseline:
        set_baseline(args.set_baseline)

    elif args.check:
        passed = check_regression(args.check)
        sys.exit(0 if passed else 1)

    elif args.check_all:
        passed = check_all_strategies()
        sys.exit(0 if passed else 1)

    elif args.report:
        print_comparison_report()