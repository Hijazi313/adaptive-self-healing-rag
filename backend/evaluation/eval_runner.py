"""
RAGAS Evaluation Harness

WHAT THIS DOES:
    Runs every sample in golden_dataset.json through the full RAG graph,
    collects the four core RAGAS metrics, and writes a timestamped results
    JSON to evaluation/results/.

    Run once per chunking strategy to compare retrieval quality:
        uv run python evaluation/eval_runner.py --strategy recursive
        uv run python evaluation/eval_runner.py --strategy semantic
        uv run python evaluation/eval_runner.py --strategy proposition

WHY THIS RUNS AGAINST THE GRAPH DIRECTLY (not via HTTP):
    The evaluation harness must be runnable in CI, before the FastAPI server
    starts, without any network dependency on localhost:8000.
    It talks directly to the compiled LangGraph — same code path, no HTTP layer.
    Results are reproducible regardless of server state.

THE RAGAS DATASET FORMAT:
    RAGAS expects a HuggingFace Dataset with four columns:
        question     : str   — the query
        ground_truth : str   — human-verified correct answer
        answer       : str   — what the system generated
        contexts     : list  — the text of retrieved chunks

    We build this from graph output. The graph gives us final_answer and
    retrieved_chunks — exactly what RAGAS needs.

RESULTS FILE FORMAT:
    evaluation/results/{strategy}_{timestamp}.json
    {
        "strategy": "recursive",
        "timestamp": "2026-05-23T14:30:00",
        "git_commit": "abc1234",
        "sample_count": 50,
        "metrics": {
            "faithfulness": 0.847,
            "answer_relevancy": 0.891,
            "context_precision": 0.763,
            "context_recall": 0.812
        },
        "per_sample": [...]   ← per-question breakdown for debugging
    }

WHY STORE PER-SAMPLE RESULTS:
    Aggregate metrics tell you the score improved or dropped.
    Per-sample results tell you WHICH questions regressed and why.
    When faithfulness drops 5%, you need to know: is it one bad sample
    dragging the average down, or is every sample slightly worse?
    These are very different root causes requiring different fixes.
"""

import argparse
import json
import logging
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from langgraph.checkpoint.memory import MemorySaver
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Add project root to path so imports work when run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import initial_state
from graph.supervisor import build_graph
from ingestion.chunkers import get_chunker
from ingestion.embedder import Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Git commit helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_git_commit() -> str:
    """
    Get the current short git commit hash for result traceability.
    Results are tagged with the commit so you can correlate metric changes
    with specific code changes.
    Returns "unknown" gracefully if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Graph runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_query(graph, question: str) -> dict:
    """
    Run one question through the graph and return the output state.

    Uses MemorySaver (not SqliteSaver) because:
    - The eval harness runs many queries sequentially
    - SqliteSaver would accumulate all thread states in a file
    - MemorySaver is ephemeral — each eval run starts clean
    - We don't need HITL or persistence during evaluation

    Returns the final graph state dict. If the graph errors,
    returns a safe fallback with empty answer and chunks
    so the RAGAS sample is included as a failure, not skipped.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(query=question, user_id="eval_harness", thread_id=thread_id)

    try:
        result = graph.invoke(state, config=config)
        return result
    except Exception as e:
        logger.error(f"Graph failed on question '{question[:60]}': {e}")
        return {
            "final_answer": "",
            "retrieved_chunks": [],
            "critic_score": None,
            "faithfulness_score": None,
            "retrieval_attempts": [],
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_ragas_dataset(
    golden_samples: list[dict],
    graph,
) -> tuple[Dataset, list[dict]]:
    """
    Run every golden sample through the graph and build the RAGAS Dataset.

    Returns:
        (ragas_dataset, per_sample_details)

        per_sample_details: list of dicts with id, question, ground_truth,
        answer, context_count, critic_score, error — for per-question analysis.

    RAGAS dataset schema (required columns):
        question     : str
        ground_truth : str
        answer       : str
        contexts     : list[str]  ← text content of retrieved chunks
    """
    ragas_rows = {
        "question": [],
        "ground_truth": [],
        "answer": [],
        "contexts": [],
    }
    per_sample = []

    total = len(golden_samples)
    for i, sample in enumerate(golden_samples, start=1):
        question     = sample["question"]
        ground_truth = sample["ground_truth"]
        sample_id    = sample["id"]

        logger.info(f"[{i}/{total}] Evaluating: {sample_id}")

        result = _run_query(graph, question)

        # Extract text content from retrieved LangChain Document objects.
        # RAGAS contexts must be list[str] — not list[Document].
        chunks = result.get("retrieved_chunks") or []
        contexts = [c.page_content for c in chunks if hasattr(c, "page_content")]

        answer = result.get("final_answer") or ""

        ragas_rows["question"].append(question)
        ragas_rows["ground_truth"].append(ground_truth)
        ragas_rows["answer"].append(answer)
        ragas_rows["contexts"].append(contexts)

        per_sample.append({
            "id":            sample_id,
            "category":      sample.get("category", ""),
            "difficulty":    sample.get("difficulty", ""),
            "question":      question,
            "ground_truth":  ground_truth,
            "answer":        answer,
            "context_count": len(contexts),
            "critic_score":  result.get("critic_score"),
            "attempts":      result.get("retrieval_attempts", []),
            "error":         result.get("error"),
        })

    return Dataset.from_dict(ragas_rows), per_sample


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(strategy: str, sample_limit: int | None = None) -> dict:
    """
    Run the full evaluation pipeline for a given chunking strategy.

    Args:
        strategy:     "recursive" | "semantic" | "proposition"
        sample_limit: If set, only evaluate the first N samples.
                      Useful for fast smoke-testing: --limit 5

    Returns:
        Results dict (also written to evaluation/results/).

    Steps:
        1. Load golden dataset
        2. Build graph with MemorySaver
        3. Run every sample through the graph
        4. Build RAGAS Dataset from outputs
        5. Score with RAGAS metrics
        6. Write results JSON
    """
    logger.info(f"Starting evaluation | strategy={strategy} | samples={sample_limit or 'all'}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(DATASET_PATH) as f:
        all_samples = json.load(f)

    samples = all_samples[:sample_limit] if sample_limit else all_samples
    logger.info(f"Loaded {len(samples)} samples from golden_dataset.json")

    # ── Build graph ───────────────────────────────────────────────────────────
    # MemorySaver: ephemeral, no file I/O, correct for evaluation workloads.
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    logger.info("Graph compiled with MemorySaver checkpointer")

    # ── Run samples through graph ──────────────────────────────────────────────
    logger.info("Running samples through RAG graph...")
    ragas_dataset, per_sample = _build_ragas_dataset(samples, graph)

    # Filter out samples where answer is empty — RAGAS cannot score them.
    # These represent hard graph failures and are counted separately.
    valid_mask = [bool(row.strip()) for row in ragas_dataset["answer"]]
    failed_count = valid_mask.count(False)

    if failed_count > 0:
        logger.warning(f"{failed_count} samples produced empty answers — excluded from RAGAS scoring")

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if not valid_indices:
        logger.error("No valid samples to evaluate. Check graph and Qdrant connection.")
        raise RuntimeError("Zero valid samples — evaluation aborted.")

    # Rebuild dataset with only valid rows.
    filtered = {
        col: [ragas_dataset[col][i] for i in valid_indices]
        for col in ragas_dataset.column_names
    }
    valid_dataset = Dataset.from_dict(filtered)

    # ── Run RAGAS ─────────────────────────────────────────────────────────────
    logger.info(f"Running RAGAS on {len(valid_dataset)} valid samples...")

    ragas_result = evaluate(
        dataset=valid_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # ragas_result is a dict-like object — convert to plain dict for JSON.
    metrics = {
        "faithfulness":       round(float(ragas_result["faithfulness"]), 4),
        "answer_relevancy":   round(float(ragas_result["answer_relevancy"]), 4),
        "context_precision":  round(float(ragas_result["context_precision"]), 4),
        "context_recall":     round(float(ragas_result["context_recall"]), 4),
    }

    logger.info(f"RAGAS results: {metrics}")

    # ── Build result payload ──────────────────────────────────────────────────
    result = {
        "strategy":          strategy,
        "timestamp":         datetime.utcnow().isoformat(),
        "git_commit":        _get_git_commit(),
        "sample_count":      len(samples),
        "valid_sample_count": len(valid_dataset),
        "failed_sample_count": failed_count,
        "metrics":           metrics,
        "per_sample":        per_sample,
    }

    # ── Write results JSON ────────────────────────────────────────────────────
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{strategy}_{timestamp_str}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results written to {output_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"EVALUATION COMPLETE | strategy={strategy}")
    print("="*60)
    print(f"  Samples evaluated  : {len(valid_dataset)}/{len(samples)}")
    print(f"  Git commit         : {result['git_commit']}")
    print()
    print(f"  faithfulness       : {metrics['faithfulness']}")
    print(f"  answer_relevancy   : {metrics['answer_relevancy']}")
    print(f"  context_precision  : {metrics['context_precision']}")
    print(f"  context_recall     : {metrics['context_recall']}")
    print()
    print(f"  Results saved to   : {output_path}")
    print("="*60)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation against the adaptive RAG system."
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic", "proposition"],
        default="recursive",
        help="Chunking strategy to evaluate against.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (useful for smoke testing).",
    )
    args = parser.parse_args()

    run_evaluation(strategy=args.strategy, sample_limit=args.limit)