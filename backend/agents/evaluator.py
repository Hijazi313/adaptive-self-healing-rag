"""
agents/evaluator.py — EvaluatorAgent

RESPONSIBILITY:
    Post-generation faithfulness check — the last node before END.
    Scores whether the final_answer is grounded in retrieved_chunks.

    This is NOT the Phase 4 full RAGAS evaluation harness.
    This is an inline, per-query faithfulness check that runs in the graph.

    The Phase 4 harness runs offline against a golden dataset.
    This node runs online, per request, providing real-time quality assurance.

FAITHFULNESS DEFINED:
    A claim in the answer is faithful if it can be directly inferred from
    the retrieved context. A hallucinated claim — even a plausible one —
    is a faithfulness failure.

    Example:
        Context: "The API rate limit is 60 requests per minute."
        Answer: "The API allows 100 requests per minute."
        Faithfulness: 0.0 — the number is not in context.

        Context: "The API rate limit is 60 requests per minute."
        Answer: "The API limits you to 60 requests per minute."
        Faithfulness: 1.0 — directly supported.

WHY gpt-3.5-turbo (not gpt-4o):
    The evaluator runs on EVERY query in production.
    Faithfulness scoring is a simpler task than answer generation.
    GPT-3.5-turbo handles it well and is ~10x cheaper.
    GPT-4o is reserved for QueryAnalyzer, Critic, and Generator where
    quality directly impacts the answer the user receives.

INTEGRATION WITH PHASE 4:
    The faithfulness_score written here is stored in state.
    The Phase 4 regression gate reads these scores from LangSmith traces
    (not from this node directly) to compute baseline comparisons.
    This node and the Phase 4 harness use compatible scoring scales (0.0–1.0).
"""

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import settings
from graph.state import RAGState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a faithfulness evaluator for a RAG system.
Score whether each claim in the answer is directly supported by the context.
Return ONLY valid JSON. No preamble.

Scoring:
- 1.0: every claim in the answer is explicitly supported by the context
- 0.5–0.9: most claims are supported; minor extrapolations present
- 0.0–0.5: significant claims in the answer are not found in the context

Return this exact JSON:
{{
  "faithfulness_score": <float 0.0 to 1.0>,
  "reasoning": "one sentence explanation"
}}"""
    ),
    (
        "human",
        "Context:\n{context}\n\nAnswer:\n{answer}"
    ),
])


def _format_context_brief(chunks: list[Document]) -> str:
    """
    Compact context format for the evaluator.
    The evaluator needs to check claims — it doesn't need full chunk formatting.
    Keep within GPT-3.5's context window efficiently.
    """
    parts = [chunk.page_content[:300] for chunk in chunks]
    return "\n---\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Agent node function
# ─────────────────────────────────────────────────────────────────────────────

def evaluator_node(state: RAGState) -> dict:
    """
    LangGraph node — EvaluatorAgent.

    Reads:  state["final_answer"], state["retrieved_chunks"]
    Writes: faithfulness_score, eval_passed

    Runs after GeneratorAgent. Last node before END.
    """
    answer = state.get("final_answer", "")
    chunks = state.get("retrieved_chunks") or []

    if not answer:
        logger.warning("evaluator_no_answer_to_evaluate")
        return {"faithfulness_score": 0.0, "eval_passed": False}

    if not chunks:
        # No context was used — cannot evaluate faithfulness.
        logger.warning("evaluator_no_chunks_for_evaluation")
        return {"faithfulness_score": 0.0, "eval_passed": False}

    logger.info(
        "evaluator_start",
        extra={"answer_length": len(answer), "chunk_count": len(chunks)}
    )

    # GPT-3.5-turbo — cheaper, sufficient for faithfulness scoring.
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = EVALUATOR_PROMPT | llm

    try:
        context = _format_context_brief(chunks)
        response = chain.invoke({"context": context, "answer": answer})
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        score = float(parsed.get("faithfulness_score", 0.0))
        score = max(0.0, min(1.0, score))

        # Phase 4 regression gate threshold: faithfulness must not drop
        # more than 5% below baseline. Here we use a fixed floor of 0.7
        # as the per-query pass/fail threshold.
        # The regression gate in evaluation/regression_gate.py handles
        # the baseline comparison across commits.
        eval_passed = score >= 0.7

        logger.info(
            "evaluator_complete",
            extra={
                "faithfulness_score": score,
                "eval_passed": eval_passed,
                "reasoning": parsed.get("reasoning", ""),
            }
        )

        return {
            "faithfulness_score": score,
            "eval_passed": eval_passed,
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error("evaluator_failed", extra={"error": str(e)})
        return {
            "faithfulness_score": 0.0,
            "eval_passed": False,
            "error": f"EvaluatorAgent failed: {str(e)}",
        }