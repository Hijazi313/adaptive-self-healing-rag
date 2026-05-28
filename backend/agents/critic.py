"""
agents/critic.py — CriticAgent

RESPONSIBILITY:
    Evaluates the quality of retrieved_chunks against the user's query.
    Produces a relevance score 0.0–1.0 and a brief reasoning string.

    This is the core of the self-healing retrieval loop.
    A low score triggers reformulation. A high score unlocks generation.

WHY AN LLM CRITIC (not a cosine similarity threshold):
    Cosine similarity between query embedding and chunk embeddings only
    measures vector proximity — it does not measure whether the chunk
    actually contains information that answers the question.

    Example where cosine similarity is high but relevance is low:
        Query: "How do I fix a database timeout error?"
        Chunk: "Database performance optimization requires careful consideration
                of indexing strategies and query planning."
    This chunk is about databases (high cosine sim) but doesn't help
    fix the timeout error (low actual relevance).

    An LLM critic understands the semantic gap — it reads the query
    and chunks the way a human would and asks: "does this actually help?"

SCORING SCALE:
    0.0–0.4  → poor: chunks are off-topic or only tangentially related
    0.4–0.7  → partial: chunks have related information but miss the core need
    0.7–1.0  → good: chunks directly address the query

    settings.critic_relevance_threshold = 0.7 (configurable in .env)
    Below 0.7 → supervisor triggers reformulation
    At or above 0.7 → GeneratorAgent proceeds

LATENCY NOTE:
    The Critic adds one LLM call to the hot path. This is acceptable because:
    1. It prevents hallucinated answers — much more expensive to handle downstream
    2. It is conditional — Phase 3 adds a semantic cache that bypasses the
       entire retrieval+critic loop for repeated similar queries
    If latency becomes a problem in production: make Critic async or
    only invoke it when QueryAnalyzer confidence is below a threshold.
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

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a retrieval quality evaluator.
Given a user query and retrieved context chunks, score how well the chunks
address the query. Return ONLY valid JSON. No preamble.

Scoring guide:
- 0.0–0.4: chunks are off-topic or only mention related concepts superficially
- 0.4–0.7: chunks contain relevant information but miss the core question
- 0.7–1.0: chunks directly contain information needed to answer the query

Return this exact JSON:
{{
  "score": <float 0.0 to 1.0>,
  "reasoning": "one sentence explaining the score"
}}"""
    ),
    (
        "human",
        "Query: {query}\n\nRetrieved chunks:\n{chunks_text}"
    ),
])


def _format_chunks(chunks: list[Document]) -> str:
    """
    Format retrieved chunks for the critic prompt.
    Truncates each chunk to 400 chars to keep the prompt within token budget.
    The critic needs enough to judge relevance — not the full text.
    """
    if not chunks:
        return "No chunks retrieved."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.page_content[:400]
        if len(chunk.page_content) > 400:
            preview += "..."
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{preview}")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Agent node function
# ─────────────────────────────────────────────────────────────────────────────

def critic_node(state: RAGState) -> dict:
    """
    LangGraph node — CriticAgent.

    Reads:  state["query"], state["retrieved_chunks"]
    Writes: critic_score, critic_reasoning

    If retrieved_chunks is empty, immediately returns score=0.0 without
    an LLM call — empty retrieval is an unambiguous failure.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []

    # Fast path: no chunks retrieved → score 0.0, no LLM call needed.
    if not chunks:
        logger.warning(
            "critic_no_chunks_retrieved",
            extra={"query_preview": query[:100]}
        )
        return {
            "critic_score": 0.0,
            "critic_reasoning": "No chunks were retrieved — retrieval failed entirely.",
        }

    logger.info(
        "critic_start",
        extra={"chunk_count": len(chunks), "query_preview": query[:100]}
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = CRITIC_PROMPT | llm

    try:
        chunks_text = _format_chunks(chunks)
        response = chain.invoke({"query": query, "chunks_text": chunks_text})
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        score = float(parsed.get("score", 0.0))
        # Clamp to [0.0, 1.0] — defensive against LLM returning 1.2 etc.
        score = max(0.0, min(1.0, score))
        reasoning = parsed.get("reasoning", "")

        logger.info(
            "critic_complete",
            extra={
                "critic_score": score,
                "threshold": settings.critic_relevance_threshold,
                "passed": score >= settings.critic_relevance_threshold,
                "reasoning": reasoning,
            }
        )

        return {
            "critic_score": score,
            "critic_reasoning": reasoning,
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error("critic_failed", extra={"error": str(e)})
        # On parse failure, return a low score to trigger reformulation.
        # Do not return 0.0 exactly — that would look like "no chunks".
        return {
            "critic_score": 0.3,
            "critic_reasoning": f"Critic evaluation failed: {str(e)}",
            "error": f"CriticAgent failed: {str(e)}",
        }