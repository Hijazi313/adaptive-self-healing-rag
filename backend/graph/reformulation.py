"""
graph/reformulation.py — Query Reformulation Strategies

RESPONSIBILITY:
    When CriticAgent scores retrieved context below threshold, the supervisor
    routes here before retrying retrieval. This node selects the next
    reformulation strategy based on how many attempts have been made,
    produces a reformulated query, and writes it back to state.

    The retriever then uses the reformulated query on the next attempt.

THREE STRATEGIES — APPLIED IN ORDER:

    Attempt 1 (original failed)  → HyDE
    Attempt 2 (HyDE failed)      → Step-Back
    Attempt 3 (Step-Back failed) → Decompose → HITL if still failing

    This ordering is deliberate:
    - HyDE is cheapest (one generation call, no structural change to query)
    - Step-Back is medium cost (one generation, abstracts the query)
    - Decompose is most expensive (multiple sub-queries, multiple retrieval calls)
      but also the most powerful for multi-intent queries

─────────────────────────────────────────────────────────────────────────────
STRATEGY 1 — HyDE (Hypothetical Document Embedding)
    Paper: Gao et al., 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496

    Problem it solves:
        Short queries produce poor embeddings. "database timeout fix" embeds
        to a point in vector space that may not align with the long, rich
        answer documents in your corpus — the embedding spaces differ in density.

    How it works:
        Generate a hypothetical document (a plausible answer) for the query.
        Embed THAT document instead of the query.
        The hypothetical answer's embedding is much closer to real answer
        chunks in vector space than the original query's embedding.

    When it works best:
        Short, conceptual queries where the query embedding is sparse/weak.

─────────────────────────────────────────────────────────────────────────────
STRATEGY 2 — Step-Back Prompting
    Paper: Zheng et al., 2023 "Take a Step Back: Evoking Reasoning via Abstraction"
    https://arxiv.org/abs/2310.06117

    Problem it solves:
        Over-specific queries don't match chunks that answer the general case.
        "Why does my Postgres connection timeout after 30 seconds on AWS RDS?"
        is too specific — the corpus likely has "PostgreSQL connection timeout
        causes and solutions" which answers it but doesn't match the specific
        AWS RDS framing.

    How it works:
        Ask the LLM to generate a more general, abstract version of the query.
        "What causes PostgreSQL connection timeouts?"
        Retrieve against the abstracted query — broader match surface.

    When it works best:
        Queries that are too specific or contain irrelevant specifics.

─────────────────────────────────────────────────────────────────────────────
STRATEGY 3 — Decomposition
    Problem it solves:
        Multi-intent queries where no single chunk addresses all aspects.
        "What are the causes and solutions for database connection pooling issues?"
        requires both "causes" chunks and "solutions" chunks — a single
        retrieval with the full query likely misses one or the other.

    How it works:
        Split the query into atomic sub-questions.
        Retrieve top-K for each sub-question.
        Deduplicate and merge the combined chunk sets.
        This is the most expensive strategy but handles multi-part queries.

    When it works best:
        Compound questions, "compare X and Y", "list causes and solutions" queries.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from graph.state import RAGState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM — shared across all three strategies
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: HyDE
# ─────────────────────────────────────────────────────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Generate a concise hypothetical document passage that would directly answer
the given question. Write it as if it were extracted from an expert reference document.
2-4 sentences only. No preamble, no explanation."""
    ),
    ("human", "{query}"),
])


def _apply_hyde(query: str) -> str:
    chain = HYDE_PROMPT | _get_llm()
    response = chain.invoke({"query": query})
    hypothetical_doc = response.content.strip()
    logger.info("reformulation_hyde_applied", extra={"query_preview": query[:80]})
    return hypothetical_doc


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Step-Back
# ─────────────────────────────────────────────────────────────────────────────

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the given question as a more general, abstract version that captures
the underlying concept without specific details that may not appear in reference documents.
Return only the rewritten question. No preamble."""
    ),
    ("human", "{query}"),
])


def _apply_step_back(query: str) -> str:
    chain = STEP_BACK_PROMPT | _get_llm()
    response = chain.invoke({"query": query})
    abstracted = response.content.strip()
    logger.info("reformulation_step_back_applied", extra={"query_preview": query[:80]})
    return abstracted


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Decompose
# ─────────────────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Break the given question into 2-3 simpler, atomic sub-questions.
Each sub-question should be independently answerable.
Return each sub-question on a new line. No numbering, no preamble."""
    ),
    ("human", "{query}"),
])


def _apply_decompose(query: str) -> str:
    """
    Decompose returns multiple sub-questions joined as a single string.
    The retriever receives this as its query — Qdrant will embed the full
    string and retrieve based on the combined semantic content.

    A more advanced implementation (Phase 2 extension) would retrieve
    separately for each sub-question and merge the result sets.
    For this portfolio project, the single-pass approach is correct:
    it broadens the retrieval surface without requiring graph refactoring.
    """
    chain = DECOMPOSE_PROMPT | _get_llm()
    response = chain.invoke({"query": query})
    sub_questions = response.content.strip()

    # Join on space — the combined embedding captures all sub-intents.
    combined = " ".join(
        line.strip() for line in sub_questions.splitlines() if line.strip()
    )

    logger.info(
        "reformulation_decompose_applied",
        extra={
            "original_query": query[:80],
            "sub_questions": sub_questions[:200],
        }
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Strategy selector — maps attempt count to strategy
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_MAP = {
    1: ("hyde", _apply_hyde),
    2: ("step_back", _apply_step_back),
    3: ("decompose", _apply_decompose),
}


# ─────────────────────────────────────────────────────────────────────────────
# Node function
# ─────────────────────────────────────────────────────────────────────────────

def reformulation_node(state: RAGState) -> dict:
    """
    LangGraph node — Reformulation.

    Reads:  state["query"], state["retrieval_attempts"]
    Writes: rewritten_query, retrieval_strategy

    Called by the supervisor when critic_score < threshold AND
    len(retrieval_attempts) < max_retries.

    The attempt number determines which strategy is applied:
        attempts = 1 → strategy 1 (HyDE)
        attempts = 2 → strategy 2 (Step-Back)
        attempts = 3 → strategy 3 (Decompose)

    After this node, the supervisor routes back to RetrieverAgent.
    The RetrieverAgent reads state["rewritten_query"] — which is now the
    reformulated version — and retries hybrid search.
    """
    original_query = state["query"]
    attempts_so_far = len(state.get("retrieval_attempts") or [])

    # attempts_so_far is the number of retrieval attempts completed.
    # The NEXT attempt number determines which strategy to apply.
    next_attempt = attempts_so_far + 1

    strategy_name, strategy_fn = _STRATEGY_MAP.get(
        next_attempt,
        ("decompose", _apply_decompose),  # fallback: decompose for attempt > 3
    )

    logger.info(
        "reformulation_start",
        extra={
            "attempt": next_attempt,
            "strategy": strategy_name,
            "original_query": original_query[:80],
        }
    )

    try:
        reformulated = strategy_fn(original_query)

        logger.info(
            "reformulation_complete",
            extra={
                "strategy": strategy_name,
                "reformulated_preview": reformulated[:100],
            }
        )

        return {
            "rewritten_query": reformulated,
            "retrieval_strategy": strategy_name,
        }

    except Exception as e:
        logger.error(
            "reformulation_failed",
            extra={"strategy": strategy_name, "error": str(e)}
        )
        # On failure: fall back to original query with next strategy label.
        # This prevents the loop from being stuck — it still increments
        # retrieval_attempts on the next retriever call.
        return {
            "rewritten_query": original_query,
            "retrieval_strategy": strategy_name,
            "error": f"Reformulation ({strategy_name}) failed: {str(e)}",
        }