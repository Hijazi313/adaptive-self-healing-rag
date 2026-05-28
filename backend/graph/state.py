"""
graph/state.py — Shared State Contract for the Adaptive RAG Graph

THIS IS THE MOST IMPORTANT FILE IN THE SYSTEM.

Every agent reads from this state. Every agent writes a partial update back.
The supervisor routes based on fields in this state. The evaluation harness
reads the final state after the graph completes.

If this schema is wrong, every other file is wrong downstream.

DESIGN PRINCIPLES APPLIED:
    1. TypedDict, not Pydantic BaseModel
       LangGraph is designed around TypedDict. Pydantic adds runtime validation
       on graph inputs but NOT on subsequent nodes. That partial protection at
       the cost of Pydantic's recursive validation overhead is not worth it here.
       TypedDict gives full static analysis via mypy/pyright with zero runtime cost.

    2. Minimal — only what must survive node transitions
       Transient values (prompts, intermediate vectors, temp variables) stay
       inside node functions. If a field isn't read by at least two different
       agents, it probably doesn't belong here.

    3. Reducers only where accumulation is needed
       operator.add for lists that grow (retrieval_attempts log, eval_scores).
       Plain overwrite for scalar fields that are replaced each time.

    4. Optional fields use explicit None defaults
       Every field that isn't set at graph entry must have a default.
       LangGraph requires all state fields to be initializable at START.

FLOW THIS STATE SUPPORTS:
    START
      → QueryAnalyzerAgent   (sets: query_type, retrieval_strategy, rewritten_query)
      → RetrieverAgent       (sets: retrieved_chunks)
      → CriticAgent          (sets: critic_score, critic_reasoning)
      → [conditional edge]
            score >= threshold → GeneratorAgent
            score <  threshold AND retries < max → reformulate → RetrieverAgent
            retries == max → interrupt() → human review
      → GeneratorAgent       (sets: final_answer, source_citations)
      → EvaluatorAgent       (sets: faithfulness_score, eval_passed)
      → END
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """
    Shared state for the Adaptive RAG graph.

    total=False means all fields are optional at the TypedDict level.
    Each agent only touches the fields it owns.
    The graph's initial input only needs to provide `query` and `user_id`.

    Fields are grouped by which agent writes them.
    """

    # ── INPUT (provided by caller at graph entry) ─────────────────────────────

    query: str
    # The original user question. Never mutated after entry.
    # Every agent can read this — the CriticAgent compares chunks against it,
    # the GeneratorAgent uses it to frame the answer.

    user_id: str
    # Identifies the user for mem0 long-term memory lookup (Phase 3).
    # Optional — anonymous queries are valid.

    thread_id: str
    # LangGraph checkpointer key. Set by the caller (FastAPI layer).
    # Required for HITL interrupt/resume and session memory.

    # ── QUERY ANALYZER OUTPUT ─────────────────────────────────────────────────

    query_type: Optional[str]
    # One of: "keyword" | "conceptual" | "hybrid"
    # Determines the dense/sparse alpha weight in RetrieverAgent.
    # "keyword"    → weight sparse (BM25) higher — exact term matching
    # "conceptual" → weight dense higher — semantic similarity
    # "hybrid"     → balanced

    retrieval_strategy: Optional[str]
    # First-attempt strategy: always "dense_sparse_hybrid"
    # On retries, set by reformulation.py: "hyde" | "step_back" | "decompose"

    rewritten_query: Optional[str]
    # The query actually sent to the retriever.
    # On first attempt: same as `query` (possibly lightly cleaned).
    # On retries: the reformulated version (HyDE doc, step-back question, etc.)

    dense_weight: Optional[float]
    # Alpha for hybrid search fusion: 0.0 = full sparse, 1.0 = full dense.
    # Set by QueryAnalyzerAgent based on query_type.
    # RetrieverAgent reads this to configure the search call.

    # ── RETRIEVER OUTPUT ──────────────────────────────────────────────────────

    retrieved_chunks: Optional[list[Document]]
    # The top-K chunks returned by hybrid search.
    # CriticAgent scores these. GeneratorAgent synthesises from these.
    # Not annotated with a reducer — replaced entirely on each retrieval attempt.

    retrieval_attempts: Annotated[list[str], operator.add]
    # Append-only log of strategies attempted so far.
    # e.g. ["original", "hyde", "step_back"]
    # The supervisor reads len(retrieval_attempts) to enforce max retries.
    # Using operator.add reducer — this list grows across retry cycles.

    # ── CRITIC OUTPUT ─────────────────────────────────────────────────────────

    critic_score: Optional[float]
    # Relevance score 0.0–1.0 for retrieved_chunks against the query.
    # Below settings.critic_relevance_threshold → trigger reformulation.
    # Above threshold → proceed to GeneratorAgent.

    critic_reasoning: Optional[str]
    # Brief explanation from CriticAgent of why the score was assigned.
    # Stored for LangSmith tracing and HITL display — not used for routing.

    # ── GENERATOR OUTPUT ──────────────────────────────────────────────────────

    final_answer: Optional[str]
    # The grounded answer produced by GeneratorAgent.
    # Only set once critic_score >= threshold.

    source_citations: Optional[list[dict]]
    # List of source references used in final_answer.
    # Each entry: {"source": str, "chunk_index": int, "excerpt": str}
    # Used by the API layer to return structured citations to the caller.

    # ── EVALUATOR OUTPUT ──────────────────────────────────────────────────────

    faithfulness_score: Optional[float]
    # Post-generation RAGAS faithfulness: does the answer stay within context?
    # 0.0–1.0. Written by EvaluatorAgent after GeneratorAgent completes.

    eval_passed: Optional[bool]
    # True if faithfulness_score >= threshold (Phase 4 regression gate).
    # The FastAPI layer can surface this to the caller for transparency.

    # ── CONTROL FLOW ─────────────────────────────────────────────────────────

    error: Optional[str]
    # Set if any agent hits an unrecoverable error.
    # The supervisor routes to END when this is non-None, skipping remaining agents.
    # Never raises exceptions through the graph — always surfaces via this field.

    requires_human_review: Optional[bool]
    # Set to True by the supervisor when retrieval_attempts reaches max retries
    # AND critic_score is still below threshold.
    # Triggers LangGraph interrupt() in the supervisor (Phase 5 / HITL).


def initial_state(query: str, user_id: str = "", thread_id: str = "") -> RAGState:
    """
    Factory for a valid initial state at graph entry.

    WHY THIS EXISTS:
        LangGraph requires all Annotated fields with reducers to have an
        initial value when the graph starts. The `retrieval_attempts` field
        uses operator.add — if it's missing at entry, the reducer has nothing
        to add to and raises a TypeError.

        Rather than requiring callers to know this, they call initial_state()
        and get a correctly initialised dict.

    USAGE (in supervisor.py and FastAPI layer):
        state = initial_state(query="What causes customer churn?", user_id="u_123")
        result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    """
    return RAGState(
        query=query,
        user_id=user_id,
        thread_id=thread_id,
        query_type=None,
        retrieval_strategy=None,
        rewritten_query=None,
        dense_weight=None,
        retrieved_chunks=None,
        retrieval_attempts=[],      # reducer field — must start as empty list
        critic_score=None,
        critic_reasoning=None,
        final_answer=None,
        source_citations=None,
        faithfulness_score=None,
        eval_passed=None,
        error=None,
        requires_human_review=False,
    )