"""

RESPONSIBILITY:
    Executes hybrid search against Qdrant using the query and alpha weight
    set by QueryAnalyzerAgent. Returns the top-K chunks and appends the
    current retrieval strategy to the attempts log.

HYBRID SEARCH MECHANICS:
    Qdrant's Query API runs dense and sparse search in parallel, then merges
    results using Reciprocal Rank Fusion (RRF).

    RRF score for a document d:
        RRF(d) = Σ 1 / (k + rank_i(d))
        where k=60 (standard constant), rank_i is the rank in each sub-result.

    The dense_weight (alpha) from QueryAnalyzerAgent shifts which signal
    dominates. We pass this to QdrantVectorStore's search_kwargs.

    On first attempt:   uses state["rewritten_query"]
    On retry attempts:  uses state["rewritten_query"] (already reformulated
                        by reformulation.py before this node is called again)

WHAT THIS NODE DOES NOT DO:
    - Does not score results (that's CriticAgent)
    - Does not rerank (future enhancement — not in scope)
    - Does not call the LLM

INTEGRATION WITH EMBEDDER:
    The Embedder built in Day 3 exposes get_vector_store().
    RetrieverAgent calls this to get the configured QdrantVectorStore
    and invokes similarity_search_with_score() directly.
    No separate client setup — one source of truth.
"""

import logging
from langchain_qdrant import RetrievalMode

from config import settings
from graph.state import RAGState
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


def retriever_node(state: RAGState) -> dict:
    """
    LangGraph node — RetrieverAgent.

    Reads:  state["rewritten_query"], state["dense_weight"],
            state["retrieval_strategy"]
    Writes: retrieved_chunks, retrieval_attempts (appended via reducer)

    The retrieval_attempts list uses operator.add reducer in state —
    we return a single-element list here; LangGraph appends it to
    the existing list automatically. This is the correct reducer pattern.
    """
    query = state.get("rewritten_query") or state["query"]
    dense_weight = state.get("dense_weight") or 0.55
    strategy = state.get("retrieval_strategy") or "dense_sparse_hybrid"

    logger.info(
        "retriever_start",
        extra={
            "query_preview": query[:100],
            "dense_weight": dense_weight,
            "strategy": strategy,
            "top_k": settings.retrieval_top_k,
        }
    )

    try:
        embedder = Embedder()
        vector_store = embedder.get_vector_store()

        # QdrantVectorStore.similarity_search accepts search_kwargs for
        # passing retrieval-mode-specific parameters.
        # score_threshold=0.0 ensures we always get top_k results
        # even for low-similarity queries — the CriticAgent judges quality.
        chunks = vector_store.similarity_search(
            query=query,
            k=settings.retrieval_top_k,
            # search_kwargs={
            #     "score_threshold": 0.0,
            # },
        )

        logger.info(
            "retriever_complete",
            extra={
                "chunks_retrieved": len(chunks),
                "strategy": strategy,
            }
        )

        return {
            "retrieved_chunks": chunks,
            # Append this attempt to the log (operator.add reducer handles merge).
            "retrieval_attempts": [strategy],
        }

    except Exception as e:
        logger.error(
            "retriever_failed",
            extra={"error": str(e), "query_preview": query[:100]}
        )
        return {
            "retrieved_chunks": [],
            "retrieval_attempts": [strategy],
            "error": f"RetrieverAgent failed: {str(e)}",
        }