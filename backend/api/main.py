"""
api/main.py — FastAPI Application Layer

ROLE OF THIS FILE:
    This is the outermost shell of the system. It does not contain any
    RAG logic. Its only job is to:

    1. Initialize shared resources once at startup (graph, cache, memory)
    2. Accept HTTP requests and validate their shape (Pydantic models)
    3. Orchestrate the three-step flow: memory-read → graph → memory-write
    4. Return structured HTTP responses

    Every piece of actual logic lives in the modules it imports.
    If you find business logic in this file, it belongs somewhere else.

THREE-STEP FLOW PER QUERY:
    ┌─────────────────────────────────────────────────────────┐
    │  POST /query                                            │
    │                                                         │
    │  1. SemanticCache.get(query)                            │
    │        hit  → return cached answer immediately          │
    │        miss → continue                                  │
    │                                                         │
    │  2. UserMemory.get_context(user_id, query)              │
    │        → append to query context (if non-empty)         │
    │                                                         │
    │  3. graph.invoke(initial_state)  [in thread pool]       │
    │        → full RAG pipeline runs                         │
    │                                                         │
    │  4. SemanticCache.set(query, result)                    │
    │  5. UserMemory.store(user_id, query, answer)            │
    │  6. Return QueryResponse                                 │
    └─────────────────────────────────────────────────────────┘

SYNC GRAPH IN ASYNC FASTAPI:
    LangGraph's SqliteSaver is synchronous (sqlite3 is not async-safe).
    We call graph.invoke() via asyncio.run_in_executor(None, ...) which
    runs it in FastAPI's default thread pool without blocking the event loop.
    This is the correct pattern — not a workaround. FastAPI's docs explicitly
    recommend run_in_executor for blocking I/O in async endpoints.

STARTUP / SHUTDOWN:
    We use the lifespan context manager (@asynccontextmanager) — the current
    FastAPI standard. @app.on_event is deprecated since FastAPI 0.95.
    All shared resources (graph, cache, memory) are initialized once at startup
    and stored in app.state. This prevents re-initialization per request.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import settings
from graph.state import initial_state
from graph.supervisor import get_graph
from ingestion.chunkers import get_chunker
from ingestion.embedder import Embedder
from memory.semantic_cache import SemanticCache
from memory.user_memory import user_memory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — initialize once, share via app.state
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.

    WHY app.state:
        FastAPI's app.state is a simple namespace for storing application-level
        objects. Using it means every endpoint accesses the SAME graph instance,
        SAME cache instance — not a new one per request.

        This matters for the graph: SqliteSaver holds a file connection.
        Re-creating it per request would create connection contention.
        Create once, reuse always.
    """
    logger.info("adaptive_rag_api_starting_up")

    # Compile the graph with SqliteSaver checkpointer.
    # This is the most expensive initialization — do it once.
    app.state.graph = get_graph()
    logger.info("graph_initialized")

    app.state.cache = SemanticCache()
    logger.info("semantic_cache_initialized")

    # user_memory is already a module-level singleton from memory/user_memory.py
    # We just reference it here for clarity.
    app.state.user_memory = user_memory
    logger.info("user_memory_initialized")

    logger.info("adaptive_rag_api_ready")
    yield
    # Shutdown — nothing to explicitly close for SqliteSaver or Redis client.
    # Python's garbage collector handles connection cleanup on exit.
    logger.info("adaptive_rag_api_shutting_down")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Adaptive Multi-Agent RAG System",
    description=(
        "Self-healing RAG with hybrid retrieval, critic evaluation, "
        "semantic caching, and user memory."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    user_id: str = Field(default="", description="Optional user ID for memory personalization")
    thread_id: Optional[str] = Field(
        default=None,
        description="Session thread ID. Auto-generated if not provided.",
    )

    model_config = {"json_schema_extra": {"example": {
        "query": "What causes database connection timeouts?",
        "user_id": "user_abc123",
    }}}


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_citations: list[dict]
    critic_score: Optional[float]
    faithfulness_score: Optional[float]
    eval_passed: Optional[bool]
    retrieval_attempts: list[str]
    cache_hit: bool
    thread_id: str
    requires_human_review: bool


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Raw text strings to index")
    source: str = Field(..., description="Document source identifier (filename, URL, etc.)")
    chunk_strategy: str = Field(
        default="recursive",
        description="Chunking strategy: recursive | semantic | proposition",
    )

    model_config = {"json_schema_extra": {"example": {
        "texts": ["PostgreSQL connection pooling allows reuse of database connections..."],
        "source": "postgres_docs.txt",
        "chunk_strategy": "recursive",
    }}}


class IngestResponse(BaseModel):
    indexed_chunks: int
    chunk_strategy: str
    source: str


class HitlResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID of the interrupted run")
    approved: bool = Field(..., description="True to allow one final retrieval attempt, False to end")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main RAG query endpoint.

    Flow:
        1. Semantic cache check — return immediately on hit
        2. User memory context retrieval
        3. Graph execution (in thread pool — graph is sync)
        4. Cache and memory write
        5. Return structured response
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    query_text = request.query.strip()

    # ── Step 1: Semantic cache check ─────────────────────────────────────────
    # This is the most important optimization in the system.
    # A cache hit completely bypasses graph execution — no LLM calls, no Qdrant.
    # At scale, the majority of queries in a knowledge base will be repetitive.
    cached = app.state.cache.get(query_text)
    if cached:
        logger.info(
            "query_served_from_cache",
            extra={"similarity": cached.get("cache_similarity_score"), "query_preview": query_text[:80]}
        )
        return QueryResponse(
            query=query_text,
            answer=cached.get("final_answer", ""),
            source_citations=cached.get("source_citations", []),
            critic_score=None,             # not available from cache
            faithfulness_score=cached.get("faithfulness_score"),
            eval_passed=cached.get("eval_passed"),
            retrieval_attempts=[],
            cache_hit=True,
            thread_id=thread_id,
            requires_human_review=False,
        )

    # ── Step 2: User memory context ──────────────────────────────────────────
    # Retrieve past memories relevant to this query.
    # get_context() returns a formatted string — empty string if no memories yet.
    # We pass it as part of the query enrichment. The QueryAnalyzerAgent can
    # use the enriched context if we add it to the query or state.
    # For now: stored in state as part of the initial context (future: user_context field).
    user_context = ""
    if request.user_id:
        user_context = app.state.user_memory.get_context(
            user_id=request.user_id,
            query=query_text,
        )
        if user_context:
            logger.info(
                "user_memory_context_retrieved",
                extra={"user_id": request.user_id, "context_length": len(user_context)}
            )

    # ── Step 3: Graph execution ───────────────────────────────────────────────
    # graph.invoke() is synchronous — run in thread pool to avoid blocking.
    #
    # Why partial() instead of lambda?
    # partial(fn, arg1, arg2) is pickle-safe. Lambda functions are not.
    # run_in_executor uses a thread pool that may need to serialize the callable
    # in some executor configurations. partial is the safe, explicit choice.
    graph_state = initial_state(
        query=query_text,
        user_id=request.user_id,
        thread_id=thread_id,
    )
    graph_config = {"configurable": {"thread_id": thread_id}}

    loop = asyncio.get_event_loop()
    graph_fn = partial(app.state.graph.invoke, graph_state, graph_config)

    try:
        result = await loop.run_in_executor(None, graph_fn)
    except Exception as e:
        logger.error("graph_invocation_failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph execution failed: {str(e)}",
        )

    # ── Check for HITL interrupt ──────────────────────────────────────────────
    # If the graph reached the human_review_node, it returned an interrupt
    # event rather than a final state. LangGraph signals this differently
    # from a normal result — the result will have requires_human_review=True.
    if result.get("requires_human_review"):
        return QueryResponse(
            query=query_text,
            answer=(
                "This query requires human review before an answer can be generated. "
                f"Use POST /hitl/resume with thread_id='{thread_id}' to approve or reject."
            ),
            source_citations=[],
            critic_score=result.get("critic_score"),
            faithfulness_score=None,
            eval_passed=False,
            retrieval_attempts=result.get("retrieval_attempts", []),
            cache_hit=False,
            thread_id=thread_id,
            requires_human_review=True,
        )

    # ── Step 4: Cache and memory write ───────────────────────────────────────
    # Only cache and store if we got a real answer (not an error, not HITL).
    final_answer = result.get("final_answer", "")

    if final_answer and not result.get("error"):
        # Cache the result for future semantically similar queries.
        app.state.cache.set(query_text, result)

        # Store the Q&A interaction in user long-term memory.
        if request.user_id:
            app.state.user_memory.store(
                user_id=request.user_id,
                query=query_text,
                final_answer=final_answer,
                eval_passed=result.get("eval_passed"),
            )

    # ── Step 5: Return response ───────────────────────────────────────────────
    return QueryResponse(
        query=query_text,
        answer=final_answer or "No answer was generated.",
        source_citations=result.get("source_citations") or [],
        critic_score=result.get("critic_score"),
        faithfulness_score=result.get("faithfulness_score"),
        eval_passed=result.get("eval_passed"),
        retrieval_attempts=result.get("retrieval_attempts") or [],
        cache_hit=False,
        thread_id=thread_id,
        requires_human_review=False,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Index documents into Qdrant.

    Accepts raw text strings, chunks them using the specified strategy,
    embeds them, and upserts into the Qdrant collection.

    After indexing, flushes the semantic cache — any cached answers based
    on the old corpus are now potentially stale.

    WHY flush after ingest:
        If you update a document (e.g., a policy changes), cached answers
        that cited the old document are now wrong. Flushing forces all
        future queries through the full RAG pipeline with fresh retrieval.
        This is the simplest correct invalidation strategy for a document
        corpus that updates infrequently.
    """
    if request.chunk_strategy not in ("recursive", "semantic", "proposition"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid chunk_strategy: '{request.chunk_strategy}'. "
                   f"Must be one of: recursive, semantic, proposition.",
        )

    # Build LangChain Documents from raw text strings.
    # Each text is treated as one source document.
    documents = [
        Document(
            page_content=text,
            metadata={"source": request.source},
        )
        for text in request.texts
        if text.strip()
    ]

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All provided texts were empty.",
        )

    # Chunking and embedding are CPU/network-bound — run in thread pool.
    def _ingest_sync():
        chunker = get_chunker(request.chunk_strategy)
        chunks = chunker.chunk(documents)

        if not chunks:
            raise ValueError("Chunking produced zero chunks from the provided documents.")

        embedder = Embedder()
        return embedder.index(chunks)

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(None, _ingest_sync)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error("ingest_failed", extra={"error": str(e), "source": request.source})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )

    # Flush semantic cache — corpus has changed, cached answers may be stale.
    flushed = app.state.cache.flush()
    logger.info(
        "ingest_complete_cache_flushed",
        extra={
            "source": request.source,
            "chunks": summary["total_chunks"],
            "cache_keys_flushed": flushed,
        }
    )

    return IngestResponse(
        indexed_chunks=summary["total_chunks"],
        chunk_strategy=request.chunk_strategy,
        source=request.source,
    )


@app.post("/hitl/resume")
async def hitl_resume(request: HitlResumeRequest):
    """
    Resume a graph execution that was interrupted at the human_review node.

    HOW THIS WORKS:
        1. The interrupted run is identified by thread_id.
        2. The checkpointer (SqliteSaver) has the full graph state saved.
        3. We call graph.invoke(Command(resume=approved), config=thread_config).
        4. LangGraph restores state at the interrupt point and continues.
        5. If approved: graph runs one more retrieval attempt then generates.
        6. If rejected: graph ends with a user-facing failure message.

    The caller must use the SAME thread_id from the original interrupted run.
    """
    thread_config = {"configurable": {"thread_id": request.thread_id}}
    command = Command(resume=request.approved)

    loop = asyncio.get_event_loop()
    resume_fn = partial(app.state.graph.invoke, command, thread_config)

    try:
        result = await loop.run_in_executor(None, resume_fn)
    except Exception as e:
        logger.error(
            "hitl_resume_failed",
            extra={"thread_id": request.thread_id, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HITL resume failed: {str(e)}",
        )

    return {
        "thread_id": request.thread_id,
        "approved": request.approved,
        "answer": result.get("final_answer", "No answer generated after human review."),
        "eval_passed": result.get("eval_passed"),
        "retrieval_attempts": result.get("retrieval_attempts", []),
    }


@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """
    Return all stored memories for a user.
    Useful for debugging personalization and auditing stored facts.
    """
    memories = app.state.user_memory.get_all(user_id=user_id)
    return {"user_id": user_id, "memory_count": len(memories), "memories": memories}


@app.delete("/memory/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user_memory(user_id: str):
    """
    Delete all memories for a user (GDPR compliance).
    Irreversible — use with caution.
    """
    success = app.state.user_memory.delete_all(user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user memories.",
        )
    return {"user_id": user_id, "status": "deleted"}


@app.post("/admin/cache/flush")
async def flush_cache():
    """
    Manually flush the semantic cache.

    Use this after re-indexing documents to ensure stale cached answers
    are evicted. The /ingest endpoint does this automatically, but this
    endpoint allows manual flushing when documents are updated via other means.
    """
    deleted = app.state.cache.flush()
    return {"status": "flushed", "keys_deleted": deleted}


@app.get("/admin/cache/stats")
async def cache_stats():
    """Return semantic cache statistics."""
    return app.state.cache.stats()


@app.get("/health")
async def health():
    """
    Infrastructure health check.

    Checks: Qdrant reachability, Redis reachability.
    Returns 200 if all healthy, 503 if any service is down.

    Used by Docker HEALTHCHECK, load balancers, and monitoring systems.
    A health endpoint that just returns 200 regardless of infra state
    is worse than no health endpoint — it lies to your orchestrator.
    """
    from qdrant_client import QdrantClient
    from ingestion.indexer import get_qdrant_client
    import redis as redis_lib

    health_status = {"qdrant": "unknown", "redis": "unknown"}
    all_healthy = True

    # Qdrant check
    try:
        client: QdrantClient = get_qdrant_client()
        client.get_collections()
        health_status["qdrant"] = "ok"
    except Exception as e:
        health_status["qdrant"] = f"error: {str(e)}"
        all_healthy = False

    # Redis check
    try:
        r = redis_lib.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            socket_connect_timeout=2,
        )
        r.ping()
        health_status["redis"] = "ok"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
        all_healthy = False

    if not all_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "services": health_status},
        )

    return {"status": "healthy", "services": health_status}