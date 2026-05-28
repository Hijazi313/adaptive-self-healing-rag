"""
memory/user_memory.py — User Long-Term Memory via mem0

WHAT THIS IS:
    Cross-session memory for individual users. Remembers what topics they've
    asked about, what they found useful, and their stated preferences.
    This context enriches query processing across separate conversations.

WHY mem0:
    mem0 handles the full memory lifecycle automatically:
    - Fact extraction from conversations (LLM-powered)
    - Deduplication of similar memories
    - Semantic search over stored memories
    - Conflict resolution (if a user says something contradictory)

    You don't need to engineer this — mem0 does it. Your job is to call
    .add() after a conversation and .search() before one.

HOW IT INTEGRATES WITH THE GRAPH:
    This module is called by the FastAPI layer AROUND graph.invoke() —
    not inside any graph node.

    BEFORE graph.invoke():
        memories = user_memory.get_context(user_id, query)
        # → string of relevant past memory facts
        # → passed into initial_state as user_context (if we add that field)
        # OR used by the API layer to enrich the system prompt

    AFTER graph.invoke():
        user_memory.store(user_id, query, final_answer)
        # → mem0 extracts facts from the Q&A and stores them

    WHY NOT INSIDE A GRAPH NODE:
        Memory operations are I/O-bound and add latency.
        They don't affect routing decisions in the graph.
        Keeping them outside the graph keeps the graph logic pure and testable.
        The FastAPI layer can make memory calls async while the graph is sync.

MEM0 MODE — self-hosted (Memory class):
    We use the open-source Memory class, not the cloud MemoryClient.
    Reasons:
    1. No API key required for local development
    2. Data stays local — relevant for enterprise/sensitive use cases
    3. Your mem0_api_key in .env is kept for the managed platform upgrade path

    mem0's Memory class by default uses:
        - OpenAI gpt-4.1-nano for fact extraction (cheap, fast)
        - text-embedding-3-small for memory embeddings
        - Local Qdrant at /tmp/qdrant for vector storage
        - SQLite at ~/.mem0/history.db for history

    IMPORTANT: mem0 spins up its OWN local Qdrant instance at /tmp/qdrant.
    This is separate from your main Qdrant instance on port 6333.
    They do not conflict — mem0's Qdrant is internal to mem0's data.

WHAT GETS STORED:
    After each Q&A:
        - The topic the user asked about
        - Whether they received a useful answer (eval_passed)
        - Any preferences or context they mentioned in the query

    Over time, mem0 builds a user profile like:
        "User frequently asks about PostgreSQL performance"
        "User prefers concise technical answers"
        "User has asked about connection pooling before"

    This context is retrieved and can be prepended to future queries or
    used to bias retrieval strategy selection in QueryAnalyzerAgent.
"""

import logging
from typing import Optional

from mem0 import Memory

from config import settings

logger = logging.getLogger(__name__)


class UserMemory:
    """
    Wrapper around mem0's Memory class for user long-term memory.

    Single instance per application (module-level singleton pattern,
    same as `settings`). mem0's Memory class manages its own connection
    pool and vector store — instantiate once, reuse everywhere.
    """

    def __init__(self) -> None:
        """
        Initialize mem0 Memory with OpenAI configuration.

        mem0 needs an LLM for fact extraction. We configure it to use
        gpt-4o-mini — cheap and fast for extraction, not used for answers.
        We also align the embedding model with what's available in settings
        for consistency in the overall system.
        """
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    # gpt-4o-mini: correct choice for fact extraction.
                    # It's the mem0 use case: extract structured facts from
                    # conversational text. Full gpt-4o is overkill here.
                    "api_key": settings.openai_api_key,
                    "temperature": 0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    # text-embedding-3-small for mem0's internal vectors.
                    # Not text-embedding-3-large — mem0 stores many small
                    # memory facts; smaller dimensions = faster memory search.
                    # Consistency with the RAG corpus embedding is not required
                    # here because mem0 has its own separate vector store.
                    "api_key": settings.openai_api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    # mem0's own Qdrant instance — separate from port 6333.
                    "host": "localhost",
                    "port": 6335,
                    # Port 6335 avoids collision with the main Qdrant (6333).
                    # mem0 will start its own embedded Qdrant if this port
                    # isn't running a server — or you can add qdrant:6335
                    # to docker-compose.yml for a fully managed setup.
                    # For local dev: mem0 manages its own storage at /tmp/qdrant.
                    "collection_name": "adaptive_rag_user_memory",
                },
            },
        }

        try:
            self._memory = Memory.from_config(config)
            logger.info("user_memory_initialized")
        except Exception as e:
            # mem0 init failure is non-fatal — the system degrades gracefully.
            # Queries still work; they just don't have user memory context.
            logger.warning(
                "user_memory_init_failed_degrading_gracefully",
                extra={"error": str(e)}
            )
            self._memory = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def get_context(self, user_id: str, query: str, limit: int = 5) -> str:
        """
        Retrieve relevant memories for a user given the current query.

        Returns:
            A formatted string of memory facts, or empty string if none found
            or if mem0 is unavailable.

            Example output:
                "- User frequently asks about PostgreSQL performance tuning.
                 - User has previously asked about connection pooling.
                 - User prefers technical, concise answers."

        Used by the FastAPI layer to optionally prepend this context to the
        query before invoking the graph, or to pass it as additional context
        to the QueryAnalyzerAgent (future enhancement: add user_context field
        to RAGState and have QueryAnalyzer read it).
        """
        if not self._memory or not user_id:
            return ""

        try:
            results = self._memory.search(
                query=query,
                user_id=user_id,
                limit=limit,
            )

            memories = results.get("results", [])
            if not memories:
                return ""

            # Format as a bullet list for readability in prompts.
            formatted = "\n".join(
                f"- {m['memory']}" for m in memories if m.get("memory")
            )

            logger.debug(
                "user_memory_retrieved",
                extra={
                    "user_id": user_id,
                    "memory_count": len(memories),
                    "query_preview": query[:80],
                }
            )

            return formatted

        except Exception as e:
            logger.warning(
                "user_memory_get_error",
                extra={"user_id": user_id, "error": str(e)}
            )
            return ""

    def store(
        self,
        user_id: str,
        query: str,
        final_answer: str,
        eval_passed: Optional[bool] = None,
    ) -> bool:
        """
        Store a Q&A interaction as user memory.

        mem0 extracts facts from the conversation messages and stores
        them as structured memory entries for this user_id.

        Args:
            user_id:      The user identifier. Must be consistent across sessions.
            query:        What the user asked.
            final_answer: What the system answered.
            eval_passed:  Whether the answer passed faithfulness evaluation.
                          Stored as context — a failed answer is still useful
                          for mem0 to remember what the user was asking about.

        Returns:
            True if stored, False on failure (non-fatal).
        """
        if not self._memory or not user_id:
            return False

        # Build conversation messages for mem0's fact extraction.
        # mem0 reads the conversation and extracts semantic facts.
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": final_answer},
        ]

        try:
            self._memory.add(
                messages=messages,
                user_id=user_id,
                metadata={
                    # metadata is stored alongside the memory for filtering.
                    # eval_passed lets you later filter for only good answers.
                    "eval_passed": eval_passed,
                    "source": "adaptive_rag_system",
                },
            )

            logger.info(
                "user_memory_stored",
                extra={
                    "user_id": user_id,
                    "query_preview": query[:80],
                    "eval_passed": eval_passed,
                }
            )
            return True

        except Exception as e:
            logger.warning(
                "user_memory_store_error",
                extra={"user_id": user_id, "error": str(e)}
            )
            return False

    def get_all(self, user_id: str) -> list[dict]:
        """
        Return all stored memories for a user.
        Used by the Phase 5 /memory/{user_id} API endpoint for inspection.
        """
        if not self._memory or not user_id:
            return []

        try:
            results = self._memory.get_all(user_id=user_id)
            return results.get("results", [])
        except Exception as e:
            logger.warning(
                "user_memory_get_all_error",
                extra={"user_id": user_id, "error": str(e)}
            )
            return []

    def delete_all(self, user_id: str) -> bool:
        """
        Delete all memories for a user. For GDPR compliance / user data deletion.
        Exposed via Phase 5 DELETE /memory/{user_id} endpoint.
        """
        if not self._memory or not user_id:
            return False

        try:
            self._memory.delete_all(user_id=user_id)
            logger.info("user_memory_deleted_all", extra={"user_id": user_id})
            return True
        except Exception as e:
            logger.warning(
                "user_memory_delete_error",
                extra={"user_id": user_id, "error": str(e)}
            )
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

# Same pattern as `settings` in config.py.
# Import `user_memory` everywhere — don't instantiate UserMemory() per request.
user_memory = UserMemory()