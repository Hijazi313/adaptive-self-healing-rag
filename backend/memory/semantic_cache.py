"""
memory/semantic_cache.py — Semantic Cache

WHAT THIS IS:
    A cache layer that sits in front of the full RAG graph.
    Before hitting the vector store, we check if we've answered a
    semantically similar query recently. If yes — return cached answer.
    If no — run the full graph, then store the result.

WHY SEMANTIC, NOT EXACT-MATCH:
    A key-value cache keyed on query string misses:
        "What is database connection pooling?"
        "Explain database connection pools"
        "How do connection pools work in databases?"
    These are the same question. Exact-match treats them as three cache misses.
    Semantic cache embeds the query and compares cosine similarity against
    stored query embeddings — all three hit the same cached answer.

HOW IT WORKS:
    ON QUERY:
        1. Embed the incoming query (text-embedding-3-large)
        2. Scan all cached query embeddings in Redis (KEYS pattern)
        3. Compute cosine similarity against each
        4. If max similarity >= threshold (0.92) → return cached answer (cache hit)
        5. If below threshold → cache miss → run full graph

    ON ANSWER (after graph completes):
        1. Store the query embedding in Redis as a float list (JSON)
        2. Store the full answer payload under a linked key
        3. Both keys share the same TTL (default 1 hour)

CACHE INVALIDATION STRATEGY:
    When source documents update, cached answers derived from them become stale.
    Two mechanisms handle this:

    1. TTL-based expiry (default 3600s) — all cached answers expire automatically.
       Tune TTL in .env: shorter for frequently updated corpora, longer for static.

    2. Manual flush: semantic_cache.flush() clears all cache entries.
       Call this from the indexer after a document re-index. The Phase 5 FastAPI
       layer exposes a /admin/cache/flush endpoint protected by an internal key.

STORAGE LAYOUT IN REDIS:
    Key: "semcache:emb:{cache_id}"  → JSON array of floats (embedding vector)
    Key: "semcache:ans:{cache_id}"  → JSON object (full answer payload)

    cache_id = first 16 chars of SHA256(query) — short, unique, human-debuggable.
    Using SHA256 prefix as ID is safe at cache scale (millions of entries before
    collision risk). Not used for security — only for key namespacing.

PERFORMANCE NOTE:
    Scanning all keys on every query is O(n) in the number of cached entries.
    For a portfolio system with hundreds of cached queries, this is negligible.
    At production scale (millions of entries): switch to a Redis vector index
    (RedisSearch with HNSW) which gives O(log n) approximate nearest-neighbor.
    That's a production upgrade path, not a Day 6 requirement.
"""

import json
import hashlib
import logging
from typing import Optional

import numpy as np
import redis

from config import settings
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

# Redis key prefixes — consistent namespacing prevents key collisions
# when Redis is shared with other applications (or with LangGraph checkpointer).
_EMB_PREFIX = "semcache:emb:"
_ANS_PREFIX = "semcache:ans:"


class SemanticCache:
    """
    Redis-backed semantic similarity cache.

    Usage pattern (in the FastAPI layer, Phase 5):

        cache = SemanticCache()

        # Before graph.invoke():
        hit = cache.get(query)
        if hit:
            return hit  # short-circuit: no graph execution

        # After graph.invoke():
        cache.set(query, answer_payload)
    """

    def __init__(self) -> None:
        self._redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            # decode_responses=True: Redis returns str not bytes.
            # All our values are JSON strings — this is correct.
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self._threshold = settings.semantic_cache_similarity_threshold
        self._ttl = settings.semantic_cache_ttl_seconds

        # Embedder for query embedding — reuse the same model as indexing.
        # Consistent embedding model = comparable vector spaces.
        self._embedder = Embedder()

        logger.debug(
            "semantic_cache_initialized",
            extra={
                "threshold": self._threshold,
                "ttl_seconds": self._ttl,
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string using the system's dense embedding model."""
        vector_store = self._embedder.get_vector_store()
        # Access the underlying OpenAI embedder from the vector store.
        embeddings = self._embedder._dense_embeddings
        return embeddings.embed_query(query)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    @staticmethod
    def _cache_id(query: str) -> str:
        """Short, deterministic ID from query content. Used for Redis key suffix."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _is_redis_available(self) -> bool:
        """Ping Redis — return False (not raise) if unreachable. Cache is non-critical."""
        try:
            self._redis.ping()
            return True
        except Exception:
            logger.warning("semantic_cache_redis_unavailable_bypassing_cache")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[dict]:
        """
        Look up a semantically similar cached answer.

        Returns:
            dict with keys: final_answer, source_citations, faithfulness_score,
                            cached_query (the original query that was cached),
                            cache_hit=True
            None if no sufficiently similar cached entry found.

        This method is designed to never raise — cache failure is a miss,
        not an error. The graph runs normally if Redis is down.
        """
        if not self._is_redis_available():
            return None

        try:
            query_embedding = self._embed_query(query)

            # Scan all cached embedding keys.
            emb_keys = list(self._redis.scan_iter(match=f"{_EMB_PREFIX}*"))

            if not emb_keys:
                logger.debug("semantic_cache_empty")
                return None

            best_score = 0.0
            best_cache_id = None

            for key in emb_keys:
                raw = self._redis.get(key)
                if not raw:
                    continue
                cached_embedding = json.loads(raw)
                score = self._cosine_similarity(query_embedding, cached_embedding)
                if score > best_score:
                    best_score = score
                    # Extract cache_id from key: "semcache:emb:{cache_id}"
                    best_cache_id = key[len(_EMB_PREFIX):]

            if best_score >= self._threshold and best_cache_id:
                ans_raw = self._redis.get(f"{_ANS_PREFIX}{best_cache_id}")
                if ans_raw:
                    payload = json.loads(ans_raw)
                    payload["cache_hit"] = True
                    payload["cache_similarity_score"] = round(best_score, 4)

                    logger.info(
                        "semantic_cache_hit",
                        extra={
                            "similarity": round(best_score, 4),
                            "threshold": self._threshold,
                            "query_preview": query[:80],
                        }
                    )
                    return payload

            logger.debug(
                "semantic_cache_miss",
                extra={
                    "best_score": round(best_score, 4),
                    "threshold": self._threshold,
                }
            )
            return None

        except Exception as e:
            logger.warning(
                "semantic_cache_get_error",
                extra={"error": str(e), "query_preview": query[:80]}
            )
            return None

    def set(self, query: str, answer_payload: dict) -> bool:
        """
        Store a query-answer pair in the cache.

        Args:
            query:          The original user query string.
            answer_payload: Dict to cache. Must contain at minimum:
                            final_answer, source_citations.
                            Typically the full RAGState output dict.

        Returns:
            True if stored successfully, False on failure.

        What is stored:
            - Embedding of the query (for future similarity comparisons)
            - The answer payload (what we return on a cache hit)
        """
        if not self._is_redis_available():
            return False

        try:
            cache_id = self._cache_id(query)
            query_embedding = self._embed_query(query)

            # Store embedding.
            self._redis.setex(
                name=f"{_EMB_PREFIX}{cache_id}",
                time=self._ttl,
                value=json.dumps(query_embedding),
            )

            # Store answer payload — include the original query for debugging.
            storable = {
                "final_answer": answer_payload.get("final_answer", ""),
                "source_citations": answer_payload.get("source_citations", []),
                "faithfulness_score": answer_payload.get("faithfulness_score"),
                "eval_passed": answer_payload.get("eval_passed"),
                "cached_query": query,
            }
            self._redis.setex(
                name=f"{_ANS_PREFIX}{cache_id}",
                time=self._ttl,
                value=json.dumps(storable),
            )

            logger.info(
                "semantic_cache_set",
                extra={
                    "cache_id": cache_id,
                    "ttl_seconds": self._ttl,
                    "query_preview": query[:80],
                }
            )
            return True

        except Exception as e:
            logger.warning(
                "semantic_cache_set_error",
                extra={"error": str(e)}
            )
            return False

    def flush(self) -> int:
        """
        Delete all semantic cache entries from Redis.

        Called by the indexer after a document corpus update to prevent
        stale answers from being served.

        Returns:
            Number of keys deleted. 0 if Redis unavailable or cache empty.
        """
        if not self._is_redis_available():
            return 0

        try:
            keys = list(self._redis.scan_iter(match=f"{_EMB_PREFIX}*"))
            keys += list(self._redis.scan_iter(match=f"{_ANS_PREFIX}*"))

            if keys:
                deleted = self._redis.delete(*keys)
                logger.info(
                    "semantic_cache_flushed",
                    extra={"keys_deleted": deleted}
                )
                return deleted

            return 0

        except Exception as e:
            logger.warning("semantic_cache_flush_error", extra={"error": str(e)})
            return 0

    def stats(self) -> dict:
        """
        Return cache statistics for the /admin/cache/stats API endpoint (Phase 5).
        """
        if not self._is_redis_available():
            return {"status": "unavailable"}

        try:
            emb_keys = list(self._redis.scan_iter(match=f"{_EMB_PREFIX}*"))
            return {
                "status": "available",
                "cached_entries": len(emb_keys),
                "similarity_threshold": self._threshold,
                "ttl_seconds": self._ttl,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}