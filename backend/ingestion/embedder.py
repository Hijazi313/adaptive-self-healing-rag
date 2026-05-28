"""
ingestion/embedder.py — Embedding and Qdrant Upsert Pipeline

RESPONSIBILITY:
    Accepts chunked Documents (output of any BaseChunker),
    generates dense (OpenAI) + sparse (BM25/FastEmbed) vectors,
    and upserts them as points into the Qdrant collection.

THIS FILE IS THE BRIDGE:
    Chunkers  →  [Embedder]  →  Qdrant

DESIGN DECISIONS:

1. Deterministic point IDs via content hash
   Why: If you re-index the same document, the point ID must be the same.
   An upsert with the same ID overwrites the existing point — no duplicates.
   A random UUID on every run doubles your collection on every re-index.
   ID = SHA256(chunk_content + source + chunk_index)[:16] → deterministic UUID5.

2. Explicit batch control
   Why: Qdrant has a default payload size limit (~32MB per request).
   Proposition chunking can produce thousands of small chunks per document.
   We batch at UPSERT_BATCH_SIZE (default 100) to stay well under the limit
   and give progress visibility via structured logging.

3. QdrantVectorStore wraps the client for the retriever interface
   Why: The RetrieverAgent (Phase 2) uses LangChain's retriever abstraction.
   By initialising QdrantVectorStore here (not in the RetrieverAgent),
   we have one place that knows about collection config, vector names,
   and embedding models. The agent just calls .as_retriever().

4. FastEmbed runs locally — no API call for sparse vectors
   Why: BM25/SPLADE encoding is CPU-bound inference, not a network call.
   Sparse vectors are cheap to produce. Do not batch-limit based on API rate.

IDF MODIFIER NOTE:
   FastEmbed's BM25 encoder intentionally omits IDF (Inverse Document Frequency).
   Qdrant's collection must have the IDF modifier enabled on the sparse index
   for correct BM25 ranking. This was configured in indexer.py Day 1.
   Do not change the sparse index config without also updating the encoder here.
"""

import hashlib
import logging
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from config import settings
from ingestion.indexer import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
    create_collection_if_not_exists,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

UPSERT_BATCH_SIZE = 100
# 100 chunks per Qdrant upsert request.
# Conservative default — each chunk payload + two vectors (dense 3072d, sparse)
# is ~50-80KB. 100 chunks ≈ 5-8MB per request, comfortably under Qdrant's limit.
# Increase to 200-500 if indexing speed is a bottleneck and chunks are small.

SPARSE_MODEL_NAME = "Qdrant/bm25"
# Qdrant's official BM25 sparse encoder via FastEmbed.
# Runs locally via ONNX — no API call.
# Alternative: "prithivida/Splade_PP_en_v1" for neural sparse (better quality,
# slower inference). BM25 is the correct default for a portfolio-scale corpus.


# ─────────────────────────────────────────────────────────────────────────────
# Point ID generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_point_id(chunk: Document) -> str:
    """
    Generate a deterministic UUID for a chunk based on its content and position.

    Inputs to the hash:
        - page_content : the actual text (primary signal)
        - source       : origin document (prevents cross-doc collisions)
        - chunk_index  : position within source (prevents within-doc collisions)

    Returns a UUID5 string — Qdrant accepts both integer and UUID string IDs.
    UUID5 is deterministic (same inputs → same UUID), unlike UUID4.

    Why UUID5 over truncated SHA256?
        Qdrant validates point IDs as valid UUIDs when using string format.
        UUID5 guarantees valid UUID structure while remaining deterministic.
    """
    content = chunk.page_content
    source = chunk.metadata.get("source", "")
    chunk_index = str(chunk.metadata.get("chunk_index", 0))

    fingerprint = f"{source}::{chunk_index}::{content}"
    # UUID5 with DNS namespace — standard practice for content-derived UUIDs.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, fingerprint))


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class Embedder:
    """
    Embeds chunked Documents and upserts them into Qdrant.

    Lifecycle:
        embedder = Embedder()
        embedder.index(chunks)   # call once per chunking run

    The Embedder also exposes get_vector_store() so the RetrieverAgent
    can get a ready-to-query QdrantVectorStore without re-initialising
    embeddings or client connections.
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        """
        Args:
            qdrant_client: Injected client for testing. If None, creates one
                           from settings. Dependency injection keeps this testable.
        """
        self._client = qdrant_client or get_qdrant_client()

        # Dense embedding model — text-embedding-3-large, 3072 dims.
        # Used for: semantic similarity, conceptual queries.
        self._dense_embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key,
            # openai_api_key=settings.openai_api_key,
        )

        # Sparse embedding model — BM25 via FastEmbed, runs locally.
        # Used for: keyword matching, exact term queries.
        self._sparse_embeddings = FastEmbedSparse(
            model_name=SPARSE_MODEL_NAME,
        )

        # Ensure collection exists before any upsert attempt.
        # Idempotent — safe to call every time.
        create_collection_if_not_exists(self._client)

        logger.info(
            "embedder_initialized",
            extra={
                "dense_model": settings.embedding_model,
                "sparse_model": SPARSE_MODEL_NAME,
                "collection": settings.qdrant_collection_name,
                "batch_size": UPSERT_BATCH_SIZE,
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_vector_store(self) -> QdrantVectorStore:
        """
        Instantiate QdrantVectorStore configured for HYBRID retrieval.

        This is the LangChain abstraction the RetrieverAgent will use.
        Centralised here so config never drifts between indexing and retrieval.
        """
        return QdrantVectorStore(
            client=self._client,
            collection_name=settings.qdrant_collection_name,
            embedding=self._dense_embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _upsert_batch(
        self,
        vector_store: QdrantVectorStore,
        batch: list[Document],
        ids: list[str],
    ) -> None:
        """
        Upsert a single batch of Documents into Qdrant.

        QdrantVectorStore.add_documents() handles:
            - Calling OpenAI for dense embeddings (batched internally)
            - Running FastEmbed locally for sparse vectors
            - Formatting PointStructs with named vectors
            - Executing the Qdrant upsert

        The @retry decorator handles transient failures:
            - OpenAI rate limits (429)
            - Qdrant write timeouts
            - Network blips

        We pass explicit IDs so Qdrant uses our deterministic UUIDs,
        not random ones. This is the key to idempotent re-indexing.
        """
        vector_store.add_documents(documents=batch, ids=ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def index(self, chunks: list[Document]) -> dict:
        """
        Embed and index a list of chunks into Qdrant.

        Args:
            chunks: Output from any BaseChunker. Expected metadata keys:
                    source, chunk_index, chunk_strategy, char_count.

        Returns:
            Summary dict for logging and eval harness consumption:
            {
                "total_chunks": int,
                "batches": int,
                "indexed_ids": list[str],
                "chunk_strategy": str,   # from first chunk's metadata
                "collection": str,
            }

        Process:
            1. Generate deterministic point ID per chunk
            2. Split into batches of UPSERT_BATCH_SIZE
            3. For each batch: embed (dense + sparse) and upsert
            4. Return summary
        """
        if not chunks:
            logger.warning("embedder_index_called_with_empty_chunk_list")
            return {"total_chunks": 0, "batches": 0, "indexed_ids": []}

        # Validate chunks have the expected metadata contract.
        _validate_chunks(chunks)

        vector_store = self._build_vector_store()

        # Generate all point IDs upfront — deterministic, no side effects.
        point_ids = [_make_point_id(chunk) for chunk in chunks]

        # Split into batches.
        batches = [
            (chunks[i: i + UPSERT_BATCH_SIZE], point_ids[i: i + UPSERT_BATCH_SIZE])
            for i in range(0, len(chunks), UPSERT_BATCH_SIZE)
        ]

        chunk_strategy = chunks[0].metadata.get("chunk_strategy", "unknown")

        logger.info(
            "embedder_indexing_start",
            extra={
                "total_chunks": len(chunks),
                "total_batches": len(batches),
                "chunk_strategy": chunk_strategy,
                "collection": settings.qdrant_collection_name,
            }
        )

        for batch_num, (batch_docs, batch_ids) in enumerate(batches, start=1):
            logger.info(
                "embedder_batch_start",
                extra={
                    "batch": batch_num,
                    "of": len(batches),
                    "size": len(batch_docs),
                }
            )

            self._upsert_batch(vector_store, batch_docs, batch_ids)

            logger.info(
                "embedder_batch_complete",
                extra={"batch": batch_num, "of": len(batches)}
            )

        logger.info(
            "embedder_indexing_complete",
            extra={
                "total_chunks": len(chunks),
                "chunk_strategy": chunk_strategy,
                "collection": settings.qdrant_collection_name,
            }
        )

        return {
            "total_chunks": len(chunks),
            "batches": len(batches),
            "indexed_ids": point_ids,
            "chunk_strategy": chunk_strategy,
            "collection": settings.qdrant_collection_name,
        }

    def get_vector_store(self) -> QdrantVectorStore:
        """
        Return a configured QdrantVectorStore for use by the RetrieverAgent.

        The RetrieverAgent calls this once at init time and calls
        .as_retriever() on the result. No embedding config lives in the agent.

        This is the single source of truth for retrieval configuration.
        """
        return self._build_vector_store()


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_chunks(chunks: list[Document]) -> None:
    """
    Verify chunks conform to the BaseChunker metadata contract.
    Fails loudly at indexing time rather than silently at retrieval time.

    Required metadata fields (set by BaseChunker._build_chunk_metadata):
        - chunk_strategy : str
        - chunk_index    : int
        - char_count     : int

    'source' is checked but allowed to be missing with a warning —
    some document loaders don't set it, and it's not retrieval-critical.
    """
    required_fields = {"chunk_strategy", "chunk_index", "char_count"}

    for i, chunk in enumerate(chunks):
        missing = required_fields - set(chunk.metadata.keys())
        if missing:
            raise ValueError(
                f"Chunk at index {i} is missing required metadata fields: {missing}. "
                f"Ensure chunks are produced by a BaseChunker subclass."
            )
        if not chunk.page_content or not chunk.page_content.strip():
            raise ValueError(
                f"Chunk at index {i} has empty page_content. "
                f"Empty chunks produce zero-signal embeddings — filter before indexing."
            )

    # Non-fatal warning for missing source.
    if any("source" not in c.metadata for c in chunks):
        logger.warning(
            "some_chunks_missing_source_metadata",
            extra={"hint": "Set 'source' in document metadata before chunking for full traceability."}
        )