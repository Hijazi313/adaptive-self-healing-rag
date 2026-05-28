"""
ingestion/indexer.py — Qdrant Collection Setup for Hybrid Search

THIS FILE DOES TWO THINGS:
    1. Creates the Qdrant collection with the correct vector configuration
       (dense for semantic search, sparse for BM25/keyword search)
    2. Provides a verified, reusable QdrantClient factory for all other modules

WHY HYBRID VECTOR CONFIGURATION MATTERS:
    A Qdrant collection is schema-defined at creation time.
    You cannot add a sparse vector config to an existing dense-only collection.
    Getting this right on Day 1 prevents a full re-index later.

    Dense vectors  → float arrays, e.g. [0.021, -0.134, ...] × 3072 dims
    Sparse vectors → {index: value} dict, e.g. {5821: 0.73, 10042: 1.2}
                     Only non-zero values stored — efficient for BM25

ARCHITECTURE NOTE — Why FastEmbed for sparse vectors?
    Qdrant's own FastEmbed library provides the sparse encoder that integrates
    natively with Qdrant's sparse vector format. It runs locally (no API call),
    is deterministic, and produces SPLADE-style sparse vectors that outperform
    traditional BM25 on short text segments (the RAG use case).
"""

import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# The named configs for dense and sparse vectors within the collection.
# These string names are referenced in every search call — treat as constants.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Qdrant's distance metric for dense vectors.
# COSINE is correct for OpenAI embeddings — they are normalized unit vectors,
# so cosine similarity = dot product. DOT is slightly faster but requires
# normalized vectors. COSINE is safer and semantically equivalent here.
DENSE_DISTANCE = models.Distance.COSINE


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Returns a configured QdrantClient.

    This is a factory function, not a singleton — QdrantClient manages its own
    connection pool internally. Call this once at application startup and pass
    the client instance around (dependency injection pattern).

    For Qdrant Cloud, swap the constructor:
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    """
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=30,
        # 30s timeout: generous for local dev, appropriate for cloud.
        # Default is 5s which causes false failures on slow hardware.
    )


# ─────────────────────────────────────────────────────────────────────────────
# Collection Setup
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def create_collection_if_not_exists(client: QdrantClient) -> bool:
    """
    Idempotent collection creation.

    Returns:
        True  — collection was created now
        False — collection already existed, no action taken

    WHY IDEMPOTENT:
        This function is called at every application startup.
        If the collection exists, we skip creation — not an error.
        This is the safe pattern for infrastructure setup in services
        that restart frequently (containers, serverless, etc).

    WHY @retry:
        Qdrant may not be ready immediately after Docker starts.
        Exponential backoff: 2s → 4s → 10s. 3 attempts total.
        This replaces fragile time.sleep() startup waits.
    """
    collection_name = settings.qdrant_collection_name

    # Check existence first — cheaper than catching create errors.
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info(
            "collection_exists",
            extra={"collection": collection_name}
        )
        return False

    logger.info(
        "creating_collection",
        extra={
            "collection": collection_name,
            "dense_dims": settings.embedding_dimensions,
            "distance": "COSINE",
        }
    )

    client.create_collection(
        collection_name=collection_name,

        # ── Dense Vector Config ────────────────────────────────────────────
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=settings.embedding_dimensions,
                # 3072 for text-embedding-3-large.
                # CRITICAL: if you switch embedding models, this value
                # must change AND you must recreate the collection.
                # Mismatched dimensions = Qdrant rejects all upserts.

                distance=DENSE_DISTANCE,

                on_disk=False,
                # on_disk=True = memory-mapped storage (lower RAM, slower).
                # False = keep in RAM for fast local dev.
                # In production with millions of vectors, set to True.
            )
        },

        # ── Sparse Vector Config (BM25/SPLADE) ────────────────────────────
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                    # Sparse index in RAM for fast development.
                    # Production: set to True and use Qdrant Cloud's NVMe.
                )
            )
        },

        # ── HNSW Index Config for Dense Vectors ───────────────────────────
        hnsw_config=models.HnswConfigDiff(
            m=16,
            # Number of edges per node in the HNSW graph.
            # Higher m = better recall, more memory, slower indexing.
            # 16 is the production default for most embedding dims.

            ef_construct=100,
            # Nodes examined during index construction.
            # Higher = better recall, slower indexing.
            # 100 is the standard default.
        ),

        # ── Optimizers ─────────────────────────────────────────────────────
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2,
            # Number of segments for parallel indexing.
            # 2 is fine for local dev. Production: set to CPU count.
        ),
    )

    logger.info(
        "collection_created",
        extra={"collection": collection_name}
    )
    return True


def verify_collection(client: QdrantClient) -> dict:
    """
    Returns collection info for verification and logging.

    Call this after create_collection_if_not_exists() to confirm
    the configuration is exactly what you expect.

    In Phase 4 (evaluation harness), this is also used to log
    the collection state at the start of each eval run.
    """
    info = client.get_collection(settings.qdrant_collection_name)

    return {
        "name": settings.qdrant_collection_name,
        "status": info.status,
        "indexed_vectors_count": info.indexed_vectors_count,
        "points_count": info.points_count,
        "dense_vector_size": (
            info.config.params.vectors.get(DENSE_VECTOR_NAME).size
            if isinstance(info.config.params.vectors, dict)
            else None
        ),
        "sparse_vectors": list(
            info.config.params.sparse_vectors.keys()
        ) if info.config.params.sparse_vectors else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint — Run directly to initialize infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Entry point for infrastructure setup.
     run this using 
     uv run setup-qdrant
    """
    import json
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    log = structlog.get_logger()

    log.info("initializing_qdrant_infrastructure")

    client = get_qdrant_client()

    # Verify Qdrant is reachable before attempting collection creation.
    try:
        client.get_collections()
        log.info("qdrant_reachable", host=settings.qdrant_host, port=settings.qdrant_port)
    except Exception as e:
        log.error(
            "qdrant_unreachable",
            error=str(e),
            hint="Is `docker compose up -d` running? Check: http://localhost:6333/healthz"
        )
        raise SystemExit(1)

    created = create_collection_if_not_exists(client)

    if created:
        log.info("collection_setup_complete", action="created")
    else:
        log.info("collection_setup_complete", action="already_exists")

    info = verify_collection(client)
    log.info("collection_verified", **info)

    print("\n✓ Qdrant infrastructure ready.")
    print(f"  Collection : {info['name']}")
    print(f"  Status     : {info['status']}")
    print(f"  Dense dims : {info['dense_vector_size']}")
    print(f"  Sparse keys: {info['sparse_vectors']}")
    print(f"\n  Dashboard  : http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()