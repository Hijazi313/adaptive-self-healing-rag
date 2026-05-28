"""
config.py — Central configuration for the Adaptive RAG System.

WHY THIS EXISTS:
    Every module that needs a setting imports from here.
    Nobody calls os.getenv() directly. Nobody hardcodes values.

    This gives you:
    - Type safety (Pydantic validates types at startup)
    - One place to audit all configuration
    - Easy environment switching (local → staging → prod = just swap .env)
    - Fail-fast: if OPENAI_API_KEY is missing, the app crashes at startup
      with a clear error, not at 2am when a request hits an unconfigured path.

USAGE:
    from config import settings
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Pydantic BaseSettings reads from environment variables (case-insensitive).
    With python-dotenv installed, it also reads from .env automatically.

    Field(...) = required (no default)
    Field("default") = optional with fallback
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # extra="ignore" means unknown env vars don't cause validation errors.
        # Useful when your shell has many unrelated env vars set.
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="adaptive_rag_docs")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    semantic_cache_ttl_seconds: int = Field(default=3600)
    semantic_cache_similarity_threshold: float = Field(default=0.92)

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dimensions: int = Field(default=3072)

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: str = Field(default="true")
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="adaptive-rag-system")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ── mem0 ──────────────────────────────────────────────────────────────────
    mem0_api_key: str = Field(default="")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=10)
    max_retrieval_retries: int = Field(default=3)
    critic_relevance_threshold: float = Field(default=0.7)

    # ── Chunking ──────────────────────────────────────────────────────────────
    default_chunk_strategy: str = Field(default="recursive")
    recursive_chunk_size: int = Field(default=512)
    recursive_chunk_overlap: int = Field(default=64)


# Module-level singleton.
# Import `settings` everywhere — don't instantiate Settings() in each module.
# Pydantic reads and validates once at import time.
settings = Settings()