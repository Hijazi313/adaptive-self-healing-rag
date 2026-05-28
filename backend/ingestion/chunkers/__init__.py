"""
ingestion/chunkers/__init__.py

WHY A UNIFIED INTERFACE EXISTS:
    The RetrieverAgent, the indexer, and the evaluation harness all need to
    call chunkers. None of them should contain an if/elif block deciding which
    chunker class to instantiate — that's the factory's job.

    This module provides:
    1. BaseChunker — the abstract contract every chunker must satisfy
    2. get_chunker() — factory function: strategy name → chunker instance

USAGE:
    from ingestion.chunkers import get_chunker

    chunker = get_chunker("recursive")
    chunks = chunker.chunk(documents)

    # Swap strategy without touching caller code:
    chunker = get_chunker("proposition")
    chunks = chunker.chunk(documents)
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    Every chunker must implement chunk() with this exact signature.
    The evaluation harness depends on this contract — do not break it.
    """

    strategy_name: str  # Set as a class variable in each subclass.

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of source documents into chunks.

        Args:
            documents: Source LangChain Document objects. Each has
                       .page_content (str) and .metadata (dict).

        Returns:
            List of chunk Documents. Each chunk's metadata MUST contain:
                - source         : str  — origin doc identifier
                - chunk_index    : int  — position within source document
                - chunk_strategy : str  — which strategy produced this chunk
                - char_count     : int  — character length of chunk content

            Plus any metadata inherited from the source document.
        """
        ...

    def _build_chunk_metadata(
        self,
        source_metadata: dict,
        chunk_index: int,
        char_count: int,
    ) -> dict:
        """
        Shared metadata builder — guarantees every chunker emits the same
        metadata schema. Override in subclasses only to add strategy-specific
        fields, not to replace the base fields.
        """
        return {
            **source_metadata,          # inherit all source doc metadata
            "chunk_index": chunk_index,
            "chunk_strategy": self.strategy_name,
            "char_count": char_count,
        }


def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    """
    Factory function — maps strategy name to chunker instance.

    Args:
        strategy: One of "recursive", "semantic", "proposition"
        **kwargs: Passed through to the chunker constructor.
                  Allows caller to override defaults (e.g. chunk_size).

    Raises:
        ValueError: For unknown strategy names — fail loudly, not silently.
    """
    # Import inside function to avoid circular imports and keep startup fast.
    # Only the requested chunker is imported and instantiated.
    if strategy == "recursive":
        from ingestion.chunkers.recursive_chunker import RecursiveChunker
        return RecursiveChunker(**kwargs)

    elif strategy == "semantic":
        from ingestion.chunkers.semantic_chunker import SemanticChunker
        return SemanticChunker(**kwargs)

    elif strategy == "proposition":
        from ingestion.chunkers.proposition_chunker import PropositionChunker
        return PropositionChunker(**kwargs)

    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Valid options: 'recursive', 'semantic', 'proposition'."
        )