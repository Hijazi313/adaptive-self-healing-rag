"""
ingestion/chunkers/recursive_chunker.py

WHAT IS RECURSIVE CHUNKING:
    Split text by a hierarchy of separators: paragraphs → sentences → words → chars.
    At each level, if a chunk is still too large, split at the next separator.

    Example hierarchy (English text):
        ["\n\n", "\n", ". ", " ", ""]

    This preserves semantic boundaries as long as possible.
    A paragraph break is always preferred over a mid-sentence split.

WHY THIS IS THE DEFAULT STRATEGY:
    - Fast: no LLM calls, no embedding at chunk time
    - Deterministic: same input always produces same output
    - Works well for most document types (PDFs, web pages, markdown)
    - The chunk_size / chunk_overlap parameters are tunable for your corpus

WHEN IT UNDERPERFORMS:
    - Long, unbroken paragraphs (cuts mid-idea)
    - Documents with domain-specific structure (code, tables, forms)
    - When semantic coherence within a chunk is critical

PRODUCTION NOTE — chunk_size is a hyperparameter:
    512 tokens ≈ 380–420 words ≈ 2,000–2,500 characters (rough rule of thumb).
    Your RAGAS context_precision score is directly downstream of this.
    The evaluation harness in Phase 4 will tell you if this needs tuning.
    Never hardcode it — it lives in .env / settings.
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from ingestion.chunkers import BaseChunker

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):

    strategy_name = "recursive"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Args:
            chunk_size:    Max characters per chunk. Defaults to settings value.
            chunk_overlap: Overlap between consecutive chunks. Defaults to settings value.

        Why character-based, not token-based?
            Token counting requires a tokenizer call (tiktoken) per chunk.
            For the recursive splitter, character-based approximation is
            acceptable and significantly faster at indexing time.
            Token-accurate splitting is reserved for the proposition chunker
            where LLM context limits are the actual constraint.
        """
        self.chunk_size = chunk_size or settings.recursive_chunk_size
        self.chunk_overlap = chunk_overlap or settings.recursive_chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",   # paragraph boundary — highest priority
                "\n",     # line break
                ". ",     # sentence boundary
                "! ",
                "? ",
                "; ",
                ", ",
                " ",      # word boundary
                "",       # character — last resort, never preferred
            ],
            length_function=len,
            is_separator_regex=False,
            # is_separator_regex=False: separators are treated as literal strings.
            # Set to True only if you need regex patterns as separators (advanced).
            keep_separator=False,
            # keep_separator=False: strip the separator from chunk boundaries.
            # True would include the "\n\n" at the start of each chunk — noise.
        )

        logger.debug(
            "recursive_chunker_initialized",
            extra={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split source documents into recursive character-based chunks.

        Each output chunk includes full metadata lineage so the evaluation
        harness can trace every chunk back to its source document and strategy.
        """
        if not documents:
            logger.warning("recursive_chunker_received_empty_document_list")
            return []

        all_chunks: list[Document] = []

        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(
                    "skipping_empty_document",
                    extra={"source": doc.metadata.get("source", "unknown")}
                )
                continue

            raw_chunks: list[Document] = self._splitter.split_documents([doc])

            for idx, chunk in enumerate(raw_chunks):
                chunk.metadata = self._build_chunk_metadata(
                    source_metadata=doc.metadata,
                    chunk_index=idx,
                    char_count=len(chunk.page_content),
                )
                all_chunks.append(chunk)

        logger.info(
            "recursive_chunking_complete",
            extra={
                "input_docs": len(documents),
                "output_chunks": len(all_chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        )

        return all_chunks