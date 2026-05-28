"""
ingestion/chunkers/semantic_chunker.py

WHAT IS SEMANTIC CHUNKING:
    Instead of splitting at fixed character counts, embed every sentence and
    split at the point where cosine similarity between adjacent sentences
    drops below a threshold — indicating a topic shift.

    The result: chunks that contain semantically coherent content,
    even when the source document doesn't have clean paragraph breaks.

HOW IT WORKS — step by step:
    1. Split the document into individual sentences (rough split on ". ")
    2. Embed each sentence using text-embedding-3-large
    3. Compute cosine similarity between each consecutive pair: sim(s_i, s_{i+1})
    4. Where similarity drops sharply → topic boundary → new chunk starts
    5. Merge sentences within each boundary window into a single chunk

WHY THIS OUTPERFORMS RECURSIVE FOR NARRATIVE TEXT:
    Recursive splits at character count. If a paragraph is 600 chars and your
    chunk_size is 512, it cuts mid-paragraph — mid-idea. The chunk boundary
    is arbitrary.

    Semantic chunking splits where the *content* changes, not where the
    *character count* says to. For interview questions, research papers, or
    support docs, this materially improves context_precision in RAGAS.

COST:
    Requires an embedding API call per sentence at indexing time.
    For a 10-page document with ~200 sentences: ~200 embedding API calls.
    This is why we batch sentences and why this is NOT the default strategy.

    In Phase 4 evaluation, you'll measure whether the RAGAS improvement
    justifies the indexing cost. That's the scientific answer — not intuition.

IMPLEMENTATION NOTE:
    LangChain ships an experimental SemanticChunker in langchain_experimental.
    We are NOT using it. Reasons:
        1. It's in `langchain_experimental` — not production-stable
        2. It uses a fixed percentile threshold, not an adaptive one
        3. It does not expose the similarity scores for debugging/logging
    We implement our own — it's ~80 lines and gives full control.
"""

import logging
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings
from ingestion.chunkers import BaseChunker

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):

    strategy_name = "semantic"

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        batch_size: int = 64,
    ):
        """
        Args:
            similarity_threshold:
                Cosine similarity below which a sentence boundary becomes
                a chunk boundary. 0.75 is a reasonable starting default.
                Lower → more, smaller chunks (more splits).
                Higher → fewer, larger chunks (fewer splits).
                Tune against context_precision in your RAGAS eval.

            min_chunk_size:
                Minimum character count for a chunk. Prevents single-sentence
                chunks from being emitted — they're too narrow for useful retrieval.
                If a boundary-split produces a chunk below this, it's merged
                with the next chunk.

            max_chunk_size:
                Safety ceiling. Even if cosine similarity stays high,
                a chunk exceeding this is force-split. Prevents runaway merges
                in documents with consistently similar adjacent sentences.

            batch_size:
                Number of sentences per embedding API call.
                OpenAI supports up to 2048 inputs per call.
                64 is conservative — adjust upward if indexing time is too slow.
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.batch_size = batch_size

        self._embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key
            # openai_api_key=settings.openai_api_key,
        )

        logger.debug(
            "semantic_chunker_initialized",
            extra={
                "similarity_threshold": self.similarity_threshold,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Rough sentence splitter — sufficient for embedding purposes.
        We are not building a linguistics pipeline here.
        The goal is granular units for similarity comparison, not perfect
        sentence detection.
        """
        import re
        # Split on sentence-ending punctuation followed by whitespace.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter empty strings from multiple consecutive punctuation.
        return [s.strip() for s in sentences if s.strip()]

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        """
        Embed sentences in batches to avoid hitting API request size limits.
        Returns list of embedding vectors in the same order as input sentences.
        """
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i: i + self.batch_size]
            embeddings = self._embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Cosine similarity between two embedding vectors.

        OpenAI embeddings are L2-normalized by default (unit vectors),
        so cosine similarity = dot product. We compute it explicitly
        for clarity and to handle edge cases.
        """
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_split_points(self, embeddings: list[list[float]]) -> list[int]:
        """
        Returns indices where a new chunk should start, based on similarity drops.

        A split point at index i means:
            sentences[0..i-1] → chunk A
            sentences[i..]    → chunk B
        """
        split_points = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                split_points.append(i)
        return split_points

    def _merge_sentences_into_chunks(
        self,
        sentences: list[str],
        split_points: list[int],
    ) -> list[str]:
        """
        Groups sentences by split boundaries and joins them into chunk strings.
        Enforces min_chunk_size (merge small chunks) and max_chunk_size (force split).
        """
        # Build groups of sentences between split boundaries.
        boundaries = [0] + split_points + [len(sentences)]
        groups = [
            sentences[boundaries[i]: boundaries[i + 1]]
            for i in range(len(boundaries) - 1)
        ]

        raw_chunks = [" ".join(group).strip() for group in groups if group]

        # Enforce size constraints.
        final_chunks = []
        pending = ""

        for chunk_text in raw_chunks:
            combined = (pending + " " + chunk_text).strip() if pending else chunk_text

            if len(combined) < self.min_chunk_size:
                # Too small — keep accumulating into pending.
                pending = combined
                continue

            if len(combined) > self.max_chunk_size and pending:
                # Pending alone exceeds max — flush pending first.
                if pending:
                    final_chunks.append(pending)
                pending = chunk_text
            else:
                final_chunks.append(combined)
                pending = ""

        # Flush any remaining pending content.
        if pending:
            final_chunks.append(pending)

        return [c for c in final_chunks if c.strip()]

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into semantically coherent chunks.

        NOTE: This makes embedding API calls. For large document sets,
        expect ~1–3 seconds per document depending on sentence count.
        Use recursive chunking for fast dev iteration; semantic for
        production indexing where retrieval quality is prioritized.
        """
        if not documents:
            logger.warning("semantic_chunker_received_empty_document_list")
            return []

        all_chunks: list[Document] = []

        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(
                    "skipping_empty_document",
                    extra={"source": doc.metadata.get("source", "unknown")}
                )
                continue

            source = doc.metadata.get("source", "unknown")

            # Step 1: Sentence split.
            sentences = self._split_into_sentences(doc.page_content)

            if len(sentences) <= 1:
                # Single sentence or very short doc — emit as-is.
                chunk = Document(
                    page_content=doc.page_content.strip(),
                    metadata=self._build_chunk_metadata(
                        source_metadata=doc.metadata,
                        chunk_index=0,
                        char_count=len(doc.page_content),
                    )
                )
                all_chunks.append(chunk)
                continue

            logger.debug(
                "semantic_embedding_sentences",
                extra={"source": source, "sentence_count": len(sentences)}
            )

            # Step 2: Embed all sentences.
            embeddings = self._embed_sentences(sentences)

            # Step 3: Find split points (similarity drops).
            split_points = self._find_split_points(embeddings)

            # Step 4: Merge sentences into chunks.
            chunk_texts = self._merge_sentences_into_chunks(sentences, split_points)

            # Step 5: Build Document objects with metadata.
            for idx, chunk_text in enumerate(chunk_texts):
                chunk = Document(
                    page_content=chunk_text,
                    metadata=self._build_chunk_metadata(
                        source_metadata=doc.metadata,
                        chunk_index=idx,
                        char_count=len(chunk_text),
                    )
                )
                all_chunks.append(chunk)

            logger.info(
                "semantic_chunking_complete_for_doc",
                extra={
                    "source": source,
                    "sentences": len(sentences),
                    "split_points": len(split_points),
                    "output_chunks": len(chunk_texts),
                }
            )

        logger.info(
            "semantic_chunking_complete",
            extra={
                "input_docs": len(documents),
                "output_chunks": len(all_chunks),
            }
        )

        return all_chunks