"""
ingestion/chunkers/proposition_chunker.py

WHAT IS PROPOSITION-LEVEL CHUNKING:
    Convert paragraphs into atomic, self-contained factual statements
    before embedding them. Each proposition can be retrieved independently
    without requiring its surrounding context to be meaningful.

    Source paragraph:
        "The rate of churn increased by 12% in Q3 due to pricing changes
         and onboarding friction, which affected primarily new users."

    Propositions produced:
        1. "Customer churn rate increased by 12% in Q3."
        2. "Pricing changes contributed to the Q3 churn increase."
        3. "Onboarding friction contributed to the Q3 churn increase."
        4. "The Q3 churn increase primarily affected new users."

    Each proposition is:
        - Standalone (makes sense without context)
        - Atomic (one fact per statement)
        - Embeddable (can be retrieved for a specific sub-question)

WHY THIS IS STATE-OF-THE-ART:
    Dense RAG retrieval fetches chunks. If a chunk contains 5 facts but the
    query only needs 1 of them, the other 4 are noise — they hurt context
    precision. Proposition chunking maximizes the signal-to-noise ratio
    in retrieved context.

    Context_precision in RAGAS measures exactly this. You will see the
    difference in Phase 4 evaluation.

ACADEMIC REFERENCE:
    "Dense X Retrieval: What Retrieval Granularity Should We Use?"
    Chen et al., 2023 — the paper that introduced proposition-level retrieval.
    https://arxiv.org/abs/2312.06648

COST PROFILE:
    - Requires 1 LLM call per paragraph (or per N characters)
    - For a 10-page document with ~40 paragraphs: ~40 GPT-4o calls
    - This is expensive — use batching and cache by document hash
    - Never run proposition chunking on every indexing call without a hash check

PRODUCTION STRATEGY:
    1. Compute SHA256 hash of source document content
    2. Cache propositions keyed by hash in Redis (or a simple JSON file)
    3. On re-index: if hash unchanged → load from cache, skip LLM calls
    4. If hash changed → recompute, update cache

    Phase 1 implements the core logic. Caching is added in Phase 3
    when the Redis layer is live.
"""

import json
import logging
import hashlib
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from config import settings
from ingestion.chunkers import BaseChunker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt — the core of proposition chunking quality
# ─────────────────────────────────────────────────────────────────────────────

PROPOSITION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise information extraction engine.
Your task is to decompose a paragraph into atomic, self-contained propositions.

Rules:
1. Each proposition must be a single, standalone factual statement.
2. Each proposition must be fully interpretable without reading any other proposition.
   - BAD:  "This contributed to the increase."  (what increase? what contributed?)
   - GOOD: "Pricing changes contributed to the Q3 customer churn increase."
3. Preserve all specific values: numbers, percentages, dates, names, locations.
4. Do not add information that is not in the source paragraph.
5. Do not merge multiple facts into one proposition.
6. Return ONLY a valid JSON array of strings. No preamble, no explanation.

Example input:
"The churn rate rose 12% in Q3 due to pricing changes and poor onboarding."

Example output:
["The churn rate rose 12% in Q3.",
 "Pricing changes contributed to the Q3 churn rate increase.",
 "Poor onboarding contributed to the Q3 churn rate increase."]"""
    ),
    (
        "human",
        "Extract atomic propositions from this paragraph:\n\n{paragraph}"
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

class PropositionChunker(BaseChunker):

    strategy_name = "proposition"

    def __init__(
        self,
        model: str = "gpt-4o",
        paragraph_min_chars: int = 80,
        batch_size: int = 5,
    ):
        """
        Args:
            model:
                LLM for proposition extraction. Use gpt-4o — not gpt-3.5-turbo.
                Quality of propositions is the core of this strategy.
                GPT-3.5 frequently merges related facts or loses specifics.
                The higher cost is the cost of state-of-the-art retrieval.

            paragraph_min_chars:
                Paragraphs shorter than this are emitted as-is without LLM
                processing. Prevents API calls for headings, captions, or
                single-sentence paragraphs where decomposition adds no value.

            batch_size:
                Number of paragraphs processed concurrently using asyncio.
                Controls API rate limit exposure. 5 is a conservative default.
                Increase if your OpenAI tier supports higher RPM.
        """
        self.paragraph_min_chars = paragraph_min_chars
        self.batch_size = batch_size

        # Temperature=0 — we need deterministic, factual extraction.
        # This is not a creative task. Any temperature > 0 risks fabrication.
        self._llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )

        self._chain = PROPOSITION_EXTRACTION_PROMPT | self._llm

        logger.debug(
            "proposition_chunker_initialized",
            extra={
                "model": model,
                "paragraph_min_chars": paragraph_min_chars,
                "batch_size": batch_size,
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        """
        Split document text into paragraphs on double newlines.
        Strips whitespace and filters empty strings.
        """
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def _compute_doc_hash(content: str) -> str:
        """
        SHA256 hash of document content.
        Used as the cache key in Phase 3 to skip reprocessing unchanged docs.
        Stored in chunk metadata so the evaluation harness can detect
        when a document was re-indexed vs loaded from cache.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _extract_propositions(self, paragraph: str) -> list[str]:
        """
        Call the LLM to extract atomic propositions from a single paragraph.

        Returns a list of proposition strings.
        Falls back to the original paragraph as a single-element list
        if LLM output cannot be parsed as JSON — silent failure prevention.

        The @retry decorator handles transient OpenAI API errors:
        rate limits, timeouts, 5xx responses.
        """
        response = self._chain.invoke({"paragraph": paragraph})
        raw_text = response.content.strip()

        try:
            propositions = json.loads(raw_text)

            if not isinstance(propositions, list):
                raise ValueError(f"Expected list, got {type(propositions)}")

            # Validate every element is a non-empty string.
            cleaned = [p.strip() for p in propositions if isinstance(p, str) and p.strip()]

            if not cleaned:
                raise ValueError("LLM returned empty proposition list")

            return cleaned

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "proposition_parse_failed_using_fallback",
                extra={
                    "error": str(e),
                    "raw_response": raw_text[:200],
                    "paragraph_preview": paragraph[:100],
                }
            )
            # Fallback: treat the paragraph as a single proposition.
            # This is always safe — we lose atomicity but not content.
            return [paragraph]

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Convert documents into proposition-level chunks via LLM extraction.

        Processing order:
            For each document:
                Split into paragraphs
                For each paragraph above min_chars:
                    Extract propositions via LLM
                For each paragraph below min_chars:
                    Emit as-is (skip LLM call)
                Build chunk Documents with full metadata

        NOTE on async:
            This implementation is synchronous for clarity and debuggability.
            Phase 3 adds async batch processing when Redis caching is in place
            — there's no value in async without caching since the LLM call
            is the bottleneck, not I/O wait.
        """
        if not documents:
            logger.warning("proposition_chunker_received_empty_document_list")
            return []

        all_chunks: list[Document] = []
        chunk_index = 0  # Global chunk index across the document's propositions.

        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(
                    "skipping_empty_document",
                    extra={"source": doc.metadata.get("source", "unknown")}
                )
                continue

            source = doc.metadata.get("source", "unknown")
            doc_hash = self._compute_doc_hash(doc.page_content)
            paragraphs = self._split_into_paragraphs(doc.page_content)

            logger.info(
                "proposition_chunking_doc",
                extra={
                    "source": source,
                    "doc_hash": doc_hash,
                    "paragraph_count": len(paragraphs),
                }
            )

            chunk_index = 0  # Reset per document.

            for para in paragraphs:
                if len(para) < self.paragraph_min_chars:
                    # Short paragraph — emit directly without LLM.
                    chunk = Document(
                        page_content=para,
                        metadata=self._build_chunk_metadata(
                            source_metadata={**doc.metadata, "doc_hash": doc_hash},
                            chunk_index=chunk_index,
                            char_count=len(para),
                        )
                    )
                    # Add strategy-specific metadata.
                    chunk.metadata["is_proposition"] = False
                    chunk.metadata["source_paragraph"] = para[:100]
                    all_chunks.append(chunk)
                    chunk_index += 1
                    continue

                # Extract propositions via LLM.
                propositions = self._extract_propositions(para)

                for prop in propositions:
                    chunk = Document(
                        page_content=prop,
                        metadata=self._build_chunk_metadata(
                            source_metadata={**doc.metadata, "doc_hash": doc_hash},
                            chunk_index=chunk_index,
                            char_count=len(prop),
                        )
                    )
                    # Tag with proposition-specific metadata for eval analysis.
                    chunk.metadata["is_proposition"] = True
                    chunk.metadata["source_paragraph"] = para[:100]
                    all_chunks.append(chunk)
                    chunk_index += 1

            logger.info(
                "proposition_chunking_complete_for_doc",
                extra={
                    "source": source,
                    "doc_hash": doc_hash,
                    "paragraphs": len(paragraphs),
                    "propositions": chunk_index,
                }
            )

        logger.info(
            "proposition_chunking_complete",
            extra={
                "input_docs": len(documents),
                "output_chunks": len(all_chunks),
            }
        )

        return all_chunks