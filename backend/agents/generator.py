"""
agents/generator.py — GeneratorAgent

RESPONSIBILITY:
    Produces the final answer, grounded strictly in retrieved_chunks.
    Also extracts structured source citations for the API response.

GROUNDING PRINCIPLE:
    The generator must not answer from its parametric memory (training data).
    It must answer ONLY from the provided context chunks.
    This is enforced via the system prompt and is what "grounded generation" means.

    If the context doesn't contain the answer, the correct response is:
    "I cannot answer this question based on the available information."
    This is preferable to a confident hallucination.

WHY CITATIONS:
    source_citations gives the API caller (and the evaluator) traceability.
    Every claim in the answer can be traced to a specific chunk and source document.
    This is what makes the system auditable — a non-negotiable property
    in legal, medical, or enterprise knowledge base use cases.

CITATION FORMAT:
    Each citation:
    {
        "source": "path/to/document.pdf",
        "chunk_index": 3,
        "excerpt": "first 150 chars of the chunk used"
    }

MODEL CHOICE:
    gpt-4o — the generator needs to synthesise coherently across multiple chunks.
    Using gpt-3.5-turbo here would degrade answer quality and RAGAS faithfulness scores.
    The evaluator uses gpt-3.5-turbo (cheaper for batch scoring) — generator does not.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import settings
from graph.state import RAGState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided context chunks.
Do not use your training knowledge. Do not speculate beyond the context.

Rules:
1. If the context contains the answer, answer clearly and completely.
2. If the context only partially answers the question, answer what you can
   and explicitly state what information is missing.
3. If the context does not contain enough information to answer,
   respond with: "I cannot answer this question based on the available information."
4. Keep your answer factual and concise.
5. Do not fabricate sources, names, numbers, or facts not in the context."""
    ),
    (
        "human",
        "Question: {query}\n\nContext:\n{context}"
    ),
])


def _format_context(chunks: list[Document]) -> str:
    """
    Format chunks into a numbered context block for the generator prompt.
    Each chunk is numbered so the model can reference them in its answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{chunk.page_content}")
    return "\n\n".join(parts)


def _extract_citations(chunks: list[Document]) -> list[dict]:
    """
    Build structured citations from the chunks provided to the generator.

    We cite ALL chunks that were provided — not just the ones the model
    "used". This is conservative but safe: it's the caller's job to
    display citations; it's our job to provide traceability.

    In a future iteration, the generator could return chunk reference numbers
    in its answer and we could filter to only cited chunks.
    """
    citations = []
    for chunk in chunks:
        citations.append({
            "source": chunk.metadata.get("source", "unknown"),
            "chunk_index": chunk.metadata.get("chunk_index", -1),
            "chunk_strategy": chunk.metadata.get("chunk_strategy", "unknown"),
            "excerpt": chunk.page_content[:150],
        })
    return citations


# ─────────────────────────────────────────────────────────────────────────────
# Agent node function
# ─────────────────────────────────────────────────────────────────────────────

def generator_node(state: RAGState) -> dict:
    """
    LangGraph node — GeneratorAgent.

    Reads:  state["query"], state["retrieved_chunks"]
    Writes: final_answer, source_citations

    Only called after CriticAgent has approved the retrieved chunks.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        # Should not reach here — supervisor guards on critic_score.
        # Defensive fallback.
        logger.error("generator_called_with_no_chunks")
        return {
            "final_answer": "I cannot answer this question — no relevant context was retrieved.",
            "source_citations": [],
        }

    logger.info(
        "generator_start",
        extra={"chunk_count": len(chunks), "query_preview": query[:100]}
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = GENERATOR_PROMPT | llm

    try:
        context = _format_context(chunks)
        response = chain.invoke({"query": query, "context": context})
        answer = response.content.strip()
        citations = _extract_citations(chunks)

        logger.info(
            "generator_complete",
            extra={
                "answer_length": len(answer),
                "citations": len(citations),
            }
        )

        return {
            "final_answer": answer,
            "source_citations": citations,
        }

    except Exception as e:
        logger.error("generator_failed", extra={"error": str(e)})
        return {
            "final_answer": "Answer generation failed due to an internal error.",
            "source_citations": [],
            "error": f"GeneratorAgent failed: {str(e)}",
        }