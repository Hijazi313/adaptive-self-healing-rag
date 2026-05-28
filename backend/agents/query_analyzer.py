"""
agents/query_analyzer.py — QueryAnalyzerAgent

RESPONSIBILITY:
    First node in the graph. Receives the raw user query and:
    1. Classifies it as "keyword", "conceptual", or "hybrid"
    2. Sets the dense/sparse weight (alpha) for hybrid search
    3. Sets an initial rewritten_query (cleaned, ready for retrieval)

WHY CLASSIFICATION MATTERS:
    The same question asked two different ways needs different retrieval:

    "What is database connection pooling?" → conceptual
        Dense retrieval dominates — semantic similarity finds explanations
        even if they use different terminology. Sparse (BM25) is less useful
        here because the user isn't searching for specific tokens.

    "ERROR: ECONNREFUSED 127.0.0.1:5432" → keyword
        Sparse (BM25) dominates — exact token matching finds the specific
        error. Dense retrieval would find vaguely related connection topics,
        not the specific error code.

    "How does OAuth2 handle token refresh?" → hybrid
        Both matter — the conceptual flow (dense) and specific terms like
        "refresh_token", "access_token" (sparse).

IMPLEMENTATION:
    Uses GPT-4o with structured output (JSON) for classification.
    Temperature=0 — classification is deterministic, not creative.
    Single LLM call. Output feeds all downstream agents via state.

DENSE_WEIGHT VALUES:
    keyword    → 0.25  (75% sparse, 25% dense)
    conceptual → 0.85  (85% dense, 15% sparse)
    hybrid     → 0.55  (balanced with slight dense preference)
    These are starting defaults — tune against RAGAS context_precision.
"""

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from graph.state import RAGState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query classification engine for a RAG retrieval system.
Classify the user query and return ONLY valid JSON. No preamble.

Classification rules:
- "keyword": query contains specific error codes, product IDs, names, exact phrases,
  or technical tokens where exact string matching matters most.
- "conceptual": query asks how something works, why something happens, or requests
  an explanation. Semantic similarity matters more than exact terms.
- "hybrid": query combines specific terms with conceptual understanding needs.

Return this exact JSON structure:
{{
  "query_type": "keyword" | "conceptual" | "hybrid",
  "reasoning": "one sentence explanation",
  "rewritten_query": "cleaned version of query, fixing typos, removing filler words"
}}"""
    ),
    ("human", "{query}"),
])

# Stable alpha values per query type.
DENSE_WEIGHTS = {
    "keyword": 0.25,
    "conceptual": 0.85,
    "hybrid": 0.55,
}


# ─────────────────────────────────────────────────────────────────────────────
# Agent node function
# ─────────────────────────────────────────────────────────────────────────────

def query_analyzer_node(state: RAGState) -> dict:
    """
    LangGraph node — QueryAnalyzerAgent.

    Reads:  state["query"]
    Writes: query_type, dense_weight, rewritten_query, retrieval_strategy

    Returns a partial state update dict. LangGraph merges this into
    the full state — we do not return the entire state object.
    """
    query = state["query"]

    logger.info("query_analyzer_start", extra={"query_preview": query[:100]})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = ANALYZER_PROMPT | llm

    try:
        response = chain.invoke({"query": query})
        raw = response.content.strip()

        # Strip markdown fences if the model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        query_type = parsed.get("query_type", "hybrid")
        if query_type not in DENSE_WEIGHTS:
            logger.warning(
                "query_analyzer_unexpected_type",
                extra={"type": query_type, "defaulting_to": "hybrid"}
            )
            query_type = "hybrid"

        rewritten_query = parsed.get("rewritten_query", query).strip() or query
        dense_weight = DENSE_WEIGHTS[query_type]

        logger.info(
            "query_analyzer_complete",
            extra={
                "query_type": query_type,
                "dense_weight": dense_weight,
                "reasoning": parsed.get("reasoning", ""),
            }
        )

        return {
            "query_type": query_type,
            "dense_weight": dense_weight,
            "rewritten_query": rewritten_query,
            "retrieval_strategy": "dense_sparse_hybrid",
        }

    except (json.JSONDecodeError, Exception) as e:
        # Fail gracefully — default to hybrid so the graph continues.
        logger.error(
            "query_analyzer_failed",
            extra={"error": str(e), "defaulting_to": "hybrid"}
        )
        return {
            "query_type": "hybrid",
            "dense_weight": DENSE_WEIGHTS["hybrid"],
            "rewritten_query": query,
            "retrieval_strategy": "dense_sparse_hybrid",
            "error": f"QueryAnalyzer failed: {str(e)}",
        }