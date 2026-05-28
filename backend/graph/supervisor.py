"""
graph/supervisor.py — LangGraph Supervisor: The Executable Graph

RESPONSIBILITY:
    Wires all agents and the reformulation node into a compiled, executable
    LangGraph StateGraph. Defines every edge — fixed and conditional.
    Implements the self-healing retrieval loop and HITL interrupt.

THIS FILE IS THE SYSTEM.
    Everything built across Days 1–4 is assembled here.
    After this file, the system is runnable end-to-end.

GRAPH TOPOLOGY:

    START
      │
      ▼
    query_analyzer
      │
      ▼
    retriever  ◄──────────────────────────────┐
      │                                        │
      ▼                                        │
    critic                                     │
      │                                        │
      ▼                                        │
    [route_after_critic]                       │
      │                                        │
      ├── score >= threshold ──────────────► generator
      │                                        │
      ├── retries < max ──────────────────► reformulate
      │                                   (updates rewritten_query)
      │                                        │
      │                                        └──────────────────────────────┘
      │
      └── retries == max ─────────────────► human_review
                                              │
                                    interrupt() pauses here
                                    waits for Command(resume=...)
                                              │
                                    ┌─── approved ──► retriever (one more try)
                                    └─── rejected ──► END (with error in state)
      ▼
    generator
      │
      ▼
    evaluator
      │
      ▼
    END

CHECKPOINTER CHOICE — SqliteSaver:
    MemorySaver: fast, loses state on process restart. Fine for unit tests.
    SqliteSaver: file-backed, survives restarts, single-process safe.
                 Correct for local dev and HITL workflows where a human
                 may not respond for minutes or hours.

    The graph is compiled with SqliteSaver so interrupt() works correctly.
    If you switch to FastAPI + uvicorn multi-worker: upgrade to PostgresSaver.

THREAD ID:
    Every graph.invoke() call must pass a thread_id in config.
    This is the key that scopes checkpointed state to one conversation.
    The FastAPI layer generates a uuid4 thread_id per user session.
    HITL resume must use the SAME thread_id as the interrupted run.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from config import settings
from graph.state import RAGState, initial_state
from graph.reformulation import reformulation_node
from agents.query_analyzer import query_analyzer_node
from agents.retriever import retriever_node
from agents.critic import critic_node
from agents.generator import generator_node
from agents.evaluator import evaluator_node

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Node names — string constants prevent typo-based silent misroutes
# ─────────────────────────────────────────────────────────────────────────────

NODE_QUERY_ANALYZER = "query_analyzer"
NODE_RETRIEVER      = "retriever"
NODE_CRITIC         = "critic"
NODE_REFORMULATE    = "reformulate"
NODE_HUMAN_REVIEW   = "human_review"
NODE_GENERATOR      = "generator"
NODE_EVALUATOR      = "evaluator"


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions — deterministic Python, no LLM involved
# ─────────────────────────────────────────────────────────────────────────────

def route_after_critic(
    state: RAGState,
) -> Literal["generator", "reformulate", "human_review", "__end__"]:
    """
    The core routing decision of the self-healing loop.

    Decision tree (evaluated in priority order):
        1. Error in state → END immediately (don't attempt generation)
        2. Critic score >= threshold → proceed to generator
        3. Retries exhausted → escalate to human review
        4. Retries remaining → reformulate and retry

    IMPORTANT: returns string literals that match EXACTLY the keys in the
    conditional_edges mapping dict in build_graph(). A typo here causes a
    silent misroute — the node name constants above prevent this.
    """
    # 1. Hard error — abort the run.
    if state.get("error"):
        logger.warning(
            "routing_to_end_due_to_error",
            extra={"error": state.get("error")}
        )
        return "__end__"

    critic_score = state.get("critic_score") or 0.0
    attempts = len(state.get("retrieval_attempts") or [])

    # 2. Context quality is sufficient — proceed to answer generation.
    if critic_score >= settings.critic_relevance_threshold:
        logger.info(
            "routing_to_generator",
            extra={
                "critic_score": critic_score,
                "threshold": settings.critic_relevance_threshold,
                "attempts": attempts,
            }
        )
        return "generator"

    # 3. All reformulation strategies exhausted — escalate.
    if attempts >= settings.max_retrieval_retries:
        logger.warning(
            "routing_to_human_review",
            extra={
                "critic_score": critic_score,
                "attempts": attempts,
                "max": settings.max_retrieval_retries,
            }
        )
        return "human_review"

    # 4. Retry with next reformulation strategy.
    logger.info(
        "routing_to_reformulate",
        extra={
            "critic_score": critic_score,
            "threshold": settings.critic_relevance_threshold,
            "attempts": attempts,
            "max": settings.max_retrieval_retries,
        }
    )
    return "reformulate"


# ─────────────────────────────────────────────────────────────────────────────
# HITL node — interrupt() pauses the graph here
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(
    state: RAGState,
) -> Command[Literal["retriever", "__end__"]]:
    """
    HITL node. Pauses the graph and surfaces review context to the caller.

    HOW interrupt() WORKS:
        1. Graph execution reaches this node.
        2. interrupt() serializes the payload and SAVES graph state via checkpointer.
        3. The graph.invoke() call in the API layer RETURNS (with an interrupt event).
        4. The API surfaces the payload to the human (UI, webhook, admin interface).
        5. Human responds via a separate API call:
               graph.invoke(Command(resume=decision), config=thread_config)
           where thread_config carries the SAME thread_id — checkpointer restores state.
        6. Execution resumes at this node. interrupt() returns the human's decision.
        7. Command(goto=...) routes to the correct next node.

    PAYLOAD DESIGN — send only what the reviewer needs:
        - The original query
        - Why retrieval failed (critic score and reasoning)
        - How many strategies were tried
        - The last reformulated query that was attempted

    RESUME VALUES (from API layer):
        True  → approved: give the retriever one final attempt with original query
        False → rejected: end the run and surface the failure to the user
    """
    # Build a minimal, human-readable review payload.
    # Do NOT dump the full state — retrieved_chunks are large and irrelevant
    # to the human reviewer's decision.
    review_payload = {
        "message": (
            "Retrieval quality is below acceptable threshold after all "
            f"{settings.max_retrieval_retries} automated strategies. "
            "Please review and decide whether to approve one final attempt."
        ),
        "original_query": state.get("query", ""),
        "last_query_attempted": state.get("rewritten_query", ""),
        "strategies_tried": state.get("retrieval_attempts", []),
        "final_critic_score": state.get("critic_score", 0.0),
        "critic_reasoning": state.get("critic_reasoning", ""),
        "threshold": settings.critic_relevance_threshold,
        "instructions": {
            "approve": "Send Command(resume=True) to allow one final retrieval attempt.",
            "reject":  "Send Command(resume=False) to end this run with a failure response.",
        },
    }

    logger.info(
        "human_review_interrupt_triggered",
        extra={
            "query": state.get("query", "")[:80],
            "attempts": len(state.get("retrieval_attempts") or []),
            "critic_score": state.get("critic_score"),
        }
    )

    # interrupt() halts execution and returns the payload to the caller.
    # Execution resumes here when graph.invoke(Command(resume=decision)) is called.
    decision = interrupt(review_payload)

    if decision:
        # Human approved: reset to original query, one final retrieval attempt.
        logger.info("human_review_approved_resuming_retrieval")
        return Command(
            goto=NODE_RETRIEVER,
            update={
                "rewritten_query": state.get("query", ""),
                "retrieval_strategy": "human_approved_retry",
                "requires_human_review": False,
            },
        )
    else:
        # Human rejected: surface failure to user.
        logger.info("human_review_rejected_ending_run")
        return Command(
            goto="__end__",
            update={
                "final_answer": (
                    "I was unable to find sufficient information to answer this question "
                    "after exhausting all retrieval strategies. "
                    "Please rephrase your question or check if the relevant documents "
                    "have been indexed."
                ),
                "requires_human_review": False,
                "eval_passed": False,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> object:
    """
    Assemble and compile the full RAG StateGraph.

    Args:
        checkpointer: LangGraph checkpointer instance. If None, defaults to
                      SqliteSaver with a local file. Pass MemorySaver() for tests.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for .invoke() or .stream().

    WHY build_graph() AS A FUNCTION (not module-level):
        The checkpointer often requires a context manager (SqliteSaver uses
        'with' statement). Building the graph lazily in a function lets the
        caller control the checkpointer lifecycle — critical for testing
        (MemorySaver) vs production (SqliteSaver or PostgresSaver).
    """
    builder = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node(NODE_QUERY_ANALYZER, query_analyzer_node)
    builder.add_node(NODE_RETRIEVER,      retriever_node)
    builder.add_node(NODE_CRITIC,         critic_node)
    builder.add_node(NODE_REFORMULATE,    reformulation_node)
    builder.add_node(NODE_HUMAN_REVIEW,   human_review_node)
    builder.add_node(NODE_GENERATOR,      generator_node)
    builder.add_node(NODE_EVALUATOR,      evaluator_node)

    # ── Fixed edges (always taken) ────────────────────────────────────────────
    builder.add_edge(START,               NODE_QUERY_ANALYZER)
    builder.add_edge(NODE_QUERY_ANALYZER, NODE_RETRIEVER)
    builder.add_edge(NODE_RETRIEVER,      NODE_CRITIC)
    # Reformulate → Retriever: the cycle that makes this self-healing.
    builder.add_edge(NODE_REFORMULATE,    NODE_RETRIEVER)
    # Generator → Evaluator: always evaluate after generation.
    builder.add_edge(NODE_GENERATOR,      NODE_EVALUATOR)
    builder.add_edge(NODE_EVALUATOR,      END)

    # ── Conditional edge: the self-healing routing logic ──────────────────────
    builder.add_conditional_edges(
        NODE_CRITIC,
        route_after_critic,
        {
            # Map return values of route_after_critic() → node names.
            # "__end__" is LangGraph's internal name for END in routing maps.
            "generator":    NODE_GENERATOR,
            "reformulate":  NODE_REFORMULATE,
            "human_review": NODE_HUMAN_REVIEW,
            "__end__":      END,
        },
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    compiled = builder.compile(
        checkpointer=checkpointer,
        # interrupt_before / interrupt_after not used here — we use the
        # dynamic interrupt() inside human_review_node instead.
        # Dynamic interrupts are conditional; static breakpoints are not.
    )

    logger.info("graph_compiled_successfully")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory with SqliteSaver — the production-local entry point
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "checkpoints.db"
# SQLite file created in the project root.
# Add to .gitignore — it's runtime state, not source code.


def get_graph():
    """
    Returns a compiled graph with SqliteSaver checkpointer.

    Called by the FastAPI layer (Phase 5) and the CLI runner below.

    WHY SqliteSaver here, MemorySaver in tests:
        SqliteSaver writes a file. Tests run in parallel and in CI where
        file state leaks between test cases. MemorySaver is ephemeral and
        isolated. The build_graph(checkpointer=...) parameter makes this
        injectable without if/else in every call site.
    """
    import sqlite3
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer=checkpointer)
    # checkpointer = SqliteSaver.from_conn_string(DB_PATH)
    # return build_graph(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner — smoke test the full graph end-to-end
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run a single query through the full graph to verify end-to-end wiring.

    Prerequisites:
        - docker compose up -d (Qdrant + Redis running)
        - python ingestion/indexer.py (collection created)
        - At least a few documents indexed via Embedder

    Usage:
        python graph/supervisor.py
    """
    import uuid
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    log = structlog.get_logger()

    test_query = "What is the main purpose of this system?"
    thread_id = str(uuid.uuid4())

    log.info("smoke_test_start", query=test_query, thread_id=thread_id)

    graph = get_graph()

    config = {"configurable": {"thread_id": thread_id}}
    state = initial_state(query=test_query, user_id="smoke_test", thread_id=thread_id)

    try:
        result = graph.invoke(state, config=config)

        print("\n" + "="*60)
        print("SMOKE TEST RESULT")
        print("="*60)
        print(f"Query              : {result.get('query')}")
        print(f"Query type         : {result.get('query_type')}")
        print(f"Retrieval attempts : {result.get('retrieval_attempts')}")
        print(f"Critic score       : {result.get('critic_score')}")
        print(f"Faithfulness       : {result.get('faithfulness_score')}")
        print(f"Eval passed        : {result.get('eval_passed')}")
        print(f"Error              : {result.get('error')}")
        print(f"\nAnswer:\n{result.get('final_answer', 'No answer generated.')}")
        print("="*60)

    except Exception as e:
        log.error("smoke_test_failed", error=str(e))
        raise