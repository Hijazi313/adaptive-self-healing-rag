# Adaptive Multi-Agent RAG System with Self-Healing Retrieval

> A production-grade Retrieval-Augmented Generation system that detects its own retrieval failures and corrects them mid-execution — with hybrid search, critic-gated generation, semantic caching, and a full RAGAS evaluation harness.

---

## The Problem This Solves

Most RAG systems fail silently. They retrieve the wrong chunks, the LLM hallucinates confidently, and nobody knows. This system detects its own retrieval failures and corrects them before generating an answer.

**What that looks like in practice:**

```
Query arrives
    ↓
Retrieve context (hybrid dense + sparse search)
    ↓
CriticAgent scores relevance → 0.4 (below threshold)
    ↓
Reformulate query using HyDE  →  Retrieve again
    ↓
CriticAgent scores relevance → 0.8 (passes)
    ↓
GeneratorAgent produces grounded answer
    ↓
EvaluatorAgent scores faithfulness → 1.0
    ↓
Answer returned with citations and quality scores
```

This loop runs up to 3 times with progressively more powerful reformulation strategies before escalating to human review.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         LANGGRAPH SUPERVISOR         │
                    │  Typed StateGraph + SqliteSaver      │
                    └────────────────┬────────────────────┘
                                     │
        ┌───────────────┬────────────┼────────────┬──────────────┐
        ▼               ▼            ▼            ▼              ▼
  QueryAnalyzer    Retriever      Critic      Generator      Evaluator
     Agent           Agent        Agent        Agent          Agent
        │               │            │            │              │
  Classify type    Hybrid search  Score 0-1   Grounded ans   RAGAS
  Set alpha        Dense + BM25   Route next  + citations    faithfulness
  HyDE/StepBack    Dynamic weight  or escalate
```

### Self-Healing Retrieval Loop

```
[CriticAgent] scores retrieved context
       │
  score < 0.7?
       │
  ┌────┴──────┐
  YES         NO ─────────────────────→ [GeneratorAgent] → [EvaluatorAgent]
  │
  retries < 3?
  │
  ├── Retry 1: HyDE        (embed a hypothetical answer instead of the query)
  ├── Retry 2: Step-Back   (abstract the query to a more general form)
  └── Retry 3: Decompose   (split into atomic sub-questions)
                  │
            retries == 3?
                  │
          [interrupt() → Human Review via POST /hitl/resume]
```

---

## Stack

| Component           | Technology                                  | Why                                                         |
| ------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| Agent Orchestration | LangGraph 0.4+                              | Typed state, conditional cycles, HITL interrupt             |
| Vector Store        | Qdrant 1.9+                                 | Native hybrid search — dense + sparse in one query          |
| Embeddings          | OpenAI text-embedding-3-large               | 3072-dim, highest quality OpenAI embedding                  |
| Sparse / BM25       | FastEmbed (Qdrant/bm25)                     | Local inference, no API call                                |
| LLM                 | GPT-4o (agents) + GPT-3.5-turbo (evaluator) | Quality where it matters, cost where it does not            |
| Evaluation          | RAGAS                                       | Faithfulness, precision, recall, relevancy                  |
| Observability       | LangSmith                                   | Full graph trace per request                                |
| Semantic Cache      | Redis                                       | Cosine similarity cache — bypasses graph on similar queries |
| User Memory         | mem0                                        | Cross-session personalization                               |
| API                 | FastAPI + uvicorn                           | Async, production-grade                                     |
| Config              | Pydantic Settings                           | Typed, validated, fail-fast                                 |

---

## Project Structure

```
adaptive-rag/
│
├── ingestion/                      Phase 1 — Data pipeline
│   ├── indexer.py                  Qdrant collection setup (dense + sparse)
│   ├── embedder.py                 Embed chunks and upsert to Qdrant
│   └── chunkers/
│       ├── __init__.py             BaseChunker interface + factory
│       ├── recursive_chunker.py    Fast, deterministic, good default
│       ├── semantic_chunker.py     Embedding-based topic boundary detection
│       └── proposition_chunker.py  LLM-powered atomic fact extraction
│
├── graph/                          Phase 2 — Agent graph
│   ├── state.py                    Typed RAGState (17 fields, TypedDict)
│   ├── supervisor.py               StateGraph: all nodes, edges, HITL
│   └── reformulation.py            HyDE / Step-Back / Decompose strategies
│
├── agents/                         Phase 2 — Individual agents
│   ├── query_analyzer.py           Classify query type, set dense/sparse weight
│   ├── retriever.py                Execute hybrid search
│   ├── critic.py                   Score retrieved context relevance 0-1
│   ├── generator.py                Grounded answer generation + citations
│   └── evaluator.py                Post-generation faithfulness check
│
├── memory/                         Phase 3 — Memory layer
│   ├── semantic_cache.py           Redis cosine-similarity cache
│   └── user_memory.py              mem0 cross-session user memory
│
├── evaluation/                     Phase 4 — Evaluation harness
│   ├── golden_dataset.json         51 labeled Q&A pairs, 11 categories
│   ├── eval_runner.py              RAGAS scorer per chunking strategy
│   ├── regression_gate.py          Baseline comparison + CI quality gate
│   └── results/                    Per-commit JSON results
│
├── api/                            Phase 5 — API layer
│   └── main.py                     FastAPI: all endpoints + lifespan management
│
├── config.py                       Central Pydantic Settings singleton
├── docker-compose.yml              Qdrant + Redis local infrastructure
├── pyproject.toml                  uv project definition
└── .env.example                    Environment variable reference
```

---

## Chunking Strategies — Treated as a Hyperparameter

| Strategy    | How It Works                                    | Best For              | Cost                             |
| ----------- | ----------------------------------------------- | --------------------- | -------------------------------- |
| Recursive   | Split at paragraphs then sentences then chars   | General documents     | Low — no API calls               |
| Semantic    | Split where sentence embedding similarity drops | Narrative documents   | Medium — embeddings per sentence |
| Proposition | LLM converts paragraphs into atomic facts       | Dense factual corpora | High — LLM call per paragraph    |

RAGAS scores are measured per strategy and stored in `evaluation/results/`. Chunking is treated as a tunable hyperparameter, not a one-time decision.

---

## Memory Architecture

Three independent memory layers serving different needs:

| Layer          | Technology                | Scope                      | Purpose                                   |
| -------------- | ------------------------- | -------------------------- | ----------------------------------------- |
| Semantic Cache | Redis + cosine similarity | Global, query-keyed        | Skip graph execution for similar queries  |
| Session Memory | LangGraph SqliteSaver     | Per thread_id              | Conversational context within one session |
| User Memory    | mem0                      | Per user_id, cross-session | Personalization across conversations      |

---

## Evaluation Harness

51-sample golden dataset covering 11 categories at three difficulty levels (basic / intermediate / advanced).

RAGAS metrics tracked per commit:

| Metric            | What It Measures                                                       |
| ----------------- | ---------------------------------------------------------------------- |
| faithfulness      | Does the answer stay within retrieved context? Measures hallucination. |
| answer_relevancy  | Does the answer address the actual question?                           |
| context_precision | Are retrieved chunks relevant? Measures retrieval noise.               |
| context_recall    | Was all necessary information retrieved? Measures completeness.        |

Regression gate: if faithfulness drops more than 5% from baseline, CI exits with code 1.

---

## API Reference

| Method | Endpoint           | Description                                      |
| ------ | ------------------ | ------------------------------------------------ |
| POST   | /query             | RAG query — full self-healing pipeline           |
| POST   | /ingest            | Index documents (recursive/semantic/proposition) |
| POST   | /hitl/resume       | Resume a human-interrupted graph execution       |
| GET    | /memory/{user_id}  | Inspect stored user memories                     |
| DELETE | /memory/{user_id}  | Delete user memories (GDPR)                      |
| POST   | /admin/cache/flush | Invalidate semantic cache after re-index         |
| GET    | /admin/cache/stats | Cache entry count and configuration              |
| GET    | /health            | Infrastructure liveness (Qdrant + Redis)         |
| GET    | /docs              | Interactive Swagger UI                           |

---

## Local Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker Desktop

### 1. Clone and configure

```bash
git clone https://github.com/your-username/adaptive-rag.git
cd adaptive-rag

cp .env.example .env
# Edit .env and set OPENAI_API_KEY and LANGCHAIN_API_KEY
```

### 2. Start infrastructure

```bash
docker compose up -d
curl http://localhost:6333/healthz
```

### 3. Install dependencies

```bash
uv sync
```

First run downloads the FastEmbed BM25 model (~50MB). Subsequent runs are instant.

### 4. Initialize Qdrant collection

```bash
uv run python ingestion/indexer.py
```

Expected:

```
Collection : adaptive_rag_docs
Status     : green
Dense dims : 3072
Sparse keys: ['sparse']
```

### 5. Start the API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for: `Application startup complete.`

### 6. Index documents and query

```bash
# Index
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your content..."], "source": "doc.txt", "chunk_strategy": "recursive"}'

# Query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question", "user_id": "user_001"}' | python3 -m json.tool
```

---

## Running the Evaluation Harness

```bash
# Evaluate each chunking strategy
uv run python evaluation/eval_runner.py --strategy recursive
uv run python evaluation/eval_runner.py --strategy semantic
uv run python evaluation/eval_runner.py --strategy proposition

# Quick smoke test (5 samples)
uv run python evaluation/eval_runner.py --strategy recursive --limit 5

# Set baseline after a satisfactory run
uv run python evaluation/regression_gate.py --set-baseline recursive

# Check for regression after any code change
uv run python evaluation/regression_gate.py --check recursive

# View strategy comparison table
uv run python evaluation/regression_gate.py --report
```

---

## Example Response

```json
{
  "query": "What is database connection pooling and why is it used?",
  "answer": "Database connection pooling maintains a cache of reusable database connections so that new connections do not need to be established for every request...",
  "source_citations": [
    {
      "source": "knowledge_base_v1.txt",
      "chunk_index": 2,
      "chunk_strategy": "recursive",
      "excerpt": "Database connection pooling is a technique..."
    }
  ],
  "critic_score": 0.9,
  "faithfulness_score": 1.0,
  "eval_passed": true,
  "retrieval_attempts": ["dense_sparse_hybrid"],
  "cache_hit": false,
  "thread_id": "ceef178e-c3ec-473b-a03d-790441580c21",
  "requires_human_review": false
}
```

---

## Key Design Decisions

**Hybrid search with dynamic alpha weighting**
Semantic queries need dense retrieval. Exact-term queries (error codes, product names) need sparse BM25. The QueryAnalyzerAgent classifies each query and sets the weight dynamically. This is what production RAG at scale actually does — not a fixed split.

**LLM Critic before generation**
Cosine similarity measures vector proximity, not whether the chunk actually answers the question. An LLM critic makes a semantic judgement. Without it, the system cannot detect when to retry. This is the core of the self-healing loop.

**Proposition-level chunking**
Standard chunking stores 4-6 facts per chunk. If a query needs only 1, the other 5 are noise — hurting context_precision. Proposition chunking stores atomic facts independently. The RAGAS harness measures the improvement with numbers.

**Deterministic point IDs (UUID5)**
Re-indexing the same document with random UUIDs duplicates every point in Qdrant. UUID5 derived from source + chunk_index + content makes re-indexing idempotent. Same document, same ID, upsert overwrites cleanly.

**SqliteSaver for HITL persistence**
HITL interrupts may wait minutes for human response. MemorySaver loses state on process restart. SqliteSaver writes to disk — interrupted graph state survives restarts and resumes correctly with the same thread_id.

---

## TODO

### High Priority

- [ ] LangSmith startup verification — explicitly log active tracing project at startup
- [ ] mem0 Docker integration — add second Qdrant service on port 6335 to docker-compose.yml for user long-term memory
- [ ] Async proposition chunking — batch LLM calls with asyncio.gather; currently sequential

### Evaluation

- [ ] Run baseline RAGAS scores across all three chunking strategies with a realistic corpus
- [ ] Commit baseline results JSON to evaluation/results/ and add comparison table to README
- [ ] GitHub Actions workflow — run regression_gate.py --check on every pull request

### Production Hardening

- [ ] Context window expansion — fetch chunk_index +/- 1 neighbors alongside retrieved chunk; metadata already supports this
- [ ] Surgical cache invalidation — track source_doc_id per cache entry; invalidate only affected entries on re-index
- [ ] Conditional critic — skip CriticAgent when QueryAnalyzer confidence is high to reduce latency
- [ ] PostgresSaver — replace SqliteSaver for multi-worker uvicorn deployments
- [ ] Per-user rate limiting on /query using Redis counters

### Features

- [ ] Streaming responses — graph.astream() with Server-Sent Events on /query
- [ ] Document management — DELETE /documents/{source} that removes chunks and invalidates cache
- [ ] Multi-tenant collections — namespace Qdrant collections per org_id for data isolation

---

## Known Issues

| Issue                                                    | Impact                                | Workaround                                                                                |
| -------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| chunk_index always 0 when ingesting multiple short texts | Citations show index 0 for all chunks | Pass one long document per ingest call; chunks will have correct indices                  |
| mem0 requires Qdrant on port 6335                        | User memory non-functional            | Add qdrant-mem0 service to docker-compose.yml                                             |
| Qdrant client/server version must match                  | UserWarning on startup                | Pin qdrant-client version in pyproject.toml to match server version in docker-compose.yml |

---

## License

MIT
