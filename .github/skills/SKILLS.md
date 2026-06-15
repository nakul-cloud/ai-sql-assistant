---
name: enterprise-ai-sql-assistant
description: >
  Architecture and implementation guide for an Enterprise AI SQL Analytics Assistant that
  lets users query a SQL Server database using natural language. Use this skill whenever
  working on any part of this project — indexing pipelines, query routing, vector search,
  SQL generation, caching layers, CSV uploads, or Streamlit UI. Triggers on any mention
  of the indexing pipeline, Qdrant, hybrid retrieval, query cache, regex pre-check,
  semantic chunking, CSV upload flow, schema extractor, BAAI/bge-m3, Gemini Flash SQL
  generation, or any file in the project's directory structure.
---

# Enterprise AI SQL Analytics Assistant

A production-ready system that lets business users ask natural language questions and
receive SQL-backed answers from a SQL Server database. It uses hybrid vector search
(Qdrant), LLM-based SQL generation (Gemini Flash), and a layered caching strategy to
stay fast and cost-efficient at scale (~100 tables).

---

## System Overview

Two major phases:

| Phase | Name | When it runs |
|---|---|---|
| 1 | Offline Indexing Pipeline | At startup, nightly, on schema change, on CSV upload |
| 2 | Online Query Pipeline | On every user message in the Streamlit chat |

Two data sources feed into the same pipeline:

| Source | Type | Ingestion trigger |
|---|---|---|
| SQL Server | Primary | App startup · Nightly scheduler · Schema change |
| CSV Upload | Secondary | On demand → staged via `CREATE TABLE AS SELECT` → incremental re-index |

---

## Phase 1 — Offline Indexing Pipeline

### Flow

```
SQL Server ──► App Startup
           ──► Nightly Scheduler    ──► Schema Extractor ──► Structural Chunk ──┐
           ──► Schema Change                                ──► Semantic Chunk  ──┤
                                                                                  ▼
CSV Upload ──► Stage to SQL ──► (triggers Schema Change)          BAAI/bge-m3 embed
                                                                        │
                                                                   Qdrant Upsert
                                                           (sql_table_schemas collection)
```

### Three Indexing Triggers

**Trigger A — App Startup (Full Index)**
File: `indexing/index_manager.py` → `run_startup_index()`
- Checks if Qdrant collection already has points
- Skips if populated; runs `build_full_index()` if empty

**Trigger B — Nightly Scheduler (Full Re-Index)**
File: `indexing/index_manager.py`
- Uses APScheduler background thread
- Drops and rebuilds the entire Qdrant collection at midnight (`hour=0, minute=0`)

**Trigger C — Incremental Re-Index**
File: `indexing/index_manager.py` → `incremental_reindex(table_name)`
- Triggered by: CSV upload OR ALTER TABLE detection
- Deletes old Qdrant chunks for that table only (using `FieldCondition` filter)
- Re-extracts, re-chunks, re-embeds, re-upserts only the affected table

### CSV Upload Ingestion Path

File: `database/csv_uploader.py` → `upload_csv_and_index(file_path, table_name)`

Four steps in order:
1. Parse CSV with Pandas
2. Stage to SQL Server (`upload_df_to_sql`, `if_exists="replace"`)
3. Call `incremental_reindex(table_name)` — makes the table queryable in seconds
4. Bust the schema metadata cache via `fetch_database_metadata(force_refresh=True)`

### Hybrid Chunking — Two Chunk Types Per Table

**Structural Chunk** (for exact column/type matching)
```
Table: <name>
Columns: col1 (type), col2 (type), ...
Primary Keys: ...
Row Count: ...
```

**Semantic Chunk** (for concept/intent matching)
- A 3-sentence LLM-generated business description
- Answers: what business questions can this table answer? what data does it hold?
- Always served from cache (see below) — LLM called only on cache miss

### LLM Description Cache

File: `indexing/semantic_description.py`
Cache file: `indexing/description_cache.json`

- Cache key = SHA-256 hash of `table_name + col_names + col_types`
- Cache HIT → return stored description (zero LLM calls)
- Cache MISS → call LLM once, persist to JSON, return result
- **Why mandatory**: 100-table nightly re-index without caching = 100 LLM calls every night

### Embedding

File: `indexing/embedder.py`
Model: `BAAI/bge-m3` (local, free, fp16)

Each chunk produces:
- `dense`: 1024-dim float vector
- `sparse`: dict of `{token_id: weight}` (BM25 lexical weights)

### Qdrant Upsert

File: `indexing/qdrant_uploader.py`
Collection: `sql_table_schemas`
Vector config: Named vectors — `"dense"` (1024-dim) + `"sparse"` (BM25)

Payload per point:
```json
{
  "table_name": "...",
  "chunk_type": "structural | semantic",
  "text": "...",
  "columns": [...]
}
```

---

## Phase 2 — Online Query Pipeline

### Flow

```
User Question (NL)
       │
       ▼
Regex Pre-Check ──► obvious chat ──► General Response
       │
       │ unclear → pass through
       ▼
Intent Classifier (LLM) ──► SQL/schema intent?
       │                              │
       │ yes                          │ cache hit ──► Return Cached Answer
       ▼                              │
Query Cache Check ◄───────────────────┘
       │ cache miss
       ▼
Hybrid Retriever (Qdrant RRF) ◄── Qdrant DB (sql_table_schemas)
       │ top 3 tables
       ▼
Schema Context Builder
       │ prompt with schema
       ▼
SQL Generator (Gemini Flash)
       │
       ▼
SQL Validator (SELECT-only guard)
       │
       ▼
SQL Server (query execution)
       │
       ▼
NL Response Generator (Gemini Flash)
       │
       ▼
Store in Query Cache ──► User sees Answer + Table
```

### Step 2.1 — Regex Pre-Check

File: `retrieval/query_router.py` → `pre_check_intent(user_query)`

- Runs before any LLM call (0ms cost)
- Eliminates ~40% of messages (greetings, thanks, confirmations)
- Returns `"CHAT"`, `"SQL_QUERY"`, or `None` (ambiguous → send to LLM)

Pattern sets:
- `CHAT_PATTERNS`: greetings, thanks, goodbye variants
- `SQL_PATTERNS`: show/list/find/count/sum + table/record/row keywords

Full router: `route_query()` — calls `pre_check_intent()` first, only falls through to `llm_classify_intent()` if result is `None`.

### Step 2.2 — Query Cache

File: `retrieval/query_cache.py`
Collection: `query_cache` (dense-only, 1024-dim)
Threshold: cosine similarity > `0.92`

- Embeds user question with `BAAI/bge-m3` dense vector
- Searches `query_cache` collection for a past match above threshold
- Cache HIT → return stored `nl_response`, `rows`, `columns` directly (skips all pipeline stages)
- Cache MISS → run full pipeline → store result via `store_in_query_cache()`
- Stores up to 50 rows per cached result in payload

**Why useful**: Business MIS users repeat near-identical questions ("Show tickets for Billy George" ≈ "Get all tickets of Billy George").

### Step 2.3 — Hybrid Table Retriever

File: `retrieval/table_retriever.py` → `retrieve_relevant_tables(user_query, top_k=3)`
Collection: `sql_table_schemas`

Uses Qdrant's native RRF (Reciprocal Rank Fusion):
- Prefetch top-20 via dense vector
- Prefetch top-20 via sparse (BM25) vector
- Fuse with `FusionQuery(fusion=Fusion.RRF)`
- Deduplicate and return top-3 table names

### Steps 2.4–2.6 — Unchanged Components

These files are not modified in the revised architecture:

| File | Role |
|---|---|
| `analysis/schema_context.py` | Builds schema context string for LLM prompt |
| `llm/query_ai.py` | SQL generation via Gemini Flash |
| `llm/response_generator.py` | NL response generation via Gemini Flash |
| `workflow/query_executor.py` | Executes validated SQL on SQL Server |

### Main Integration Point

File: `workflow/process_query.py` → `process_user_query(user_query)`

Order of operations:
1. `pre_check_intent()` → return early if obvious CHAT
2. `check_query_cache()` → return early if cache HIT
3. `retrieve_relevant_tables()` → hybrid Qdrant search
4. `generate_schema_context()` → build prompt
5. `generate_sql_query()` → Gemini Flash SQL
6. `execute_sql_query()` → SQL Server
7. `generate_natural_language_response()` → Gemini Flash NL
8. `store_in_query_cache()` → persist for future hits

---

## Directory Structure

```
ai-sql-assistant/
├── app.py                          # Streamlit entry point
│
├── indexing/
│   ├── index_manager.py            # Startup / scheduler / incremental triggers
│   ├── schema_extractor.py         # Pull schemas from SQL Server
│   ├── chunk_builder.py            # Build structural + semantic chunks
│   ├── semantic_description.py     # LLM description generator WITH cache
│   ├── description_cache.json      # Cached LLM descriptions (auto-managed)
│   ├── embedder.py                 # BAAI/bge-m3 wrapper
│   └── qdrant_uploader.py          # Upsert to Qdrant
│
├── retrieval/
│   ├── query_router.py             # Regex pre-check + LLM intent classifier
│   ├── table_retriever.py          # Hybrid Qdrant RRF search
│   └── query_cache.py              # Query cache read/write via Qdrant
│
├── analysis/
│   └── schema_context.py           # UNCHANGED
│
├── llm/
│   ├── query_ai.py                 # UNCHANGED
│   └── response_generator.py       # UNCHANGED
│
├── workflow/
│   ├── process_query.py            # Updated integration points
│   └── query_executor.py           # UNCHANGED
│
├── database/
│   ├── sql_server.py               # UNCHANGED
│   ├── schema_manager.py           # UNCHANGED (uses st.cache_data)
│   └── csv_uploader.py             # Extended with incremental_reindex call
│
├── pages/
│   ├── chat_page.py                # UNCHANGED
│   └── upload_page.py              # Updated to call new csv_uploader
│
└── .env
```

---

## Qdrant Collections

| Collection | Purpose | Vector types |
|---|---|---|
| `sql_table_schemas` | Table schema index — 2 chunks per table | Dense (1024) + Sparse (BM25) |
| `query_cache` | Past query results for cosine-similarity bypass | Dense only (1024) |

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Frontend | Streamlit | Unchanged |
| Embedding | BAAI/bge-m3 (local) | Dense + Sparse in one model, fp16 |
| Vector DB | Qdrant (Docker) | Two collections |
| Hybrid Search | Qdrant RRF Fusion | Dense + BM25 merged |
| Query Cache | Qdrant cosine sim | Threshold: 0.92 |
| Description Cache | Local JSON file | Hash-keyed by schema signature |
| Intent Router | Regex + Gemini Flash Lite | Regex first, LLM only if ambiguous |
| SQL Generator | Google Gemini Flash | Unchanged |
| NL Generator | Google Gemini Flash | Unchanged |
| Scheduler | APScheduler (background thread) | Nightly full re-index |
| DB Driver | SQLAlchemy + pyodbc | Unchanged |
| Data Processing | Pandas | Unchanged |

### Key Dependencies

```
FlagEmbedding          # BAAI/bge-m3
qdrant-client[fastembed]
apscheduler
```

---

## Build Order

### FIRST— Foundation (Indexing)
1. Install Docker, run Qdrant container (`localhost:6333`)
2. Install FlagEmbedding, verify BAAI/bge-m3 runs locally
3. Build `schema_extractor.py` (reuse `schema_manager.py` logic)
4. Build `chunk_builder.py` + `semantic_description.py` with cache
5. Build `embedder.py` + `qdrant_uploader.py`
6. Run full index on all tables — verify in Qdrant UI

###  SECOND — Query Pipeline
7. Build `table_retriever.py` — test hybrid RRF search
8. Build `query_router.py` (regex + LLM triage)
9. Build `query_cache.py`
10. Update `process_query.py` with new integration points
11. Update `csv_uploader.py` with `incremental_reindex` call

### LATER PART (DONT FOCUS FOR NOW) — Polish & Deploy
12. Add APScheduler nightly re-index
13. Add startup index check in `index_manager.py`
14. End-to-end tests across 10+ diverse questions
15. Deploy

---

## Key Design Decisions

**Why two chunk types?**
Structural chunks catch exact column/type queries. Semantic chunks catch business-intent
queries where the user doesn't know the column names. Both together = robust retrieval.

**Why cache LLM descriptions?**
100-table nightly re-index without a cache triggers 100 LLM API calls per night.
With a hash-keyed JSON cache, unchanged tables cost zero — only schema changes trigger
a new LLM call.

**Why regex before intent classifier?**
~40% of chat messages are greetings, thanks, or confirmations. Routing them through the
LLM wastes ~300ms each and adds API cost. Regex handles these in microseconds.

**Why a query cache?**
Business MIS users repeat near-identical questions daily. Cosine similarity at 0.92
threshold catches semantic near-duplicates ("Show tickets for X" ≈ "Get all tickets of X")
and returns stored results without touching SQL Server or the LLM.

**Why BAAI/bge-m3?**
Single model produces both dense (semantic) and sparse (BM25 lexical) vectors.
Running RRF fusion over both vector types beats either alone — especially important
when users query by exact column names (sparse wins) vs. business concepts (dense wins).