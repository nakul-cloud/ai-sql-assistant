# AI SQL Assistant 📊🤖

## Overview
A **Enterprise AI SQL Analytics Assistant** that lets business users query any SQL Server database with natural language.  The system automatically extracts schema metadata, generates semantic and structural chunks, embeds them with a hybrid dense‑+‑sparse model (BAAI/bge‑m3), indexes them in **Qdrant**, and answers queries via **Google Gemini Flash**.

- 📂 **Modular, production‑ready architecture**
- ⚡ **Hybrid retrieval** (dense + BM25) for fast, accurate results
- 🔄 **Incremental re‑indexing** on CSV upload or schema changes
- 🧠 **LLM description caching** to cut cost and latency
- 📅 **Nightly APScheduler** for full re‑index

---

## Tech Stack
| Layer | Tool | Notes |
|------|------|------|
| **Frontend** | Streamlit | Simple UI – chat & CSV upload |
| **Embedding** | BAAI/bge‑m3 (local) | Dense 1024‑dim + sparse BM25 |
| **Vector DB** | Qdrant (Docker) | Two collections (`sql_table_schemas`, `query_cache`) |
| **Hybrid Search** | Qdrant RRF Fusion | Dense + sparse merged |
| **Query Cache** | Qdrant cosine‑sim (threshold 0.92) |
| **Description Cache** | Local JSON file | Hash‑keyed by schema signature |
| **Intent Router** | Regex + Gemini Flash Lite |
| **SQL Generator** | Google Gemini Flash |
| **NL Generator** | Google Gemini Flash |
| **Scheduler** | APScheduler (background thread) |
| **DB Driver** | SQLAlchemy + pyodbc |
| **Data Processing** | pandas |

---

## Architecture & Workflows
```mermaid
flowchart TD
    %% Phase 1 – Offline Indexing
    subgraph Phase1 [Phase 1 – Offline Indexing]
        A[🗄️ SQL Server (primary source)] --> T1[App Startup – Full Index]
        A --> T2[Scheduler – Nightly Re‑index]
        A --> T3[Schema Change – Incremental Re‑index]
        CSV[📄 CSV Upload (secondary source)] --> STAGE[Stage to SQL (CREATE TABLE AS SELECT)]
        STAGE -.-> T3
        T1 & T2 & T3 --> SE[Schema Extractor]
        SE --> SC[Structural Chunk]
        SE --> SEM[Semantic Chunk]
        SC & SEM --> EMB[BAAI/bge‑m3]
        EMB --> QD[Qdrant Upsert (sql_table_schemas)]
    end

    %% Phase 2 – Online Query Pipeline
    subgraph Phase2 [Phase 2 – Online Query Pipeline]
        U[👤 User Question] --> R[Regex Pre‑Check]
        R -->|CHAT| CHAT[General Chat Response]
        R -->|SQL_QUERY| QC[Query Cache (cosine similarity)]
        QC -->|HIT| ANS1[Cached Answer]
        QC -->|MISS| HR[Hybrid Retriever (RRF)]
        HR --> QDR[Qdrant Schema Index]
        QDR --> SCB[Schema Context Builder]
        SCB --> SQLG[SQL Generator (Gemini)]
        SQLG --> VAL[SQL Validator (SELECT‑only guard)]
        VAL --> EXEC[SQL Server Execution]
        EXEC --> NL[NL Generator (Gemini)]
        NL --> STORE[Store in Query Cache]
        STORE --> ANS2[User Answer]
    end
```

---

## Project Structure
```
ai-sql-assistant/
│
├─ app.py                         # Streamlit entry point
│
├─ database/                      # DB layer
│   ├─ __init__.py
│   ├─ sql_server.py               # Engine & test connection
│   ├─ schema_manager.py           # Raw metadata extraction
│   └─ csv_uploader.py             # CSV → SQL + incremental re‑index
│
├─ indexing/                       # Offline pipeline
│   ├─ __init__.py
│   ├─ embedder.py                 # BAAI/bge‑m3 wrapper
│   ├─ schema_extractor.py         # Formats schema for chunking
│   ├─ chunk_builder.py            # Builds structural & semantic chunks
│   ├─ semantic_description.py      # LLM description + cache
│   ├─ qdrant_uploader.py          # Upserts chunks to Qdrant
│   └─ index_manager.py            # Startup / scheduler triggers
│
├─ retrieval/                      # Online pipeline
│   ├─ __init__.py
│   ├─ query_router.py             # Regex + LLM intent routing
│   ├─ table_retriever.py          # Hybrid RRF search
│   └─ query_cache.py              # Qdrant cache read/write
│
├─ analysis/                       # Helpers for building prompts
│   └─ schema_context.py
│
├─ llm/                            # LLM wrappers
│   ├─ query_ai.py
│   └─ response_generator.py
│
├─ workflow/                       # Orchestrates end‑to‑end flow
│   ├─ process_query.py
│   └─ query_executor.py
│
├─ pages/                          # Streamlit pages
│   ├─ chat_page.py
│   └─ upload_page.py
│
├─ logs/                           # Log files (app.log)
├─ data/uploads/                   # CSV uploads
├─ cache/                          # Description cache JSON
├─ .env                            # Configuration (DB credentials, API keys)
└─ README.md                       # ← you are reading it!
```

---

## Quick Start
1. **Clone & install**
```bash
git clone https://github.com/nakul-cloud/ai-sql-assistant.git
cd ai-sql-assistant
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
2. **Create `.env`** (copy from the template in the repo).  It already contains your DB credentials, Qdrant host, Gemini API key, etc.
3. **Start Qdrant** (Docker)
```bash
docker run -d -p 6333:6333 \
  --name qdrant \
  qdrant/qdrant
```
4. **Run the app**
```bash
streamlit run app.py
```
5. **Test components** (optional)
```bash
python -m database.sql_server          # test DB connection
python -m database.csv_uploader        # test CSV upload
python -m indexing.embedder            # test embeddings
python -m indexing.schema_extractor    # test schema extraction
```

---

## How It Works (Step‑by‑Step)
### 1️⃣ Offline Indexing (run once at startup or nightly)
1. **Extract schema** via `indexing/schema_extractor.py` (calls `database/schema_manager`).
2. **Build chunks** – a *structural* chunk (columns, PKs, sample rows) and a *semantic* chunk (LLM‑generated description, cached).
3. **Embed** each chunk with `indexing/embedder.py` (dense + sparse vectors).
4. **Upsert** into Qdrant collection `sql_table_schemas`.

### 2️⃣ CSV Upload (incremental)
1. User uploads a CSV via Streamlit → `database/csv_uploader.upload_csv_and_index`.
2. CSV is parsed, sanitized, and written to a new SQL table (`csv_<name>`).
3. `incremental_reindex` (wired later) extracts that table’s metadata, builds chunks, embeds, and upserts only the affected data.

### 3️⃣ Online Query
1. **Regex pre‑check** – quick routing for greetings or obvious SQL queries.
2. If ambiguous, **LLM intent classifier** decides between chat, SQL, or schema‑related.
3. **Query cache** – cosine‑similarity search in `query_cache`; hit returns cached NL answer.
4. **Hybrid retrieval** – dense + BM25 RRF search against `sql_table_schemas` to fetch top‑k relevant tables.
5. **Schema context** – construct a prompt with column info & sample values.
6. **SQL generation** – Gemini Flash produces a `SELECT` statement.
7. **Validator** – ensures only `SELECT` queries are sent to the DB.
8. **Execute** on SQL Server, fetch results.
9. **NL generation** – Gemini Flash converts result set into a user‑friendly answer.
10. **Cache store** – embeds the user query + response for future hits.

---

## Testing & CI
- Each module contains a `if __name__ == "__main__":` block with a **stand‑alone test** that prints clear success/failure messages.
- Run `pytest` (or simply `python -m <module>`) to verify individual components.
- CI pipeline (GitHub Actions) can run these tests on push to `main`.

---

## Contributing
1. Fork the repo.
2. Create a feature branch.
3. Ensure all new code follows the **single‑purpose function** rule and includes a test block.
4. Submit a PR – CI will run the component tests.

---

## License
MIT © 2026 Nakul

---

*Built with love for the enterprise AI community.*
