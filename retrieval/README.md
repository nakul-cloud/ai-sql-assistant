# Retrieval & Router Layer

This directory directs incoming user queries, performs hybrid vector searches to identify target tables, and manages semantic query caching to reuse database answers.

---

## Retrieval Flow

```mermaid
flowchart TD
    UserQuery[User Question] --> Router[query_router.py]
    Router -->|Regex / LLM Check| Intent{Intent?}
    
    Intent -->|CHAT| ChatPage[Conversational Response]
    Intent -->|SCHEMA_INFO| SchemaPage[System Database Metadata]
    Intent -->|SQL_QUERY| CacheCheck[query_cache.py]
    
    CacheCheck -->|HIT >= 0.92 similarity| ReturnCache[Return Cached Answer]
    CacheCheck -->|MISS| Retriever[table_retriever.py]
    
    Retriever -->|Hybrid Dense/Sparse RRF Query| Qdrant[(Qdrant Vector DB)]
    Qdrant -->|Top Table Matches| SelectedTables[Selected Schema Context]
```

---

## File Registry

### 1. `query_router.py`
Determines user intent using a tiered classification strategy:
- **Regex Check**: Screens inputs against pre-defined patterns (greetings, thanks, explicit database commands). Routes common requests instantly without calling APIs.
- **Dynamic Value Bypass**: Automatically overrides schema-description intents when queries contain specific table categorical sample values (e.g. "female") combined with filtering prepositions/conjunctions ("for", "only", "by"). This ensures specific data queries route directly to the SQL execution pipeline instead of metadata queries.
- **LLM Fallback**: Ambiguous queries are classified by Groq Llama 3 into one of three categories:
  1. `CHAT`: Conversational messages.
  2. `SQL_QUERY`: Requests to query or analyze table data.
  3. `SCHEMA_INFO`: Inquiries about database tables, columns, or keys.

### 2. `table_retriever.py`
Finds database tables using a hybrid search algorithm in Qdrant:
- **Parallel Querying**: Fetches tables using dense vector matching (for conceptual similarities) and sparse vector matching (for exact schema names) simultaneously.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse vector matches to yield a unified relevance list.
- **Observability**: Prints scores, rankings, chunk types, and text snippets to the terminal for debugging:
  ```
  Rank 01 | Score: 0.0321 | Table: dbo.csv_employees | Type: structural | Snippet: Table: dbo.csv_employees Columns...
  Rank 02 | Score: 0.0164 | Table: dbo.csv_employees | Type: semantic   | Snippet: Table: dbo.csv_employees Tracks...
  ```

### 3. `query_cache.py`
Caches natural language responses in a separate Qdrant collection:
- **Semantic Matching**: Checks incoming questions against previously cached queries using cosine similarity.
- **Fast Hits**: If a query matches a cached item with a similarity score `>= 0.92`, it returns the cached response instantly, avoiding redundant database connections and API costs.
- **Storage Limits**: Stores only the first 50 records of query results to keep the vector payloads light.

---

## Router Intent Definitions

| Intent | Routing Rule | Next Pipeline Step |
|:---|:---|:---|
| **`CHAT`** | Matches greetings, thank-you messages, and conversational text. | Simple Groq conversational response. |
| **`SCHEMA_INFO`** | Inquiries about database tables, row counts, columns, or schemas. | Formats available metadata schema summaries via LLM. |
| **`SQL_QUERY`** | Requests for statistics, aggregations, counts, or multi-table analytical results. | Runs retrieval, SQL generation, execution, enrichment, and NL response. |
| **`DESCRIBE`** | Asks "what does this table show" or "summarize the database". | Direct English explanation of tables/schemas using cached definitions. |
| **`DATA_PREVIEW`** | Requests to show/see/preview raw sample records of a table. | Generates `SELECT * FROM table` directly to show rows without LLM SQL generation. |
| **`SCHEMA_EXPLANATION`**| Asks for a specific column's definition (e.g. "what is ssc_p?"). | Looks up column metadata + semantic descriptions and explains in plain English. |
| **`CONVERSATION_SUMMARY`**| Requests a summary of what was previously discussed. | Synthesizes a conversational recap of facts stored in mem0 Cloud. |
| **`TEMPORAL`** | Questions asking about today's date, current time, or current year. | Returns a helpful pre-defined temporal scope refusal message. |
| **`GENERAL_KNOWLEDGE`** | Requests about people, news, external facts, or general knowledge. | Returns a helpful pre-defined general knowledge refusal message. |


---

## Verification

Test the retrieval components independently:

```bash
# Verify intent routing
.venv\Scripts\python.exe -m retrieval.query_router

# Test hybrid RRF table retrieval
.venv\Scripts\python.exe -m retrieval.table_retriever

# Test semantic caching
.venv\Scripts\python.exe -m retrieval.query_cache
```
