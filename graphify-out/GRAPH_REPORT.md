# Graph Report - .  (2026-06-16)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 314 nodes · 601 edges · 21 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2e2d9149`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `fetch_database_metadata()` - 25 edges
2. `process_user_query()` - 25 edges
3. `get_engine()` - 22 edges
4. `get_llm()` - 22 edges
5. `embed_text()` - 12 edges
6. `fetch_single_table()` - 11 edges
7. `FlowPDF` - 11 edges
8. `_fetch_metadata_from_db()` - 10 edges
9. `Chunk` - 10 edges
10. `build_table_chunks()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `init_background_scheduler()` --calls--> `start_scheduler()`  [EXTRACTED]
  app.py → indexing/scheduler.py
- `test_db_connection()` --calls--> `get_engine()`  [EXTRACTED]
  app.py → database/sql_server.py
- `build_intent_chain()` --calls--> `get_llm()`  [EXTRACTED]
  retrieval/query_router.py → llm/llm_client.py
- `enrich_sql_result()` --calls--> `get_engine()`  [EXTRACTED]
  analysis/result_enricher.py → database/sql_server.py
- `process_user_query()` --calls--> `enrich_sql_result()`  [EXTRACTED]
  workflow/process_query.py → analysis/result_enricher.py

## Import Cycles
- None detected.

## Communities (21 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (57): generate_schema_context(), get_all_table_summaries(), infer_semantic_description(), analysis/schema_context.py ─────────────────────────── Generates schema context, Dynamic fallback semantic inference — used only when the LLM-generated     cache, Builds a summary of all tables and columns in the database for high-level descri, Builds LLM-friendly schema text for retrieved tables., embed_text() (+49 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (46): process_csv(), DataFrame, database/csv_uploader.py ──────────────────────── Handles CSV file parsing, vali, Write a DataFrame to a SQL Server table.      Args:         df:         The Data, Full CSV pipeline: parse → upload to SQL → trigger Qdrant re-index.      This is, Clean a user-provided table name so it's safe for SQL Server.     - Strips file, Clean column names to be SQL Server-safe.     - Strip whitespace     - Replace s, Parse and validate a CSV file.      Returns:         (DataFrame, None) on succes (+38 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (34): build_all_chunks(), _build_semantic_chunk(), _build_structural_chunk(), build_table_chunks(), Chunk, indexing/chunk_builder.py ───────────────────────── Builds two chunks per table, Build both chunks (structural + semantic) for a single table.      Args:, Build chunks for ALL tables. Used during full index.      Args:         all_tabl (+26 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (31): build_describe_chain(), llm/describe_generator.py ───────────────────────── Handles 'explain this datase, generate_text(), _get_groq_client(), get_llm(), llm/llm_client.py ───────────────── Groq LLM client for the Enterprise AI SQL An, Lazy singleton Groq client., Returns a ChatGroq instance with dynamic fallback chains (Groq -> OpenAI -> Gemi (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (21): clear_cache(), _compute_table_hash(), generate_content_with_retry(), _generate_description_via_llm(), get_cache_stats(), get_semantic_description(), _is_retryable(), _LLMResponse (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (19): _init_mem0(), _build_memory_content(), clear_user_memory(), contextualize(), get_all_memories(), _get_client(), get_context_for_prompt(), init_memory() (+11 more)

### Community 6 - "Community 6"
Cohesion: 0.19
Nodes (10): init_background_scheduler(), preload_embedding_model(), test_db_connection(), get_model(), _get_streamlit_cached_model(), _load_model_resource(), indexing/embedder.py ──────────────────── Wrapper around BAAI/bge-m3 for generat, Actually load BGE-M3 model and run warmup. (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (13): Pattern, _build_dynamic_sql_pattern(), build_intent_chain(), invalidate_schema_cache(), llm_classify_intent(), _load_schema_terms(), pre_check_intent(), retrieval/query_router.py ───────────────────────── Routes user queries into dis (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (4): FPDF, FlowPDF, gen(), Generate a comprehensive PDF documenting the AI SQL Assistant end-to-end flow.

### Community 9 - "Community 9"
Cohesion: 0.24
Nodes (9): build_sql_chain(), clean_sql_query(), generate_sql_query(), Any, llm/query_ai.py ─────────────── Production-grade AI SQL generation engine using, Cleans LLM SQL response by removing markdown blocks and whitespace.     Extracts, Validates that the generated SQL is safe and is a SELECT or WITH statement., Main entry point for generating SQL from user query and schema context. (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.27
Nodes (9): enforce_row_limit(), execute_sql_query(), Any, DataFrame, workflow/query_executor.py ────────────────────────── Production-grade SQL execu, Enforces row limiting safely for large result prevention.     In T-SQL, this mea, Converts dataframe into structured response: columns, rows (as dicts), row_count, Executes T-SQL query safely. (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.40
Nodes (4): enrich_sql_result(), Any, analysis/result_enricher.py ─────────────────────────── Semantic enrichment laye, Enriches the raw SQL query result with semantic insights and statistical summari

### Community 12 - "Community 12"
Cohesion: 0.40
Nodes (4): Any, llm/langchain_agent.py ────────────────────── Optional autonomous LangChain SQL, Executes a LangChain SQL agent to autonomously query the SQL Server database., run_autonomous_sql_agent()

## Knowledge Gaps
- **9 isolated node(s):** `Any`, `Engine`, `Any`, `Any`, `Any` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `fetch_database_metadata()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Why does `get_engine()` connect `Community 1` to `Community 10`, `Community 11`, `Community 12`, `Community 6`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `get_llm()` connect `Community 3` to `Community 0`, `Community 9`, `Community 5`, `Community 7`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **What connects `Any`, `analysis/result_enricher.py ─────────────────────────── Semantic enrichment laye`, `Enriches the raw SQL query result with semantic insights and statistical summari` to the rest of the system?**
  _134 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06144393241167435 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.0693815987933635 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.09390243902439024 - nodes in this community are weakly interconnected._