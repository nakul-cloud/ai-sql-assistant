---
name: query-transformation
description: >
  Apply this skill when implementing, debugging, or extending the Advanced Query
  Transformation layer in the Enterprise AI SQL Analytics Assistant. Triggers on:
  "query decomposition", "sub-query", "multi-part query", "compound question",
  "query rewriting", "synonym alignment", "step-back prompting", "query transformation",
  "split query", "combine results", or any mention of transforming user questions
  before SQL generation. Do NOT apply to: SQL generation itself (query_ai.py),
  NL response generation (response_generator.py), intent routing (query_router.py),
  or memory contextualization (mem0_manager.py) — those are separate concerns.
---

# Advanced Query Transformation Layer

## Overview

The Query Transformation layer sits **after** intent routing and memory contextualization
but **before** cache lookup and SQL generation. It takes a standalone analytical
question and applies three transformation strategies to maximize SQL generation
accuracy and coverage.

```
User Question
    │
    ▼
Intent Router (query_router.py)
    │ SQL_QUERY intent
    ▼
Memory Contextualization (mem0_manager.py)
    │ standalone query
    ▼
┌─────────────────────────────────┐
│  QUERY TRANSFORMATION LAYER    │  ← NEW
│  ┌───────────────────────────┐ │
│  │ 1. Query Decomposition    │ │
│  │ 2. Query Rewriting        │ │
│  │ 3. Step-Back Expansion    │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
    │ transformed query (or sub-queries)
    ▼
Cache Check → Retrieval → SQL Gen → Execute → NL Response
```

---

## Strategy 1 — Query Decomposition (Multi-Part Splitting)

### What It Does

Detects compound questions that span multiple unrelated tables or ask
multiple independent analytical questions, and splits them into
separate sub-queries that are each executed independently.

### Why It's Critical

A single SQL query **cannot** bridge two completely unrelated tables
(no FK relationship). Without decomposition:

- *"What is the average salary of placed candidates, and how much did
  Agriculture spend on AI in 2025?"* → The SQL generator tries to write
  one query joining `placement_data` and `corporate_ai_adoption`. This
  either generates a Cartesian product or a CANNOT_GENERATE error.

With decomposition:
- Sub-query A: "What is the average salary of placed candidates?" → runs on `placement_data`
- Sub-query B: "How much did Agriculture spend on AI in 2025?" → runs on `corporate_ai_adoption`
- Both results are passed to the NL response generator which synthesizes a combined answer.

### Detection Heuristics

A query is compound if it contains:
- Coordinating conjunctions joining independent clauses: "and", "also", "as well as", "plus"
- Explicit list markers: commas followed by a separate analytical question
- Multiple question marks
- The LLM decomposer confirms it has ≥ 2 distinct analytical sub-questions

### Implementation

```python
# llm/query_transformer.py

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query decomposition assistant for a SQL analytics chatbot.

Given a user's analytical question, determine if it contains multiple
INDEPENDENT sub-questions that would require querying DIFFERENT database
tables or computing DIFFERENT metrics.

Rules:
- If the question asks ONE thing (even if complex), return it unchanged as a single item.
- If it asks TWO or MORE independent analytical questions joined by "and",
  "also", "as well as", commas, or multiple question marks, split them.
- Each sub-question must be a complete, standalone analytical question.
- Return a JSON array of strings. Minimum 1 item, maximum 4 items.
- Return ONLY the JSON array. No explanation.

Examples:
- "What is the average salary?" → ["What is the average salary?"]
- "Show top 5 employees and what tables exist" → ["Show top 5 employees", "What tables exist in the database?"]
- "Average salary of placed candidates and AI spending by industry" → ["What is the average salary of placed candidates?", "What is the AI spending by industry?"]
"""),
    ("human", "{query}")
])

def decompose_query(query: str) -> list[str]:
    """Returns a list of sub-queries. Single-part queries return [query]."""
    ...
```

### Pipeline Integration

```python
# In process_query.py, after contextualization, before cache check:

sub_queries = decompose_query(active_query)

if len(sub_queries) > 1:
    # Execute each sub-query through the standard pipeline independently
    # Combine all NL responses into one unified answer
    combined_result = execute_decomposed_queries(sub_queries, ...)
    return combined_result
else:
    # Single query — continue with the existing pipeline unchanged
    ...
```

---

## Strategy 2 — Query Rewriting (Schema-Aligned Synonym Expansion)

### What It Does

Rewrites user terminology to align with actual database column names
and values, improving both table retrieval accuracy and SQL generation
precision.

### Why It Helps

Even though BAAI/bge-m3 handles semantic similarity well, the SQL
generator can still struggle when user terms don't map directly to
column names:

- User says "workers" → schema has `employees`
- User says "revenue" → schema has `total_sales`
- User says "location" → schema has `country`, `region`, `city`

### Implementation

This is a lightweight LLM call that runs only when the query uses
terms that don't appear in any table/column name.

```python
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a SQL analytics chatbot.

Given the user's question and a list of available table/column names,
rewrite the question to use the exact terminology from the database schema.

Rules:
- Only change terms that have a clear synonym in the schema.
- Do NOT change the meaning or intent of the question.
- If all terms already match the schema, return the question unchanged.
- Return ONLY the rewritten question. No explanation.
"""),
    ("human", """Available schema terms: {schema_terms}

Original question: {query}

Rewritten question:""")
])
```

### When It Runs

Only when `needs_rewrite(query, schema_terms)` returns True:
- Query contains no exact column/table name matches
- Query uses common business synonyms (workers, revenue, location, etc.)

---

## Strategy 3 — Step-Back Prompting (Contextual Expansion)

### What It Does

For narrow filter questions ("What about Healthcare?", "Show me
Agriculture data"), generates a broader contextual query alongside
the specific one, giving the response generator baseline data for
comparison.

### Why It Helps

This complements the existing "Comparative & Filter Queries" rule in
`query_ai.py` (line 80). While that rule instructs the SQL generator
to include aggregates in the SQL itself, step-back prompting works
at the query level — ensuring the retriever finds the right tables
even when the user's question is too narrow.

### Implementation

```python
STEPBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an analytical assistant. Given a narrow, specific
analytical question, generate a broader "step-back" version that provides
overall context.

Rules:
- The step-back question should ask for the SAME metric but across ALL
  categories/entities, not just the specific one mentioned.
- Return ONLY the broader question. No explanation.
- If the question is already broad/general, return it unchanged.

Example:
- Specific: "What is the AI adoption rate for Healthcare?"
  Step-back: "What is the AI adoption rate across all industries?"
- Specific: "Show salary for Senior employees"
  Step-back: "What is the salary distribution across all seniority levels?"
- Broad: "What is the average salary?" (already broad)
  Step-back: "What is the average salary?" (unchanged)
"""),
    ("human", "{query}")
])
```

### When It Runs

Only when the query contains a specific entity/category filter AND the
SQL generator's comparative rule would benefit from broader retrieval context.

---

## File Structure

```
llm/
└── query_transformer.py    ← NEW: all three transformation strategies
```

Single file. Three exported functions:
- `decompose_query(query) -> list[str]`
- `rewrite_query(query, schema_terms) -> str`
- `generate_stepback_query(query) -> str | None`

One orchestrator:
- `transform_query(query) -> TransformResult`

---

## Integration Points

### `workflow/process_query.py`

The transformation layer plugs in at ONE location — after contextualization
(line ~516) and before cache check (line ~623):

```python
# After contextualization
active_query = contextualize(user_query, _user_id)

# NEW: Apply query transformations
from llm.query_transformer import transform_query
transform_result = transform_query(active_query)

if transform_result.is_decomposed:
    # Handle multi-part query
    return execute_decomposed_queries(transform_result.sub_queries, ...)
else:
    # Use the (possibly rewritten) query for the rest of the pipeline
    active_query = transform_result.rewritten_query
    # Continue with existing pipeline unchanged...
```

### What Does NOT Change

| Component | Status |
|---|---|
| `retrieval/query_router.py` | UNCHANGED — routing happens before transformation |
| `memory/mem0_manager.py` | UNCHANGED — contextualization happens before transformation |
| `retrieval/query_cache.py` | UNCHANGED — cache check uses the transformed query |
| `retrieval/table_retriever.py` | UNCHANGED — retrieval uses the transformed query |
| `llm/query_ai.py` | UNCHANGED — SQL generation uses the transformed query |
| `analysis/result_enricher.py` | UNCHANGED |
| `llm/response_generator.py` | UNCHANGED (for single queries); extended prompt for combined multi-part |
| `workflow/query_executor.py` | UNCHANGED |

---

## Execution Order Within `transform_query()`

```
Input query
    │
    ▼
1. Decompose? ──yes──► return sub-queries (skip rewrite/stepback)
    │ no
    ▼
2. Rewrite? ──yes──► rewrite query with schema terms
    │ (pass through or rewritten)
    ▼
3. Step-back? ──yes──► generate broader context query
    │ (attach as metadata, not replacing original)
    ▼
Return TransformResult(rewritten_query, stepback_query=None|str)
```

Key rule: **Decomposition is mutually exclusive with rewrite/stepback.**
If a query decomposes into sub-queries, each sub-query gets its own
independent pipeline run — rewriting and step-back happen inside each
sub-query's individual pipeline pass (if needed).

---

## Token Budget

All three prompts are lightweight (< 500 tokens each). Expected latency:
- Decomposition check: ~200ms (single short LLM call)
- Rewrite: ~150ms (only when triggered)
- Step-back: ~150ms (only when triggered)

Worst case: +350ms for a single query (decompose check + rewrite).
Best case: +0ms (decompose check returns single query, no rewrite needed → skip both).
