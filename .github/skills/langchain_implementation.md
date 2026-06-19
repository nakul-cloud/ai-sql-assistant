---
name: langchain-integration
description: >
  Apply this skill when refactoring or building any LLM-facing component of the
  Enterprise AI SQL Analytics Assistant using LangChain. Triggers on: "use langchain",
  "refactor to langchain", "LLMChain", "LCEL", "RunnableSequence", "ChatPromptTemplate",
  "ChatGoogleGenerativeAI", "langchain prompt", "langchain chain", or any mention of
  replacing manual Gemini API calls with LangChain. Do NOT apply to: BAAI/bge-m3
  embedding, Qdrant upsert/retrieval, result_enricher.py, query_cache.py,
  regex pre-check, or SQL execution — those stay as plain Python.
---

# LangChain Integration Layer

## What LangChain Replaces (and What It Does NOT)

### ✅ Replace with LangChain

| File | Current | LangChain replacement |
|---|---|---|
| `llm/query_ai.py` | Manual Gemini REST call + prompt string | `ChatPromptTemplate` + `ChatGoogleGenerativeAI` + LCEL chain |
| `llm/response_generator.py` | Manual Gemini REST call + prompt string | `ChatPromptTemplate` + LCEL chain |
| `retrieval/query_router.py` (LLM branch only) | Manual Gemini call for intent classification | `ChatPromptTemplate` + structured output chain |
| `workflow/process_query.py` | Manual if/else orchestration | `RunnableSequence` (LCEL) or keep manual — see note below |

### ❌ Do NOT replace with LangChain

| Component | Why it stays as plain Python |
|---|---|
| `indexing/embedder.py` (BAAI/bge-m3) | LangChain has no hybrid dense+sparse wrapper for bge-m3. FlagEmbedding stays. |
| `retrieval/query_cache.py` | Custom Qdrant cosine cache — no LangChain abstraction fits |
| `retrieval/table_retriever.py` | Custom RRF fusion — `QdrantVectorStore` doesn't support named vector RRF |
| `analysis/result_enricher.py` | Pure Pandas math — no LangChain needed |
| `retrieval/query_router.py` (regex branch) | Plain `re` module — 0ms, no LangChain needed |
| `workflow/query_executor.py` | SQLAlchemy execution — stays unchanged |

---

## Setup

```python
# requirements additions
# langchain
# langchain-google-genai
# langchain-core

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```

Initialize the LLM once, import everywhere:

```python
# llm/llm_client.py  ← new shared file
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=True  # required for Gemini
    )
```

---

## File 1 — `llm/query_ai.py` (SQL Generator)

Replaces manual Gemini call with a LangChain LCEL chain.

```python
# llm/query_ai.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert SQL Server query generator.
Generate a valid T-SQL SELECT query based on the user's question and schema context.

Rules:
- Only generate SELECT statements. Never INSERT, UPDATE, DELETE, DROP.
- Use table and column names exactly as provided in the schema.
- If the question cannot be answered from the schema, say: CANNOT_GENERATE
- Return ONLY the SQL query, no explanation, no markdown fences.
"""),
    ("human", """Schema context:
{schema_context}

User question: {user_query}

SQL query:""")
])

def build_sql_chain():
    llm = get_llm(temperature=0.0)
    return SQL_PROMPT | llm | StrOutputParser()

# Module-level chain — instantiated once
_sql_chain = build_sql_chain()

def generate_sql_query(user_query: str, schema_context: str) -> dict:
    sql = _sql_chain.invoke({
        "user_query": user_query,
        "schema_context": schema_context
    })
    sql = sql.strip()
    return {
        "sql_query": sql,
        "success": sql != "CANNOT_GENERATE"
    }


if __name__ == "__main__":
    test_schema = "Table: tickets (id INT PK, status VARCHAR, created_date DATE, priority INT)"
    result = generate_sql_query("Show all open tickets", test_schema)
    print(result)
```

---

## File 2 — `llm/response_generator.py` (NL Response)

```python
# llm/response_generator.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

NL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a business data analyst assistant.
Answer the user's question using the pre-analyzed data context provided.
The summary and alerts have already been calculated — use them directly.

Rules:
- Write 2-4 clear, concise sentences.
- Mention specific numbers from the data.
- Reference alerts if present.
- Do not mention SQL, databases, tables, or technical terms.
- Do not speculate beyond what the data shows.
"""),
    ("human", """User question: {user_query}

SQL used: {sql_query}

Pre-analyzed context:
- Total records: {total_rows}
- Key statistics: {numeric_summary}
- Alerts: {alerts}
- Category breakdown: {category_breakdown}

Data sample (first 5 rows):
{data_sample}

Answer:""")
])

def build_nl_chain():
    llm = get_llm(temperature=0.3)
    return NL_PROMPT | llm | StrOutputParser()

_nl_chain = build_nl_chain()

def generate_natural_language_response(
    user_query: str,
    sql_query: str,
    enriched_result: dict
) -> dict:
    """
    enriched_result is the output of result_enricher.enrich_query_result()
    Contains: rows, columns, summary (total_rows, numeric_summary, etc.), alerts
    """
    summary = enriched_result.get("summary", {})

    response_text = _nl_chain.invoke({
        "user_query": user_query,
        "sql_query": sql_query,
        "total_rows": summary.get("total_rows", len(enriched_result.get("rows", []))),
        "numeric_summary": summary.get("numeric_summary", "N/A"),
        "alerts": enriched_result.get("alerts", []),
        "category_breakdown": summary.get("category_breakdown", "N/A"),
        "data_sample": enriched_result.get("rows", [])[:5]
    })

    return {
        "response_text": response_text.strip(),
        "success": True
    }


if __name__ == "__main__":
    mock_enriched = {
        "rows": [{"ticket_id": 1, "status": "open", "priority": 1}],
        "summary": {
            "total_rows": 1,
            "numeric_summary": {"priority": {"min": 1, "max": 1, "mean": 1.0}},
            "category_breakdown": {"status": {"open": 1}}
        },
        "alerts": ["🔴 High-priority records present"]
    }
    result = generate_natural_language_response(
        "Show open tickets", "SELECT * FROM tickets WHERE status='open'", mock_enriched
    )
    print(result)
```

---

## File 3 — `retrieval/query_router.py` (Intent Classifier — LLM branch only)

The regex branch stays as plain Python. Only the LLM fallback uses LangChain.

```python
# retrieval/query_router.py
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

# ── Regex patterns (unchanged, plain Python) ─────────────────────────────────

CHAT_PATTERNS = [
    r"^(hi|hello|hey|howdy|good\s(morning|evening|afternoon))[\s!.?]*$",
    r"^(thanks?|thank you|thx|ok|okay|got it|cool|great|nice)[\s!.?]*$",
    r"^(bye|goodbye|see you|exit|quit)[\s!.?]*$",
]

SQL_PATTERNS = [
    r"\b(show|list|find|get|fetch|count|sum|average|how many|what is)\b.*(table|record|row|ticket|employee|order|customer|invoice)",
    r"\b(top|highest|lowest|most|least)\b.*\b(by|in|from)\b",
]

def pre_check_intent(user_query: str) -> str | None:
    q = user_query.strip().lower()
    for pattern in CHAT_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            return "CHAT"
    for pattern in SQL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "SQL_QUERY"
    return None

# ── LangChain intent classifier (LLM fallback only) ──────────────────────────

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's message into exactly one of these three intents:
- SQL_QUERY: The user wants data, reports, counts, lists, or analysis from the database
- SCHEMA_QUERY: The user is asking about what tables or columns exist
- CHAT: General conversation, greetings, or questions unrelated to data

Reply with ONLY one word: SQL_QUERY, SCHEMA_QUERY, or CHAT. No explanation."""),
    ("human", "{user_query}")
])

def build_intent_chain():
    llm = get_llm(temperature=0.0)
    return INTENT_PROMPT | llm | StrOutputParser()

_intent_chain = build_intent_chain()

def llm_classify_intent(user_query: str) -> str:
    result = _intent_chain.invoke({"user_query": user_query})
    result = result.strip().upper()
    # Sanitize — only accept known values
    if result not in {"SQL_QUERY", "SCHEMA_QUERY", "CHAT"}:
        return "SQL_QUERY"  # safe default
    return result

# ── Full router (unchanged logic) ────────────────────────────────────────────

def route_query(user_query: str) -> str:
    intent = pre_check_intent(user_query)
    if intent:
        return intent           # 0ms regex path
    return llm_classify_intent(user_query)  # LangChain fallback


if __name__ == "__main__":
    tests = [
        "hi there",
        "show me all open tickets",
        "what tables do you have?",
        "can you help me understand the data structure?",
    ]
    for t in tests:
        print(f"'{t}' → {route_query(t)}")
```

---

## File 4 — `llm/llm_client.py` (Shared LLM Instance)

New file. Prevents multiple LLM instantiations across modules.

```python
# llm/llm_client.py
import os
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=2)
def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Returns a cached LLM instance.
    temperature=0.0 for SQL generation (deterministic)
    temperature=0.3 for NL responses (slight variation is fine)
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=True
    )


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Say hello in one word.")
    print(response.content)
```

---

## Updated `workflow/process_query.py`

The orchestration logic stays as plain Python — no need to force LCEL here since
the pipeline has conditional branches (cache hits, regex exits) that read more clearly
as if/else than as a chain.

```python
# workflow/process_query.py
from retrieval.query_router import route_query
from retrieval.query_cache import check_query_cache, store_in_query_cache
from retrieval.table_retriever import retrieve_relevant_tables
from analysis.schema_context import generate_schema_context
from analysis.result_enricher import enrich_query_result
from llm.query_ai import generate_sql_query
from llm.response_generator import generate_natural_language_response
from workflow.query_executor import execute_sql_query

def process_user_query(user_query: str) -> dict:

    # Step 1: Regex pre-check (0ms)
    intent = route_query(user_query)
    if intent == "CHAT":
        return {"success": True, "cache_hit": False,
                "nl_response": "Hi! Ask me anything about your data."}

    # Step 2: Query cache check
    cached = check_query_cache(user_query)
    if cached:
        return {"success": True, "cache_hit": True, **cached}

    # Step 3: Hybrid Qdrant retrieval
    focus_tables = retrieve_relevant_tables(user_query)

    # Step 4: Schema context
    schema_context = generate_schema_context(user_query, focus_tables=focus_tables)

    # Step 5: SQL generation (LangChain)
    sql_result = generate_sql_query(user_query, schema_context)
    if not sql_result["success"]:
        return {"success": False,
                "nl_response": "I couldn't find relevant data to answer that question."}

    # Step 6: Execute SQL
    exec_result = execute_sql_query(sql_result["sql_query"])

    # Step 7: Enrich result (deterministic — Pandas)
    enriched = enrich_query_result(
        rows=exec_result["result"]["rows"],
        columns=exec_result["result"]["columns"],
        user_query=user_query,
        sql_query=sql_result["sql_query"]
    )

    # Step 8: NL response (LangChain)
    nl_response = generate_natural_language_response(
        user_query, sql_result["sql_query"], enriched
    )

    # Step 9: Store in query cache
    store_in_query_cache(user_query, nl_response["response_text"], exec_result["result"])

    return {
        "success": True,
        "cache_hit": False,
        "generated_sql": sql_result["sql_query"],
        "query_result": exec_result["result"],
        "nl_response": nl_response["response_text"]
    }
```

---

## Directory Changes (additions only)

```
ai-sql-assistant/
│
├── llm/
│   ├── llm_client.py          ← NEW: shared LangChain LLM instance
│   ├── query_ai.py            ← REFACTORED: ChatPromptTemplate + LCEL
│   └── response_generator.py  ← REFACTORED: ChatPromptTemplate + LCEL
│
├── retrieval/
│   └── query_router.py        ← REFACTORED: LLM branch now uses LangChain
│
└── workflow/
    └── process_query.py       ← UPDATED: imports cleaned up, enricher added
```

Everything else is unchanged.

---

## Build Order for This Refactor

Do these one file at a time. Each has a standalone `__main__` test.

1. `llm/llm_client.py` — verify Gemini API key works via LangChain
2. `llm/query_ai.py` — test SQL generation with a mock schema
3. `llm/response_generator.py` — test NL output with mock enriched data
4. `retrieval/query_router.py` — test all 4 intent cases
5. `workflow/process_query.py` — wire everything together

---

## New Dependencies

```
# Add to requirements.txt
langchain
langchain-google-genai
langchain-core
```

---

## Key Decisions

**Why not use LangChain's QdrantVectorStore?**
Your Qdrant setup uses named vectors (`"dense"` + `"sparse"`) with RRF fusion via
`Prefetch` + `FusionQuery`. LangChain's `QdrantVectorStore` only supports single-vector
search and cannot express this pattern. Replacing it would silently downgrade retrieval
quality. Keep `table_retriever.py` as plain `qdrant-client` code.

**Why not use LangChain's embeddings for bge-m3?**
`langchain-community` has a `HuggingFaceBgeEmbeddings` wrapper but it only returns
dense vectors. Your pipeline requires both dense AND sparse from the same model call.
`FlagEmbedding` stays.

**Why keep `process_query.py` as plain if/else instead of LCEL?**
LCEL `RunnableSequence` works well for linear chains. Your pipeline has three early-exit
branches (CHAT, cache hit, SQL generation failure). Expressing conditional branches in
LCEL requires `RunnableBranch` which adds complexity with no readability benefit here.
Plain Python is cleaner and equally correct.

**Why `lru_cache` on `get_llm()`?**
Prevents re-instantiating the LLM client on every query. LangChain's
`ChatGoogleGenerativeAI` constructor validates the API key on init — caching it means
this happens once at startup, not per request.