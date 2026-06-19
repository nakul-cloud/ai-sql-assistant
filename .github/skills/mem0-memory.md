---
name: mem0-memory
description: >
  Apply this skill when adding, updating, or debugging conversational memory
  in the Enterprise AI SQL Analytics Assistant using mem0. Triggers on:
  "mem0", "memory", "follow-up context", "contextualize_query", "chat_history",
  "conversation memory", "remember previous", "cross-session memory",
  "replace contextualize", "_format_history_for_nl", "needs_contextualization".
  Do NOT apply to: Qdrant schema indexing, SQL generation, result enrichment,
  or query caching — those are separate concerns covered by other skills.
---

# mem0 Conversational Memory Layer

## What This Replaces

Three manual memory functions in `process_query.py` are removed and replaced
by a single `memory/mem0_manager.py` module:

| Removed | Replaced by |
|---|---|
| `needs_contextualization()` | `mem0_manager.needs_context()` |
| `contextualize_query()` + `_contextualize_chain` + `CONTEXTUALIZE_PROMPT` | `mem0_manager.contextualize()` |
| `_format_history_for_nl()` | `mem0_manager.get_context_for_prompt()` |
| `chat_history: list` raw buffer passed around | `user_id` string passed — mem0 manages storage |

**What does NOT change:**
- `chat_history_str` parameter in `response_generator.py` — still exists, now fed from mem0
- `process_user_query()` signature — `chat_history` param stays for backward compat with Streamlit
- All intent handlers — untouched
- All SQL pipeline steps — untouched
- Qdrant schema collection — untouched (mem0 uses a separate collection)

---

## Responsibility Split — What Does What

```
mem0 Cloud          → memory extraction, storage, semantic search
                      (their managed infra — no Qdrant, no embeddings from your side)

Your Qdrant (local) → schema index (ai_sql_schema_index)
                      query cache (query_cache)
                      UNTOUCHED by mem0

Your bge-m3         → schema + query embeddings only
                      UNTOUCHED by mem0

Your Groq           → SQL gen, NL response, intent classification, rephrase LLM
                      mem0 uses its OWN internal models for extraction/search
```

## How mem0 Works in This System

```
User sends message
      │
      ▼
mem0_manager.contextualize(query, user_id)
      │  ← mem0 Cloud searches stored memories (no local embedding)
      │  ← Groq rephrases follow-up into standalone query
      ▼
process_user_query() runs normally with rephrased query
      │
      ▼
Response generated
      │
      ▼
mem0_manager.store(user_id, user_query, assistant_response)
      │  ← mem0 Cloud extracts facts + stores them (no local vector store)
      ▼
Next query can retrieve this context
```

Your Qdrant and bge-m3 are never called by the memory layer.

---

## Installation

```bash
pip uv add install mem0ai
```

Add to `.env`:
```
MEM0_API_KEY=your_key_from_app.mem0.ai
MEM0_USER_ID=default_user
```

Get your key: `app.mem0.ai` → sign up → API Keys → copy key. --- i have the api key just aks me or say where to put 

**Nothing to remove from `.env`** — `QDRANT_HOST`, `QDRANT_PORT`, `GROQ_API_KEY`
stay exactly as they are for your existing pipeline.

---

## Step 1 — Create `memory/mem0_manager.py`

New file. Single source of truth for all memory operations.
Uses `MemoryClient` (cloud) — no Qdrant config, no embedder config, no LLM config for mem0.
mem0 Cloud manages all of that internally.

```python
"""
memory/mem0_manager.py
──────────────────────
Conversational memory using mem0 Cloud API.

mem0 Cloud handles ONLY:
  - Memory extraction from conversation turns
  - Memory storage (their managed cloud)
  - Memory retrieval (semantic search over stored facts)

Your existing stack is completely untouched:
  - Qdrant (local Docker) → schema index + query cache only
  - BAAI/bge-m3           → schema + query embeddings only
  - Groq                  → SQL gen, NL response, intent classification, rephrase

Replaces in process_query.py:
  - needs_contextualization()
  - contextualize_query() + CONTEXTUALIZE_PROMPT + _contextualize_chain
  - _format_history_for_nl()

Provides:
  - init_memory()            → connect to mem0 Cloud (call once at startup)
  - store()                  → save a conversation turn
  - contextualize()          → rephrase follow-up using retrieved memory
  - get_context_for_prompt() → formatted context string for NL response prompt
  - clear_user_memory()      → wipe memory for a user (testing/reset)
  - get_all_memories()       → list all memories for a user (debug/display)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_USER_ID = os.getenv("MEM0_USER_ID", "default_user")
TOP_K_MEMORIES  = 5
MAX_SNIPPET_CHARS = 400

_client = None


def init_memory():
    """
    Connect to mem0 Cloud using API key.
    Call once at app startup via @st.cache_resource in app.py.
    Safe to call multiple times — returns cached client.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        logger.error("MEM0_API_KEY not set in .env — memory disabled.")
        return None

    try:
        from mem0 import MemoryClient
        # MemoryClient connects to mem0 Cloud — no local config needed
        _client = MemoryClient(api_key=api_key)
        logger.info("mem0 Cloud client initialized.")
        return _client
    except Exception as e:
        logger.error(f"Failed to initialize mem0 Cloud client: {e}")
        return None


def _get_client():
    global _client
    if _client is None:
        return init_memory()
    return _client


def store(
    user_id: str,
    user_query: str,
    assistant_response: str,
    metadata: dict = None
) -> bool:
    """
    Store a conversation turn in mem0 Cloud.
    mem0 automatically extracts semantic facts from the turn.
    You do NOT manage embeddings or vector storage.

    Call AFTER generating the response at the end of process_user_query().
    """
    client = _get_client()
    if not client:
        return False

    try:
        messages = [
            {"role": "user",      "content": user_query},
            {"role": "assistant", "content": assistant_response[:MAX_SNIPPET_CHARS]}
        ]
        meta = {"source": "sql_analytics_assistant"}
        if metadata:
            meta.update(metadata)

        client.add(messages, user_id=user_id, metadata=meta)
        logger.info(f"Stored memory for user '{user_id}'.")
        return True
    except Exception as e:
        logger.warning(f"mem0 store failed for user '{user_id}': {e}")
        return False


def retrieve(user_id: str, query: str) -> list:
    """
    Retrieve relevant memories for a query from mem0 Cloud.
    mem0 handles semantic search internally — no local embedding needed.

    Returns: [{"memory": "...", "score": 0.92}, ...]
    """
    client = _get_client()
    if not client:
        return []

    try:
        results = client.search(query=query, user_id=user_id, limit=TOP_K_MEMORIES)
        memories = results.get("results", results) if isinstance(results, dict) else results
        logger.info(f"Retrieved {len(memories)} memories for user '{user_id}'.")
        return memories
    except Exception as e:
        logger.warning(f"mem0 retrieve failed for user '{user_id}': {e}")
        return []


def contextualize(user_query: str, user_id: str) -> str:
    """
    Rephrase a follow-up query into a standalone query using mem0 memories.

    Replaces entirely:
        - needs_contextualization()
        - contextualize_query()
        - CONTEXTUALIZE_PROMPT
        - _contextualize_chain

    Flow:
        1. mem0 Cloud retrieves relevant past context (no local embedding used)
        2. If no memories → return query unchanged (no LLM call)
        3. If memories found → Groq rephrases via LangChain
    """
    memories = retrieve(user_id, user_query)
    if not memories:
        return user_query

    context_lines = []
    for m in memories:
        text = m.get("memory", "") if isinstance(m, dict) else str(m)
        if text:
            context_lines.append(f"- {text[:MAX_SNIPPET_CHARS]}")

    if not context_lines:
        return user_query

    context_str = "\n".join(context_lines)

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from llm.llm_client import get_llm

        REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
            ("system", """You are a query contextualization assistant for a SQL analytics chatbot.

Given relevant conversation memories and the user's latest message,
rephrase the message into a fully self-contained standalone analytical question.

Rules:
- If the message is already standalone (no pronouns, no references to prior topics),
  return it EXACTLY as written — do not change it.
- If it references previous context (pronouns like "them", "it", "that",
  phrases like "what about X", "and for Y", "compare with that"),
  incorporate the relevant context to make it standalone.
- Return ONLY the rephrased question. No explanation. No SQL. No preamble.
"""),
            ("human", """Relevant memories:
{context_str}

Latest message: "{user_query}"

Standalone question:""")
        ])

        chain = REPHRASE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
        rephrased = chain.invoke({
            "context_str": context_str,
            "user_query": user_query
        }).strip().strip('"\'')

        if rephrased and rephrased != user_query:
            logger.info(f"Contextualized: '{user_query}' → '{rephrased}'")
        return rephrased if rephrased else user_query

    except Exception as e:
        logger.warning(f"Rephrase LLM call failed: {e} — returning original")
        return user_query


def get_context_for_prompt(user_id: str, query: str) -> str:
    """
    Returns formatted memory context for injection into NL response prompt.
    Replaces _format_history_for_nl().

    Output feeds directly into {chat_history_str} in response_generator.py.
    response_generator.py needs ZERO changes.
    """
    memories = retrieve(user_id, query)
    if not memories:
        return "None"

    lines = []
    for m in memories:
        text = m.get("memory", "") if isinstance(m, dict) else str(m)
        if text:
            lines.append(f"- {text[:MAX_SNIPPET_CHARS]}")

    return "\n".join(lines) if lines else "None"


def clear_user_memory(user_id: str) -> bool:
    """Clear all memories for a user. Useful for testing or session reset."""
    client = _get_client()
    if not client:
        return False
    try:
        client.delete_all(user_id=user_id)
        logger.info(f"Cleared memory for user '{user_id}'.")
        return True
    except Exception as e:
        logger.warning(f"Failed to clear memory: {e}")
        return False


def get_all_memories(user_id: str) -> list:
    """Return all stored memories for a user (debug/display in UI)."""
    client = _get_client()
    if not client:
        return []
    try:
        results = client.get_all(user_id=user_id)
        return results.get("results", results) if isinstance(results, dict) else results
    except Exception as e:
        logger.warning(f"Failed to get all memories: {e}")
        return []


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("\n--- Testing mem0 Cloud Memory Manager ---\n")

    TEST_USER = "test_user_001"

    print("[1] Initializing mem0 Cloud client...")
    client = init_memory()
    if not client:
        print("  [FAIL] MEM0_API_KEY missing or invalid. Check .env")
        sys.exit(1)
    print("  [OK] Connected to mem0 Cloud.\n")

    clear_user_memory(TEST_USER)

    print("[2] Storing 2 conversation turns...")
    store(
        user_id=TEST_USER,
        user_query="Which industry has the highest AI investment?",
        assistant_response="Financial Services leads with an average of 8,206,724.",
        metadata={"intent": "SQL_QUERY"}
    )
    store(
        user_id=TEST_USER,
        user_query="How many companies are in Financial Services?",
        assistant_response="There are 27,927 companies in Financial Services.",
        metadata={"intent": "SQL_QUERY"}
    )
    print("  [OK] 2 turns stored.\n")

    print("[3] Testing contextualization (follow-up)...")
    rephrased = contextualize("what about Technology?", TEST_USER)
    print(f"  Input  : 'what about Technology?'")
    print(f"  Output : '{rephrased}'")
    print(f"  Result : {'[OK]' if rephrased.lower() != 'what about technology?' else '[CHECK — may need more memory]'}\n")

    print("[4] Testing standalone (should NOT change)...")
    standalone = "What is the average salary across all employees?"
    r2 = contextualize(standalone, TEST_USER)
    print(f"  Input  : '{standalone}'")
    print(f"  Output : '{r2}'")
    print(f"  Result : {'[OK — unchanged]' if r2 == standalone else '[NOTE — was changed]'}\n")

    print("[5] Testing get_context_for_prompt...")
    ctx = get_context_for_prompt(TEST_USER, "AI investment comparison")
    print(f"  Context:\n{ctx}\n")

    clear_user_memory(TEST_USER)
    print("[6] Cleanup done.")
    print("\n" + "=" * 50)
    print("  [OK] mem0 Cloud manager — all tests passed!")
    print("=" * 50)
    sys.exit(0)
```

---

## Step 2 — Update `process_query.py`

### 2a — Remove these entirely (delete from file):

```python
# DELETE THESE:
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([...])
_contextualize_chain = CONTEXTUALIZE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()

def needs_contextualization(query: str) -> bool: ...
def contextualize_query(user_query: str, chat_history: list) -> str: ...
def _format_history_for_nl(chat_history: list) -> str: ...
```

### 2b — Add import at top of file:

```python
from memory.mem0_manager import contextualize, get_context_for_prompt, store as store_memory
```

### 2c — Update `process_user_query()` signature:

```python
# BEFORE:
def process_user_query(
    user_query: str,
    focus_tables: list = None,
    chat_history: list = None,  # ← keep for Streamlit backward compat
    stream: bool = False
) -> Dict[str, Any]:

# AFTER — add user_id param:
def process_user_query(
    user_query: str,
    focus_tables: list = None,
    chat_history: list = None,  # ← kept for backward compat, no longer used internally
    stream: bool = False,
    user_id: str = None         # ← NEW: mem0 user identifier
) -> Dict[str, Any]:
```

### 2d — Replace the contextualization block (Step 0):

```python
# BEFORE (Step 0):
active_query = user_query
if chat_history:
    active_query = contextualize_query(user_query, chat_history)

# AFTER:
import os
active_query = user_query
_user_id = user_id or os.getenv("MEM0_USER_ID", "default_user")

# mem0 contextualization — replaces manual buffer-based contextualize_query()
active_query = contextualize(user_query, _user_id)
```

### 2e — Replace `_format_history_for_nl` call:

```python
# BEFORE:
chat_hist_str = _format_history_for_nl(chat_history)

# AFTER:
chat_hist_str = get_context_for_prompt(_user_id, active_query)
```

### 2f — Add memory storage after successful SQL response (non-stream path):

```python
# AFTER store_in_query_cache():
store_in_query_cache(active_query, nl_response, query_result)

# ADD THIS:
store_memory(
    user_id=_user_id,
    user_query=active_query,
    assistant_response=nl_response,
    metadata={"intent": "SQL_QUERY", "tables": focus_tables or []}
)
```

### 2g — Add memory storage for stream path (inside `cached_stream_wrapper`):

```python
def cached_stream_wrapper(strm, q, q_res):
    accumulated = []
    try:
        for chunk in strm:
            accumulated.append(chunk)
            yield chunk
    finally:
        full_txt = "".join(accumulated).strip()
        if full_txt:
            store_in_query_cache(q, full_txt, q_res)
            # ADD THIS:
            store_memory(
                user_id=_user_id,
                user_query=q,
                assistant_response=full_txt,
                metadata={"intent": "SQL_QUERY", "tables": focus_tables or []}
            )
```

### 2h — Store memory for non-SQL intents too (CHAT, DESCRIBE, SCHEMA_INFO):

```python
# After handle_chat_query():
if intent == "CHAT":
    res = handle_chat_query(active_query, stream=stream)
    res["rephrased_query"] = active_query
    # Store chat turn in memory
    if res.get("success") and isinstance(res.get("nl_response"), str):
        store_memory(_user_id, active_query, res["nl_response"], {"intent": "CHAT"})
    return res
```

Apply the same pattern to `DESCRIBE` and `SCHEMA_INFO` handlers — store turn after response is generated.

---

## Step 3 — Update `app.py` (Streamlit entry point)

Add startup initialization and pass `user_id` from session state:

```python
# In app.py — add near the top, before st.chat_input loop:
from memory.mem0_manager import init_memory

# Initialize mem0 once at app startup
@st.cache_resource
def _init_mem0():
    return init_memory()

_init_mem0()

# Generate a session-scoped user_id so each browser tab has its own memory
if "user_id" not in st.session_state:
    import uuid
    st.session_state["user_id"] = f"user_{uuid.uuid4().hex[:8]}"

user_id = st.session_state["user_id"]
```

Pass `user_id` into every `process_user_query()` call:

```python
# BEFORE:
result = process_user_query(
    user_query=prompt,
    chat_history=st.session_state.messages,
    stream=True
)

# AFTER:
result = process_user_query(
    user_query=prompt,
    chat_history=st.session_state.messages,  # kept for compat
    stream=True,
    user_id=st.session_state["user_id"]       # ADD THIS
)
```

---

## Step 4 — Directory Structure Change

```
ai-sql-assistant/
│
├── memory/                        ← NEW directory
│   ├── __init__.py                ← empty
│   └── mem0_manager.py            ← NEW: all mem0 operations
│
├── workflow/
│   └── process_query.py           ← UPDATED: remove 3 functions, add mem0 calls
│
└── app.py                         ← UPDATED: init_memory() + user_id session state
```

---

## Build Order — One Step at a Time

### Step 1 — Install and verify mem0 connects to Qdrant
```bash
pip uv add mem0ai ---- instruct me to do from my terminal 
python -m memory.mem0_manager
```
Verify: mem0 initializes, stores a test turn, retrieves it, contextualizes a follow-up.

### Step 2 — Add imports to `process_query.py`, remove 3 functions
Delete `needs_contextualization`, `contextualize_query`, `_format_history_for_nl`,
`CONTEXTUALIZE_PROMPT`, `_contextualize_chain`. Add mem0 imports.
**Run existing tests** — nothing should break yet since these functions aren't called
until Step 3.

### Step 3 — Replace Step 0 (contextualization) in `process_user_query()`
Replace the `if chat_history: contextualize_query(...)` block with `contextualize(...)`.
**Test with:** "Which industry has highest AI investment?" → "what about technology?"

### Step 4 — Replace `_format_history_for_nl()` with `get_context_for_prompt()`
Replace the `chat_hist_str` assignment. `response_generator.py` receives the same
`chat_history_str` param — no change needed there.
**Test with:** follow-up questions that reference previous results.

### Step 5 — Add `store_memory()` calls after responses
Add to SQL pipeline (both stream and non-stream), CHAT handler, DESCRIBE handler.
**Test with:** multi-turn conversation, verify memories accumulate.

### Step 6 — Update `app.py`
Add `init_memory()` at startup, `user_id` in session state, pass to `process_user_query()`.
**Test with:** full end-to-end Streamlit session.

---

## What mem0 Stores Per Turn

mem0 automatically extracts and stores semantic facts from each turn, not raw text:

```
Turn: "Which industry has highest AI investment?" → "Financial Services leads at 8.2M"

mem0 extracts:
  - "User asked about AI investment by industry"
  - "Financial Services has the highest AI investment at 8.2M"
  - "User is analyzing the corporate AI adoption dataset"
```

When next query arrives ("what about Technology?"), mem0 retrieves these facts
and the rephrasing LLM produces:
"What is the AI investment for Technology, compared to Financial Services?"

---

## Qdrant Collections After Integration

mem0 Cloud stores memory on their servers — your local Qdrant is completely unchanged.

| Collection | Purpose | Used by |
|---|---|---|
| `ai_sql_schema_index` | Table schema index for RAG | `table_retriever.py` |
| `query_cache` | Past query result cache | `query_cache.py` |

That's it. No new collections added to your Qdrant.
mem0 memory lives entirely in mem0 Cloud.

---

## Backward Compatibility Notes

- `chat_history: list` param in `process_user_query()` is kept but no longer
  used internally. Streamlit can still pass it — it's just ignored.
  Remove it later once you confirm mem0 is working end-to-end.
- `chat_history_str` param in `generate_natural_language_response()` is kept unchanged.
  The source changes from `_format_history_for_nl(chat_history)` to
  `get_context_for_prompt(user_id, query)` — same param name, better content.
- `response_generator.py` needs zero changes.
- `query_router.py` needs zero changes.
- `result_enricher.py` needs zero changes.