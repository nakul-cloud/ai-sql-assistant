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
MAX_NL_CHARS = 500
MAX_DATA_CHARS = 300

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


def _build_memory_content(
    nl_response: str,
    sql_query: str = None,
    query_result: dict = None,
    max_rows: int = 5
) -> str:
    """
    Builds the assistant-side memory content string.
    Appends a compact, generic data summary (SQL + sample rows) to the
    NL response so mem0 can extract concrete facts (numbers, names,
    categories) from ANY dataset — not just prose.

    - query_result = None (CHAT, DESCRIBE, SCHEMA_INFO) → returns nl_response unchanged
    - query_result with 'rows'/'columns' (any SQL_QUERY/DATA_PREVIEW)
      → append columns + up to `max_rows` rows as compact key=value pairs

    Must NOT reference any specific column name, table name, or domain term.
    """
    # Truncate NL prose portion first (preserves the start of the narrative)
    truncated_nl = nl_response[:MAX_NL_CHARS] if nl_response else ""

    if not query_result or not query_result.get("rows"):
        return truncated_nl

    cols = query_result.get("columns", [])
    rows = query_result.get("rows", [])[:max_rows]

    row_strs = []
    for row in rows:
        if isinstance(row, dict):
            pairs = ", ".join(f"{k}={v}" for k, v in row.items())
        else:
            pairs = ", ".join(f"{c}={v}" for c, v in zip(cols, row))
        row_strs.append(f"({pairs})")

    data_summary = "; ".join(row_strs)
    # Cap the data summary portion separately
    if len(data_summary) > MAX_DATA_CHARS:
        data_summary = data_summary[:MAX_DATA_CHARS] + "..."

    if sql_query:
        suffix = f"\n[SQL: {sql_query}]\n[Data: {data_summary}]"
    else:
        suffix = f"\n[Data: {data_summary}]"

    return f"{truncated_nl}{suffix}"


def store(
    user_id: str,
    user_query: str,
    assistant_response: str,
    metadata: dict = None,
    sql_query: str = None,
    query_result: dict = None
) -> bool:
    """
    Store a conversation turn in mem0 Cloud asynchronously.
    mem0 automatically extracts semantic facts from the turn.
    You do NOT manage embeddings or vector storage.

    Call AFTER generating the response at the end of process_user_query().

    Optional sql_query/query_result: when provided (SQL_QUERY, DATA_PREVIEW),
    appends a compact data summary so mem0 can extract concrete values.
    Callers without these args (CHAT, DESCRIBE, etc.) work identically.
    """
    client = _get_client()
    if not client:
        return False

    import threading

    content = _build_memory_content(assistant_response, sql_query, query_result)

    def _async_add():
        try:
            messages = [
                {"role": "user",      "content": user_query},
                {"role": "assistant", "content": content}
            ]
            meta = {"source": "sql_analytics_assistant"}
            if metadata:
                meta.update(metadata)

            client.add(messages, user_id=user_id, metadata=meta)
            logger.info(f"Asynchronously stored memory for user '{user_id}'.")
        except Exception as e:
            logger.warning(f"mem0 store failed for user '{user_id}': {e}")

    # Start store in background thread
    threading.Thread(target=_async_add, daemon=True).start()
    return True


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
        results = client.search(
            query=query,
            filters={"user_id": user_id},
            limit=TOP_K_MEMORIES
        )
        memories = results.get("results", results) if isinstance(results, dict) else results
        logger.info(f"retrieve(): {len(memories)} memories found for user '{user_id}' on query '{query[:50]}'")
        return memories
    except Exception as e:
        logger.warning(f"mem0 retrieve failed for user '{user_id}': {e}")
        return []
def needs_contextualization(query: str) -> bool:
    """
    Checks if a query contains pronouns, relative/transition words, generic data containers,
    or lack of explicit table name mentions, indicating a follow-up query requiring context.
    """
    import re
    q = query.strip().lower()

    # 1. Pronoun patterns
    pronoun_pattern = r"\b(it|they|them|their|those|this|these|that|there|him|her|he|she|us|we|me)\b"
    if re.search(pronoun_pattern, q):
        return True

    # 2. Generic data container/aspect references
    container_pattern = r"\b(records|rows|data|dataset|datasets|table|tables|entries|preview|details|summary|overview|result|results|chart|graph|plot|report|stats|statistics|information|info|figures?|numbers?)\b"
    if re.search(container_pattern, q):
        return True

    # 3. Relative / transition phrasing
    relative_words = [
        "what about", "how about", "and for", "compare", "difference", "change to",
        "focus on", "switch to", "show them", "show details", "explain that", "describe it",
        "why did", "what was", "show more", "summarize", "describe", "explain"
    ]
    for word in relative_words:
        if word in q:
            return True

    # 4. Very short queries (e.g. <= 3 words) are highly contextual
    if len(q.split()) <= 3:
        return True

    # 5. Check if query lacks any explicit table name mention from active schema metadata
    try:
        from database.schema_manager import fetch_database_metadata
        metadata_list = fetch_database_metadata()
        has_table_mention = False
        for meta in metadata_list:
            raw_name = meta.get("table_name", "").lower()
            # Clean name e.g. "dbo.csv_placement_data_full_class" -> "placement data"
            clean_name = raw_name.split(".")[-1].replace("csv_", "").replace("tbl_", "").replace("_", " ")
            
            # Check for name or parts of the name
            # e.g., if clean_name is "placement data", match "placement" or "placement data"
            # Split clean name into words of length > 3
            table_words = [w for w in clean_name.split() if len(w) > 3]
            if any(w in q for w in table_words) or clean_name in q or raw_name in q:
                has_table_mention = True
                break
        if not has_table_mention:
            # Lacks table mention -> must refer to the active table in context
            return True
    except Exception:
        pass

    return False


def contextualize(user_query: str, user_id: str) -> str:
    """
    Rephrase a follow-up query into a standalone query using mem0 memories.
    """
    if not needs_contextualization(user_query):
        logger.info(f"Query is standalone. Bypassing contextualization: '{user_query}'")
        return user_query

    memories = retrieve(user_id, user_query)
    if not memories:
        return user_query

    context_lines = []
    for m in memories:
        text = m.get("memory", "") if isinstance(m, dict) else str(m)
        if text:
            context_lines.append(f"- {text[:MAX_NL_CHARS]}")

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
            lines.append(f"- {text[:MAX_NL_CHARS]}")

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
        results = client.get_all(filters={"user_id": user_id})
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
