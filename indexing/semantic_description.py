"""
indexing/semantic_description.py
────────────────────────────────
Generates LLM-powered business descriptions for database tables,
with a local JSON cache to avoid redundant API calls.

How caching works:
  - Each table's structure (name + columns + types) is hashed (SHA-256).
  - If the hash exists in the cache, the stored description is reused.
  - If the hash is missing (new table or schema changed), a fresh
    LLM API call is made and the result is cached.

Provides:
  • get_semantic_description(table_meta)  → cached or fresh description
  • clear_cache()                         → wipe the cache file
  • get_cache_stats()                     → cache hit/miss info

Usage:
    python -m indexing.semantic_description
"""

import hashlib
import json
import os
import sys

from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
import logging

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────
CACHE_FILE = os.getenv("DESCRIPTION_CACHE_FILE", "indexing/description_cache.json")


def _is_retryable(exception) -> bool:
    """
    Only retry on 503 (server overload/high demand).
    Do NOT retry on:
      - 429 RESOURCE_EXHAUSTED: quota is gone, retrying won't help
      - 401 UNAUTHENTICATED: bad API key, retrying won't help
    """
    err_str = str(exception)
    # Immediately stop for quota and auth errors
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        return False
    if "401" in err_str or "UNAUTHENTICATED" in err_str:
        return False
    return True


class _LLMResponse:
    """Thin response wrapper so callers can do `response.text` unchanged."""
    def __init__(self, text: str):
        self.text = text


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_retryable),
    reraise=True
)
def generate_content_with_retry(client, model, contents):
    """
    LLM call with retry logic.
    Routes to Groq using llama-3.1-8b-instant.
    The `client` and `model` parameters are kept for backward compatibility
    but are ignored — llm_client reads Groq config from .env directly.
    """
    from llm.llm_client import generate_text
    return _LLMResponse(generate_text(contents))


def _compute_table_hash(table_meta: dict) -> str:
    """
    Compute a SHA-256 hash of a table's structural signature.
    If the schema changes (column added/removed/renamed/retyped),
    the hash changes and a fresh LLM call is triggered.
    """
    signature = f"{table_meta['table_name']}:" + ",".join(
        f"{c['name']}:{c['type']}" for c in table_meta["columns"]
    )
    return hashlib.sha256(signature.encode()).hexdigest()


def _load_cache() -> dict:
    """Load the description cache from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the description cache to disk."""
    os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _generate_description_via_llm(table_meta: dict) -> dict:
    """
    Call LLM to generate a table description and column descriptions in a JSON structure.
    """
    column_list = ", ".join(f"{c['name']} ({c['type']})" for c in table_meta["columns"])
    sample_info = ""
    if table_meta.get("sample_values"):
        sample_lines = []
        for col, vals in list(table_meta["sample_values"].items())[:5]:
            sample_lines.append(f"  {col}: {vals}")
        sample_info = "\nSample values:\n" + "\n".join(sample_lines)

    prompt = f"""You are a database documentation expert.
Analyze the following database table structure and sample values, then output a JSON object describing both the table and its columns.

Table: {table_meta['table_name']}
Row Count: {table_meta.get('row_count', 'unknown')}
Columns and Types: {column_list}
{sample_info}

You must return a valid JSON object matching this schema (with no extra keys, markdown block tags, or surrounding text):
{{
  "table_description": "A 3-sentence natural language description of this database table. Include what business questions it can answer and what kind of data it holds.",
  "columns": {{
    "column_name_1": "A brief, 1-sentence description of the business meaning of this column. Identify if it represents a salary/compensation metric, a job role name/title, a seniority/experience level, a temporal/date value, or a category.",
    "column_name_2": "..."
  }}
}}

JSON output:"""

    from llm.llm_client import generate_text
    response = generate_text(prompt)
    
    # Try to parse response as JSON
    try:
        clean = response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        
        parsed = json.loads(clean)
        if "table_description" in parsed and "columns" in parsed:
            return parsed
    except Exception as e:
        print(f"[WARNING] Failed to parse LLM JSON response for table description: {e}")
        
    return {
        "table_description": response.strip(),
        "columns": {}
    }


def get_semantic_description(table_meta: dict) -> str:
    """
    Return a semantic business description for a table.
    Uses cached version if available; calls LLM if not.

    Args:
        table_meta: Dict with keys: table_name, columns, row_count, sample_values

    Returns:
        A 3-sentence business description string.
    """
    cache = _load_cache()
    table_hash = _compute_table_hash(table_meta)

    # Cache HIT
    if table_hash in cache:
        cached_val = cache[table_hash]
        if isinstance(cached_val, dict):
            return cached_val.get("table_description", "")
        return cached_val  # backward compatibility with old string cache entries

    # Cache MISS
    meta_dict = _generate_description_via_llm(table_meta)

    # Persist to cache
    cache[table_hash] = meta_dict
    
    # Also save a table-name to hash mapping for easy runtime lookup
    table_name_clean = table_meta["table_name"].lower()
    cache[f"__table_map__:{table_name_clean}"] = table_hash
    if "." in table_name_clean:
        short_name = table_name_clean.split(".")[-1]
        cache[f"__table_map__:{short_name}"] = table_hash
        
    _save_cache(cache)

    return meta_dict["table_description"]


def get_column_descriptions(table_name: str) -> dict:
    """
    Look up the column descriptions cache for a given table name.
    """
    try:
        cache = _load_cache()
        table_name_clean = table_name.lower()
        mapping_key = f"__table_map__:{table_name_clean}"
        
        # If mapping key doesn't exist, try table name without schema
        table_hash = cache.get(mapping_key)
        if not table_hash and "." in table_name_clean:
            short_name = table_name_clean.split(".")[-1]
            table_hash = cache.get(f"__table_map__:{short_name}")
            
        if table_hash and table_hash in cache:
            cached_val = cache[table_hash]
            if isinstance(cached_val, dict):
                return cached_val.get("columns", {})
    except Exception:
        pass
    return {}


def clear_cache() -> None:
    """Delete the description cache file."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"  Cleared cache: {CACHE_FILE}")
    else:
        print(f"  No cache file to clear: {CACHE_FILE}")


def get_cache_stats() -> dict:
    """Return basic stats about the description cache."""
    cache = _load_cache()
    # Subtract __table_map__ entries from total entries to show actual table count
    table_entries = sum(1 for k in cache.keys() if not k.startswith("__table_map__"))
    return {
        "cache_file": CACHE_FILE,
        "entries": table_entries,
        "file_exists": os.path.exists(CACHE_FILE),
    }


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Testing semantic description generator...\n")

    # Verify Groq API key is set
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("  [FAIL] GROQ_API_KEY not set in .env")
        sys.exit(1)
    print(f"  API Key  : {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}")

    # Create a fake table metadata for testing
    test_meta = {
        "table_name": "dbo.SupportTickets",
        "columns": [
            {"name": "TicketID", "type": "int"},
            {"name": "CustomerName", "type": "varchar(100)"},
            {"name": "Subject", "type": "varchar(255)"},
            {"name": "Priority", "type": "varchar(20)"},
            {"name": "Status", "type": "varchar(20)"},
            {"name": "AssignedTo", "type": "varchar(100)"},
            {"name": "CreatedDate", "type": "datetime"},
            {"name": "ResolvedDate", "type": "datetime"},
        ],
        "row_count": 5000,
        "sample_values": {
            "Priority": ["High", "Medium", "Low", "Critical"],
            "Status": ["Open", "Closed", "In Progress"],
            "AssignedTo": ["Alice Johnson", "Bob Smith"],
        },
    }

    # Test 1: Generate description (cache MISS)
    print("\n--- Test 1: Generate description (cache MISS) ---")
    clear_cache()
    import time
    start = time.time()
    desc = get_semantic_description(test_meta)
    elapsed = time.time() - start
    print(f"  Time     : {elapsed:.2f}s")
    print(f"  Result   : {desc[:200]}...")

    # Test 2: Get description again (cache HIT)
    print("\n--- Test 2: Get description (cache HIT) ---")
    start = time.time()
    desc2 = get_semantic_description(test_meta)
    elapsed = time.time() - start
    print(f"  Time     : {elapsed:.4f}s  (should be near-instant)")
    assert desc == desc2, "Cache returned different result!"
    print(f"  [OK] Cache hit -- same result, no API call")

    # Test 3: Cache stats
    print("\n--- Test 3: Cache stats ---")
    stats = get_cache_stats()
    print(f"  File     : {stats['cache_file']}")
    print(f"  Entries  : {stats['entries']}")
    print(f"  Exists   : {stats['file_exists']}")

    # Test 4: Modified schema triggers cache MISS
    print("\n--- Test 4: Schema change triggers fresh LLM call ---")
    modified_meta = {**test_meta, "columns": test_meta["columns"] + [
        {"name": "ClosedBy", "type": "varchar(100)"}
    ]}
    start = time.time()
    desc3 = get_semantic_description(modified_meta)
    elapsed = time.time() - start
    print(f"  Time     : {elapsed:.2f}s  (new LLM call)")
    print(f"  Result   : {desc3[:200]}...")

    stats = get_cache_stats()
    assert stats["entries"] == 2, f"Expected 2 cache entries, got {stats['entries']}"
    print(f"  [OK] Cache now has {stats['entries']} entries")

    # Cleanup
    print("\n--- Cleanup ---")
    clear_cache()

    print("\n" + "=" * 55)
    print("  [OK]  Semantic description -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
