"""
indexing/semantic_description.py
────────────────────────────────
Generates LLM-powered business descriptions for database tables,
with a local JSON cache to avoid redundant API calls.

How caching works:
  - Each table's structure (name + columns + types) is hashed (SHA-256).
  - If the hash exists in the cache, the stored description is reused.
  - If the hash is missing (new table or schema changed), a fresh
    Gemini API call is made and the result is cached.

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

from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────
CACHE_FILE = os.getenv("DESCRIPTION_CACHE_FILE", "indexing/description_cache.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Configure Gemini client (lazy load to avoid runtime library conflicts)
_gemini_client = None

def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


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


def _generate_description_via_llm(table_meta: dict) -> str:
    """
    Call Gemini Flash to generate a 3-sentence business description
    of a database table.
    """
    column_list = ", ".join(c["name"] for c in table_meta["columns"])
    sample_info = ""
    if table_meta.get("sample_values"):
        sample_lines = []
        for col, vals in list(table_meta["sample_values"].items())[:5]:
            sample_lines.append(f"  {col}: {vals}")
        sample_info = "\nSample values:\n" + "\n".join(sample_lines)

    prompt = f"""You are a database documentation expert.
Write a 3-sentence natural language description of this database table.
Include what business questions it can answer and what kind of data it holds.
Focus on business meaning, not technical structure.

Table: {table_meta['table_name']}
Columns: {column_list}
Row Count: {table_meta.get('row_count', 'unknown')}{sample_info}

Description:"""

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def get_semantic_description(table_meta: dict) -> str:
    """
    Return a semantic business description for a table.
    Uses cached version if available; calls Gemini if not.

    Args:
        table_meta: Dict with keys: table_name, columns, row_count, sample_values

    Returns:
        A 3-sentence business description string.
    """
    cache = _load_cache()
    table_hash = _compute_table_hash(table_meta)

    # Cache HIT — no LLM call
    if table_hash in cache:
        return cache[table_hash]

    # Cache MISS — call Gemini
    description = _generate_description_via_llm(table_meta)

    # Persist to cache
    cache[table_hash] = description
    _save_cache(cache)

    return description


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
    return {
        "cache_file": CACHE_FILE,
        "entries": len(cache),
        "file_exists": os.path.exists(CACHE_FILE),
    }


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Testing semantic description generator...\n")

    # Verify Gemini API key is set
    if not GEMINI_API_KEY:
        print("  [FAIL] GEMINI_API_KEY not set in .env")
        sys.exit(1)
    print(f"  API Key  : {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}")

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
