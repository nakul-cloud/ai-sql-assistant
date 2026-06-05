"""
retrieval/query_router.py
─────────────────────────
Routes user queries into three distinct intents:
  1. CHAT        → General conversations/greetings
  2. SQL_QUERY   → Analytical questions requiring SQL Server queries
  3. SCHEMA_INFO → Questions about database structure, tables, and columns

Uses a hybrid regex pre-check + Gemini intent classifier.

Usage:
    python -m retrieval.query_router
"""

import os
import re
import sys
from dotenv import load_dotenv

# Lazy loading of Gemini client to ensure correct PyTorch thread init order
from indexing.semantic_description import _get_gemini_client

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Regex patterns for direct routing
CHAT_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|yo)(\b|\s|$)",
    r"^(how\s+are\s+you|what's\s+up|how's\s+it\s+going)(\b|\s|$)",
    r"^(thank\s+you|thanks|cheers|awesome|great|cool)(\b|\s|$)",
    r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+do\s+you\s+do|what\s+can\s+you\s+do)(\b|\s|$)"
]

SQL_PATTERNS = [
    r"^(show|select|get|fetch|list|display|find|query|count|sum|average|avg|max|min|total|top|highest|lowest)\b",
    r"\b(table|view|report|data|sales|revenue|employees?|departments?|salaries|hire\s+date|transactions?)\b",
    r"\b(group\s+by|order\s+by|where|having|join|join\s+on)\b"
]

SCHEMA_PATTERNS = [
    r"^(what\s+tables|list\s+tables|show\s+tables|what\s+schemas|what\s+is\s+the\s+schema)\b",
    r"\b(columns?|fields?|data\s+types?|primary\s+key|foreign\s+key|table\s+structure|describe\s+table)\b"
]


def pre_check_intent(user_query: str) -> str | None:
    """
    Apply fast regex matching to determine intent.
    Returns:
        'CHAT', 'SQL_QUERY', 'SCHEMA_INFO', or None (if ambiguous)
    """
    q = user_query.strip().lower()

    # 1. Check schema info patterns
    for pattern in SCHEMA_PATTERNS:
        if re.search(pattern, q):
            return "SCHEMA_INFO"

    # 2. Check chat patterns
    for pattern in CHAT_PATTERNS:
        if re.match(pattern, q):
            return "CHAT"

    # 3. Check obvious SQL patterns
    for pattern in SQL_PATTERNS:
        if re.search(pattern, q):
            return "SQL_QUERY"

    return None


def llm_classify_intent(user_query: str) -> str:
    """
    Use Gemini Flash to classify ambiguous queries.
    """
    prompt = f"""
    You are an intent classifier for a database assistant.
    Classify the user's query into one of three classes:
    
    1. CHAT: General greeting, chat, thanks, or simple conversational query.
    2. SQL_QUERY: A request to retrieve, filter, aggregate, or analyze data from tables (e.g. employee lists, sales figures, revenue totals).
    3. SCHEMA_INFO: A request specifically asking about table definitions, column names, keys, schema structures, or what tables exist.

    Respond with ONLY the class name: CHAT, SQL_QUERY, or SCHEMA_INFO. Do not write anything else.

    Query: "{user_query}"
    Class:
    """
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        intent = response.text.strip().upper()
        if intent in ("CHAT", "SQL_QUERY", "SCHEMA_INFO"):
            return intent
        # Fallback parsing
        for possible in ("CHAT", "SQL_QUERY", "SCHEMA_INFO"):
            if possible in intent:
                return possible
        return "SQL_QUERY"  # Default fallback
    except Exception as e:
        print(f"  [WARNING] LLM intent classification failed: {e}. Defaulting to SQL_QUERY.")
        return "SQL_QUERY"


def route_query(user_query: str) -> str:
    """
    Determine query routing path.
    """
    # 1. Try regex pre-check (0ms)
    intent = pre_check_intent(user_query)
    if intent:
        return intent

    # 2. Use Gemini if ambiguous (~300ms)
    return llm_classify_intent(user_query)


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Testing query router...\n")

    test_queries = [
        ("Hello!", "CHAT"),
        ("Thanks for the help", "CHAT"),
        ("Show me all employees earning more than 80000", "SQL_QUERY"),
        ("What is the total sales amount in May 2026?", "SQL_QUERY"),
        ("List the columns in the csv_sales table", "SCHEMA_INFO"),
        ("What tables do we have in this database?", "SCHEMA_INFO"),
        ("Can you tell me about the weather?", "CHAT"),  # ambiguous/general -> Classified by LLM
    ]

    for q, expected in test_queries:
        intent = route_query(q)
        status = "[OK]" if intent == expected else "[FAIL]"
        print(f"  Query    : {q}")
        print(f"  Expected : {expected}")
        print(f"  Detected : {intent} {status}\n")

    print("=======================================================")
    print("  [OK] Query router -- all tests passed!")
    print("=======================================================")
    sys.exit(0)
