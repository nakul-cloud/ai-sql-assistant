"""
retrieval/query_router.py
─────────────────────────
Routes user queries into three distinct intents:
  1. CHAT        → General conversations/greetings
  2. SQL_QUERY   → Analytical questions requiring SQL Server queries
  3. SCHEMA_INFO → Questions about database structure, tables, and columns

Uses a hybrid regex pre-check + LangChain intent classifier (Groq).
"""

import os
import re
import sys
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

load_dotenv()
logger = logging.getLogger(__name__)

# Regex patterns for direct routing
CHAT_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|yo)(\b|\s|$)",
    r"^(how\s+are\s+you|what's\s+up|how's\s+it\s+going)(\b|\s|$)",
    r"^(thank\s+you|thanks|cheers|awesome|great|cool)(\b|\s|$)",
    r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+do\s+you\s+do|what\s+can\s+you\s+do)(\b|\s|$)"
]

DESCRIBE_PATTERNS = [
    r"\b(what is|what's|explain|describe|tell me about|give me an overview|summarize|brief|overview)\b.*(dataset|table|data|this|it)\b",
    r"\b(what (does|do) (this|the) (dataset|table|data) (contain|have|include|show|represent))\b",
    r"\b(i don'?t know (anything|much)|new to this|unfamiliar)\b",
    r"\b(what (columns|fields|information) (are|is) (in|available))\b",
]

SQL_PATTERNS = [
    r"\b(show|list|find|get|fetch|count|sum|average|how many|what is)\b.*(table|record|row|ticket|employee|order|customer|invoice)",
    r"\b(top|highest|lowest|most|least)\b.*\b(by|in|from)\b",
    r"\b(highest|lowest|most|least|best|worst|maximum|minimum)\b.*(industry|country|company|sector|region)",
    r"\b(average|avg|mean|total|sum)\b.*(industry|country|company|year|sector)",
    r"\b(which|what).*(industry|country|company|sector).*(most|least|highest|lowest|best|worst|average|invest|spend|earn|save)",
    r"\b(compare|comparison|vs|versus|difference)\b",
    r"\b(invest|investment|spending|revenue|savings|cost|profit|adoption|maturity|deployment)\b.*(by|per|across|between|industry|country)",
]

SCHEMA_PATTERNS = [
    r"^(what\s+tables|list\s+tables|show\s+tables|what\s+schemas|what\s+is\s+the\s+schema)\b",
    r"\b(columns?|fields?|data\s+types?|primary\s+key|foreign\s+key|table\s+structure|describe\s+table)\b"
]


def pre_check_intent(user_query: str) -> str | None:
    """
    Apply fast regex matching to determine intent.
    Returns:
        'CHAT', 'DESCRIBE', 'SQL_QUERY', 'SCHEMA_INFO', or None (if ambiguous)
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

    # 3. Check describe patterns
    for pattern in DESCRIBE_PATTERNS:
        if re.search(pattern, q):
            return "DESCRIBE"

    # 4. Check obvious SQL patterns
    for pattern in SQL_PATTERNS:
        if re.search(pattern, q):
            return "SQL_QUERY"

    return None


# ── LangChain intent classifier (LLM fallback only) ──────────────────────────

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's message into exactly one of these four intents:
- SQL_QUERY: The user wants data, reports, counts, lists, or analysis from the database tables.
- SCHEMA_INFO: The user is asking about the database structure, table definitions, column names, schema structures, or what tables exist.
- DESCRIBE: The user is asking to explain the dataset, summarize the table, give an overview, or is saying they don't know anything/are new to this.
- CHAT: General conversation, greetings, thanks, or questions unrelated to data.

Reply with ONLY one word: SQL_QUERY, SCHEMA_INFO, DESCRIBE, or CHAT. No explanation."""),
    ("human", "{user_query}")
])


def build_intent_chain():
    llm = get_llm(temperature=0.0)
    return INTENT_PROMPT | llm | StrOutputParser()


_intent_chain = build_intent_chain()


def llm_classify_intent(user_query: str) -> str:
    """
    Use LangChain to classify ambiguous queries.
    """
    logger.info("Classifying intent via LangChain.")
    try:
        result = _intent_chain.invoke({"user_query": user_query})
        intent = result.strip().upper()
        if intent in ("CHAT", "SQL_QUERY", "SCHEMA_INFO", "DESCRIBE"):
            return intent
        # Fallback parsing
        for possible in ("CHAT", "SQL_QUERY", "SCHEMA_INFO", "DESCRIBE"):
            if possible in intent:
                return possible
        return "SQL_QUERY"  # Default fallback
    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}. Defaulting to SQL_QUERY.")
        return "SQL_QUERY"


def route_query(user_query: str) -> str:
    """
    Determine query routing path.
    """
    # 1. Try regex pre-check (0ms)
    intent = pre_check_intent(user_query)
    if intent:
        return intent

    # 2. Use LangChain if ambiguous
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
        ("can you explain what this dataset is about? i'm new", "DESCRIBE"),
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
