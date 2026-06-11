"""
retrieval/query_router.py
─────────────────────────
Routes user queries into distinct intents:
  1. CHAT        → General conversations / greetings
  2. TEMPORAL    → Questions about current date/time/day (hardcoded refusal)
  3. SCHEMA_INFO → Questions about database structure, tables, columns
  4. DESCRIBE    → "What is this dataset about?" type questions
  5. SQL_QUERY   → Analytical questions requiring SQL execution

Routing strategy:
  - Static regex  → CHAT, TEMPORAL  (never need DB knowledge)
  - Dynamic regex → SCHEMA_INFO, DESCRIBE, SQL_QUERY  (built from live schema)
  - LLM fallback  → ambiguous queries that pass all regex checks
"""

import os
import re
import sys
import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

load_dotenv()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STATIC PATTERNS (never depend on schema)
# ══════════════════════════════════════════════════════════════════════════════

# These never change regardless of what's in the database.

CHAT_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|yo)(\b|\s|$)",
    r"^(how\s+are\s+you|what's\s+up|how's\s+it\s+going)(\b|\s|$)",
    r"^(thank\s+you|thanks|cheers|awesome|great|cool)(\b|\s|$)",
    r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+do\s+you\s+do|what\s+can\s+you\s+do)(\b|\s|$)",
]

TEMPORAL_PATTERNS = [
    r"\b(what|whats|what's)\b.*(today|current\s+date|current\s+time|right\s+now)\b",
    r"\bwhat\s+(day|date|time|year|month)\s+(is\s+it|is\s+today|are\s+we\s+in)\b",
    r"\bhow\s+long\s+(ago|since|until|before)\b",
    r"\b\d{4}\b.*\b(ago|years?\s+ago|months?\s+ago|days?\s+ago)\b",
    r"\b(years?|months?|days?|weeks?)\s+(ago|back|later|forward)\b",
    r"\bhow\s+many\s+(years?|months?|days?|weeks?)\s+(ago|since|until)\b",
    r"\b(current|today'?s?|right\s+now|at\s+the\s+moment)\s+(date|time|day|year|month)\b",
]

# DB structure questions — fully domain-agnostic, no column/table words needed
SCHEMA_PATTERNS = [
    r"\b(what\s+tables?|list\s+tables?|show\s+tables?|all\s+tables?)\b",
    r"\b(what\s+schemas?|show\s+schemas?|database\s+structure)\b",
    r"\b(columns?|fields?|data\s+types?|primary\s+key|foreign\s+key|table\s+structure)\b",
    r"\bwhat\s+(columns?|fields?)\s+(are|exist|do\s+we\s+have)\b",
    r"\bdescribe\s+table\b",
    r"\bshow\s+me\s+the\s+schema\b",
]

# Dataset overview — domain-agnostic
# Dataset overview — domain-agnostic
DESCRIBE_PATTERNS = [
    r"\b(explain|describe|tell\s+me\s+about|give\s+me\s+an?\s+overview|summarize|sumarize|summarise|sumarise)\b.*\b(datasets?|data|tables?|this|it|trend|trends)\b",
    r"\bwhat\s+(is|are|does|do)\b.*\b(datasets?|data|tables?)\b.*\b(about|contain|have|include|show|represent|track|cover|purpose)\b",
    r"\b(i\s+don'?t\s+know|new\s+to\s+this|unfamiliar|no\s+idea|help\s+me\s+understand)\b",
    r"\bwhat\s+kind\s+of\s+(data|information|questions?)\b",
    r"\bwhat\s+can\s+(i|we|you)\s+(ask|query|find|look\s+up)\b",
    r"\bwhat\s+is\s+this\s+(data|datasets?|tables?)\s+about\b",
    r"\b(move\s+to|switch\s+to|change\s+to|focus\s+on|look\s+at|tell\s+me\s+about|show\s+me|show)\s+(the\s+)?([\w\s]+)?\b(datasets?|data|tables?)\b",
    r"\b(summarize|sumarize|summarise|sumarise|explain|describe)\b\s+(the\s+)?([\w\s]+)?\b(datasets?|data|tables?|trend|trends)\b",
]

# General knowledge and politics (off-topic queries)
GENERAL_KNOWLEDGE_PATTERNS = [
    r"^who\s+is\s+(the\s+)?(pm|president|prime\s+minister|ceo|cto|cfo|governor|minister|king|queen|chancellor)\b",
    r"\bwho\s+is\s+(the\s+)?(current|present|today'?s?)?\s*(president|pm|prime\s+minister|ceo|leader|head)\b",
    r"\bwhat\s+is\s+(the\s+)?capital\s+of\b",
    r"\bwho\s+(won|is\s+winning)\s+(the\s+)?(election|war|match|game|world\s+cup)\b",
    r"\b(current|latest|recent)\s+(news|events?|updates?|headlines?)\b",
    r"\bwho\s+is\s+[a-z\s]+\??\s*$",  # catches bare "who is X?" questions
]

# Universal SQL intent signals — work for ANY domain
UNIVERSAL_SQL_PATTERNS = [
    r"\b(show|list|find|get|fetch|retrieve|display)\b",
    r"\b(count|sum|total|average|avg|mean|max|min|maximum|minimum)\b",
    r"\b(how\s+many|how\s+much|what\s+is\s+the|what\s+are\s+the)\b",
    r"\b(top|bottom|highest|lowest|most|least|best|worst|ranking|ranked)\b",
    r"\b(compare|comparison|vs|versus|difference|between)\b",
    r"\b(group\s+by|grouped\s+by|broken\s+down|by\s+category|by\s+type)\b",
    r"\b(filter|where|only|excluding|except|greater\s+than|less\s+than|more\s+than)\b",
    r"\b(sort|order|sorted|ordered|ascending|descending|asc|desc)\b",
    r"\b(trend|over\s+time|year\s+by\s+year|monthly|yearly|annual|quarterly)\b",
    r"\b(percentage|percent|ratio|proportion|share)\b",
    r"\b(which|who|whose)\b.{0,50}\b(highest|lowest|most|least|best|worst|maximum|minimum)\b",
    r"\b(all|every|each)\b.{0,20}\b(record|entry|row|result|item)\b",
    r"^(what\s+is|what\s+are|what's|whats|who\s+is|who's|which|where\s+is|where\s+are)\b",
    r"\b(details|info|information|data|stats|statistics|report)\b",
    r"\b(per|each|every|by)\b",
    r"\b(number\s+of|total\s+number\s+of|amount\s+of|count\s+of)\b",
    r"\b(than|over|under|above|below)\b",
    r"\b(highest|lowest|cheapest|most\s+expensive|earliest|latest)\b",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DYNAMIC PATTERN BUILDER (from live schema)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_schema_terms() -> dict:
    """
    Pulls table names and column names from the live database.
    Cached after first call — refreshed only on app restart or manual invalidation.

    Returns a dict:
    {
        "table_names": ["employees", "sales", "tickets", ...],
        "column_names": ["salary", "revenue", "status", ...],
        "sample_values": ["female", "male", "engineering", ...],
        "all_terms": ["employees", "salary", "revenue", ...]  # combined, deduplicated
    }
    """
    try:
        from database.schema_manager import fetch_database_metadata
        metadata_list = fetch_database_metadata()

        table_names = []
        column_names = []
        sample_values = []

        for meta in metadata_list:
            # Clean table name: "dbo.csv_employees" → "employees"
            raw_name = meta.get("table_name", "")
            clean_name = raw_name.split(".")[-1]           # strip schema prefix
            clean_name = re.sub(r"^(csv_|tbl_|t_)", "", clean_name)  # strip common prefixes
            clean_name = clean_name.replace("_", " ")      # "ai_adoption" → "ai adoption"

            if clean_name:
                table_names.append(clean_name.lower())
                # Also add the underscore version for exact matching
                table_names.append(raw_name.split(".")[-1].lower())

            for col in meta.get("columns", []):
                col_name = col.get("name", "").replace("_", " ").lower()
                if col_name and len(col_name) > 2:  # skip 1-2 char column names (id, pk)
                    column_names.append(col_name)
                    # Also add underscore version
                    column_names.append(col.get("name", "").lower())

            # Extract text sample values dynamically
            for col_name, vals in meta.get("sample_values", {}).items():
                for val in vals:
                    val_str = str(val).strip().lower()
                    # Skip empty values, short terms, and purely numeric/float values
                    if val_str and len(val_str) > 1 and not val_str.replace('.', '', 1).isdigit():
                        sample_values.append(val_str)

        # Deduplicate preserving order
        seen = set()
        unique_tables = []
        for t in table_names:
            if t not in seen:
                seen.add(t)
                unique_tables.append(t)

        seen = set()
        unique_cols = []
        for c in column_names:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)

        all_terms = list(set(unique_tables + unique_cols))
        unique_samples = list(set(sample_values))

        logger.info(f"Schema terms loaded: {len(unique_tables)} tables, {len(unique_cols)} columns, {len(unique_samples)} sample values.")
        return {
            "table_names": unique_tables,
            "column_names": unique_cols,
            "sample_values": unique_samples,
            "all_terms": all_terms
        }

    except Exception as e:
        logger.warning(f"Could not load schema terms for dynamic routing: {e}")
        return {"table_names": [], "column_names": [], "sample_values": [], "all_terms": []}


@lru_cache(maxsize=1)
def _build_dynamic_sql_pattern() -> Optional[re.Pattern]:
    """
    Builds a single compiled regex that matches any query mentioning
    a known table name or column name from the live schema.

    Example output pattern (for a DB with employees + salary + revenue):
    r"\b(employees|salary|revenue|ai adoption|deployment count|...)\b"
    """
    terms = _load_schema_terms()["all_terms"]
    if not terms:
        return None

    # Sort by length descending so longer phrases match before substrings
    sorted_terms = sorted(terms, key=len, reverse=True)

    # Escape special regex chars in term names
    escaped = [re.escape(t) for t in sorted_terms if len(t) > 2]

    if not escaped:
        return None

    pattern_str = r"\b(" + "|".join(escaped) + r")\b"
    try:
        return re.compile(pattern_str, re.IGNORECASE)
    except re.error as e:
        logger.warning(f"Could not compile dynamic SQL pattern: {e}")
        return None


def invalidate_schema_cache():
    """
    Call this after a CSV upload or schema change so the router picks up new tables/columns.
    """
    _load_schema_terms.cache_clear()
    _build_dynamic_sql_pattern.cache_clear()
    logger.info("Schema term cache invalidated — will reload on next query.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def pre_check_intent(user_query: str) -> Optional[str]:
    """
    Fast regex-based intent detection. Returns intent string or None if ambiguous.

    Check order matters:
    1. TEMPORAL  — must be before SQL (date queries can contain numbers/years)
    2. CHAT      — must be before SQL (greetings can mention data words)
    3. GENERAL_KNOWLEDGE — must be before SQL (general "who is" off-topic questions)
    4. SCHEMA_INFO
    5. DESCRIBE
    6. SQL_QUERY — universal signals first, then dynamic schema terms
    """
    q = user_query.strip().lower()

    # ── 1. TEMPORAL (hardcoded refusal — never reaches LLM) ──────────────────
    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            logger.debug(f"TEMPORAL match: '{q}'")
            return "TEMPORAL"

    # ── 2. CHAT ───────────────────────────────────────────────────────────────
    for pattern in CHAT_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            logger.debug(f"CHAT match: '{q}'")
            return "CHAT"

    # ── 3. GENERAL_KNOWLEDGE (before SQL — "who is X" can look like SQL) ─────
    for pattern in GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            logger.debug(f"GENERAL_KNOWLEDGE match: '{q}'")
            return "GENERAL_KNOWLEDGE"

    # ── 4. SCHEMA_INFO ────────────────────────────────────────────────────────
    for pattern in SCHEMA_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            logger.debug(f"SCHEMA_INFO match: '{q}'")
            return "SCHEMA_INFO"

    # ── 5. DESCRIBE ───────────────────────────────────────────────────────────
    for pattern in DESCRIBE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            # Bypass DESCRIBE if the query indicates subset filtering or data aggregation
            bypass_describe = False
            if re.search(r"\b(only|where|having|when|by|for|who|whose|which|with|vs|versus|compare|comparison|difference|between)\b", q):
                if not re.search(r"\b(move|switch|change|focus)\b", q):
                    bypass_describe = True
            
            # Dynamic check: if query mentions any known categorical/text sample value from the database, bypass DESCRIBE
            terms = _load_schema_terms()
            for val in terms.get("sample_values", []):
                # Search value as a whole word to prevent partial matching (e.g. "it" in "split")
                if re.search(r"\b" + re.escape(val) + r"\b", q):
                    bypass_describe = True
                    break
            
            if not bypass_describe:
                logger.debug(f"DESCRIBE match: '{q}'")
                return "DESCRIBE"

    # ── 6. SQL_QUERY — universal signals ─────────────────────────────────────
    for pattern in UNIVERSAL_SQL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            logger.debug(f"SQL_QUERY (universal) match: '{q}'")
            return "SQL_QUERY"

    # ── 7. SQL_QUERY — dynamic schema term match ─────────────────────────────
    dynamic_pattern = _build_dynamic_sql_pattern()
    if dynamic_pattern and dynamic_pattern.search(q):
        logger.debug(f"SQL_QUERY (dynamic schema) match: '{q}'")
        return "SQL_QUERY"

    # ── 8. Ambiguous — defer to LLM ──────────────────────────────────────────
    return None


# ── LangChain intent classifier (LLM fallback only) ──────────────────────────

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's message into exactly one of these six intents:

- SQL_QUERY        : The user wants data, reports, counts, lists, or analysis from the database.
- SCHEMA_INFO      : The user is asking about database structure, table names, column names, or schema.
- DESCRIBE         : The user wants a plain English explanation of what a dataset or table contains.
- TEMPORAL         : The user is asking about the current date, time, day, or how long ago something was.
- GENERAL_KNOWLEDGE: The user asks about real-world facts, people, politics, news, or events not related to the database (e.g. "who is the PM", "what is the capital of France").
- CHAT             : General conversation, greetings, thanks, or small talk.

Reply with ONLY one word from the list above. No explanation, no punctuation."""),
    ("human", "{user_query}")
])


def build_intent_chain():
    llm = get_llm(temperature=0.0)
    return INTENT_PROMPT | llm | StrOutputParser()


_intent_chain = build_intent_chain()

VALID_INTENTS = {"CHAT", "SQL_QUERY", "SCHEMA_INFO", "DESCRIBE", "TEMPORAL", "GENERAL_KNOWLEDGE"}


def llm_classify_intent(user_query: str) -> str:
    """
    LangChain LLM fallback for ambiguous queries.
    Only called when all regex checks return None.
    """
    logger.info(f"Ambiguous query — classifying via LLM: '{user_query}'")
    try:
        result = _intent_chain.invoke({"user_query": user_query})
        intent = result.strip().upper()

        if intent in VALID_INTENTS:
            return intent

        # Partial match fallback
        for possible in VALID_INTENTS:
            if possible in intent:
                return possible

        logger.warning(f"LLM returned unrecognized intent '{intent}' — defaulting to SQL_QUERY.")
        return "SQL_QUERY"

    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e} — defaulting to SQL_QUERY.")
        return "SQL_QUERY"


def route_query(user_query: str) -> str:
    """
    Main entry point. Returns one of:
    CHAT | TEMPORAL | SCHEMA_INFO | DESCRIBE | SQL_QUERY
    """
    intent = pre_check_intent(user_query)
    if intent:
        logger.info(f"Intent resolved by regex: {intent}")
        return intent

    intent = llm_classify_intent(user_query)
    logger.info(f"Intent resolved by LLM: {intent}")
    return intent


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n[INFO] Testing query router...\n")

    test_queries = [
        # CHAT
        ("Hello!", "CHAT"),
        ("Thanks for the help", "CHAT"),
        ("Who are you?", "CHAT"),

        # TEMPORAL — must never reach LLM
        ("How long ago was 2015?", "TEMPORAL"),
        ("What is today's date?", "TEMPORAL"),
        ("What day is it?", "TEMPORAL"),
        ("How many years ago was 2010?", "TEMPORAL"),

        # GENERAL_KNOWLEDGE
        ("who is pm ?", "GENERAL_KNOWLEDGE"),
        ("who is President of USA ?", "GENERAL_KNOWLEDGE"),
        ("what is the capital of France?", "GENERAL_KNOWLEDGE"),
        ("who won the 2024 election?", "GENERAL_KNOWLEDGE"),

        # SCHEMA_INFO
        ("What tables do we have in this database?", "SCHEMA_INFO"),
        ("List the columns in the sales table", "SCHEMA_INFO"),
        ("Show me the schema", "SCHEMA_INFO"),

        # DESCRIBE
        ("What is this dataset about?", "DESCRIBE"),
        ("Can you explain the data?", "DESCRIBE"),
        ("I'm new to this, help me understand", "DESCRIBE"),
        ("What kind of questions can I ask?", "DESCRIBE"),

        # SQL_QUERY — universal signals
        ("Show me the top 5 records", "SQL_QUERY"),
        ("What is the average score?", "SQL_QUERY"),
        ("Count how many entries exist", "SQL_QUERY"),
        ("Compare results between groups", "SQL_QUERY"),

        # SQL_QUERY — should also catch dynamic schema terms if DB is connected
        ("Show me all employees earning more than 80000", "SQL_QUERY"),
        ("What is the total sales amount?", "SQL_QUERY"),
        ("Which industry has the highest AI investment?", "SQL_QUERY"),
    ]

    passed = 0
    failed = 0

    for q, expected in test_queries:
        detected = route_query(q)
        status = "[OK]" if detected == expected else "[FAIL]"
        if detected == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status:6s}  [{expected:12s}]  Query: '{q}'")
        if detected != expected:
            print(f"       Got: {detected}")

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(test_queries)} tests")
    print(f"{'='*55}")
    sys.exit(0 if failed == 0 else 1)
