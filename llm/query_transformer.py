"""
llm/query_transformer.py
────────────────────────
Advanced Query Transformation layer.

Three strategies applied in order:
1. Query Decomposition — splits compound questions into independent sub-queries
2. Query Rewriting     — aligns user terminology with database schema terms
3. Step-Back Prompting — generates a broader context query for narrow filters

Plugs in AFTER mem0 contextualization and BEFORE cache check / SQL generation.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TransformResult:
    original_query: str
    rewritten_query: str
    is_decomposed: bool = False
    sub_queries: List[str] = field(default_factory=list)
    stepback_query: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — Query Decomposition
# ══════════════════════════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query decomposition assistant for a SQL analytics chatbot.

Given a user's analytical question, determine if it contains multiple
INDEPENDENT sub-questions that would require querying DIFFERENT database
tables or computing DIFFERENT, UNRELATED metrics.

Rules:
- If the question asks ONE thing (even if complex with filters or grouping), return it as a single item.
- If it asks TWO or MORE independent analytical questions joined by "and",
  "also", "as well as", commas, or multiple question marks, split them.
- Each sub-question must be a complete, standalone analytical question.
- Do NOT split a single question that has multiple conditions (e.g. "salary by department and gender" is ONE question).
- Do NOT split comparative questions (e.g. "compare X and Y" is ONE question).
- Return a JSON array of strings. Minimum 1 item, maximum 4 items.
- Return ONLY the JSON array. No explanation, no markdown.

Examples:
Input: "What is the average salary?"
Output: ["What is the average salary?"]

Input: "Show top 5 employees and what is the AI adoption rate by industry?"
Output: ["Show top 5 employees", "What is the AI adoption rate by industry?"]

Input: "Average salary of placed candidates and AI spending by industry"
Output: ["What is the average salary of placed candidates?", "What is the AI spending by industry?"]

Input: "Show salary by department and gender"
Output: ["Show salary by department and gender"]

Input: "Compare Healthcare and Technology AI spending"
Output: ["Compare Healthcare and Technology AI spending"]
"""),
    ("human", "{query}")
])


def _has_compound_signals(query: str) -> bool:
    """Quick heuristic — returns True only if the query MIGHT be compound."""
    q = query.lower().strip()

    # Multiple question marks
    if query.count("?") >= 2:
        return True

    # Analytical verb ... conjunction ... analytical verb
    av = r'(?:show|list|find|get|what|how|count|average|total|number|display|give|tell)'
    pattern = rf'\b{av}\b.+\b(?:and|also|plus|as well as)\b.+\b{av}\b'
    if re.search(pattern, q):
        return True

    # Comma followed by an analytical verb
    if re.search(r',\s*(?:and\s+)?(?:what|how|show|list|find|get|count|average|total)\b', q):
        return True

    return False


def decompose_query(query: str) -> List[str]:
    """
    Check if a query is compound and split it into independent sub-queries.
    Returns [query] unchanged for single-part queries.
    Uses a fast heuristic pre-check to avoid unnecessary LLM calls.
    """
    if not _has_compound_signals(query):
        return [query]

    try:
        chain = DECOMPOSE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
        raw = chain.invoke({"query": query}).strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()

        sub_queries = json.loads(raw)
        if isinstance(sub_queries, list) and len(sub_queries) >= 1:
            sub_queries = [sq.strip() for sq in sub_queries if sq.strip()]
            if len(sub_queries) >= 1:
                logger.info(f"Query decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                return sub_queries

        return [query]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e} — using original query")
        return [query]


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — Query Rewriting (Schema-Aligned Synonym Expansion)
# ══════════════════════════════════════════════════════════════════════════════

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a SQL analytics chatbot.

Given the user's question and a list of available database terms (table names,
column names), rewrite the question to use terminology that more closely matches
the database schema.

Rules:
- Only change terms that have a clear synonym in the schema terms list.
- Do NOT change the meaning, intent, or scope of the question.
- Do NOT add information that wasn't in the original question.
- If all terms already match the schema, return the question EXACTLY unchanged.
- Return ONLY the rewritten question. No explanation, no quotes.
"""),
    ("human", """Available schema terms: {schema_terms}

Original question: {query}

Rewritten question:""")
])


def _get_schema_terms() -> set:
    """Extract all meaningful terms from the database schema."""
    try:
        from database.schema_manager import fetch_database_metadata
        metadata = fetch_database_metadata()
        terms = set()
        for meta in metadata:
            raw = meta["table_name"].split(".")[-1].replace("csv_", "").replace("tbl_", "")
            for w in raw.split("_"):
                if len(w) > 2:
                    terms.add(w.lower())
            for col in meta["columns"]:
                for w in col["name"].replace("_", " ").lower().split():
                    if len(w) > 2:
                        terms.add(w)
        return terms
    except Exception as e:
        logger.warning(f"Failed to get schema terms: {e}")
        return set()


def _needs_rewrite(query: str, schema_terms: set) -> bool:
    """Check if the query uses terms that don't match any schema term."""
    if not schema_terms:
        return False

    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "what", "how", "who", "which",
        "when", "where", "why", "show", "list", "find", "get", "give", "tell",
        "from", "with", "this", "that", "they", "them", "their", "have", "been",
        "will", "each", "make", "like", "many", "some", "than", "most", "very",
        "does", "about", "into", "over", "such", "average", "total", "count",
        "number", "much", "between", "across", "per", "every", "any",
    }
    content_words = q_words - stop_words

    if not content_words:
        return False

    # If at least one content word matches a schema term, probably fine
    if content_words & schema_terms:
        return False

    return True  # No content words match schema — rewrite could help


def rewrite_query(query: str, schema_terms: set = None) -> str:
    """
    Rewrite the query to align with database schema terminology.
    Only calls the LLM if the query uses terms not found in the schema.
    """
    if schema_terms is None:
        schema_terms = _get_schema_terms()

    if not _needs_rewrite(query, schema_terms):
        return query

    try:
        chain = REWRITE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
        rewritten = chain.invoke({
            "schema_terms": ", ".join(sorted(schema_terms)),
            "query": query
        }).strip().strip('"\'')

        if rewritten and rewritten != query:
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        return query
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e} — using original query")
        return query


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3 — Step-Back Prompting (Contextual Expansion)
# ══════════════════════════════════════════════════════════════════════════════

STEPBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an analytical assistant. Given a narrow, specific
analytical question that filters on ONE specific category or entity, generate a
broader "step-back" version that provides overall context for comparison.

Rules:
- The step-back question should ask for the SAME metric but across ALL
  categories/entities, not just the specific one mentioned.
- Return ONLY the broader question. No explanation.
- If the question is already broad/general (no specific entity filter), return
  the word NONE.

Examples:
- "What is the AI adoption rate for Healthcare?" → "What is the AI adoption rate across all industries?"
- "Show salary for Senior employees" → "What is the salary distribution across all seniority levels?"
- "How many companies in Agriculture?" → "How many companies are in each industry?"
- "What is the average salary?" → NONE
- "Show all employees" → NONE
"""),
    ("human", "{query}")
])


def _might_need_stepback(query: str) -> bool:
    """Quick heuristic to check if a query filters on a specific entity."""
    # Contains "for/in/of/about/from <Capitalized Word>"
    if re.search(r'\b(?:for|in|of|about|from)\s+[A-Z][a-z]+', query):
        return True

    # Contains a quoted value
    if '"' in query or "'" in query:
        return True

    # Short query with a capitalized entity word
    words = query.split()
    skip = {"What", "How", "Show", "List", "Find", "Get", "Which", "Who",
            "Where", "When", "Does", "The", "Is", "Are", "Can", "Do"}
    if len(words) <= 8 and any(w[0].isupper() and w not in skip for w in words if w):
        return True

    return False


def generate_stepback_query(query: str) -> Optional[str]:
    """
    Generate a broader context query for narrow-filter questions.
    Returns None if the query is already broad or step-back isn't needed.
    """
    if not _might_need_stepback(query):
        return None

    try:
        chain = STEPBACK_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
        result = chain.invoke({"query": query}).strip().strip('"\'')

        if result.upper() == "NONE" or not result or result == query:
            return None

        logger.info(f"Step-back query generated: '{query}' → '{result}'")
        return result
    except Exception as e:
        logger.warning(f"Step-back generation failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def transform_query(query: str) -> TransformResult:
    """
    Main entry point. Applies query transformations in order:
    1. Decomposition (if compound → return sub-queries, skip 2 & 3)
    2. Rewriting (align with schema terms)
    3. Step-back (generate broader context query)
    """
    result = TransformResult(original_query=query, rewritten_query=query)

    # 1. Decomposition
    sub_queries = decompose_query(query)
    if len(sub_queries) > 1:
        result.is_decomposed = True
        result.sub_queries = sub_queries
        logger.info(f"Query decomposed into {len(sub_queries)} parts — skipping rewrite/stepback")
        return result

    # 2. Rewriting
    schema_terms = _get_schema_terms()
    rewritten = rewrite_query(query, schema_terms)
    result.rewritten_query = rewritten

    # 3. Step-back
    stepback = generate_stepback_query(rewritten)
    result.stepback_query = stepback

    return result


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dotenv import load_dotenv
    load_dotenv(override=True)

    print("\n--- Testing Query Transformer ---\n")

    test_queries = [
        "What is the average salary of placed candidates?",
        "What is the average salary of placed candidates, and how much did Agriculture spend on AI?",
        "Show me workers in the tech sector",
        "What about Healthcare?",
        "Show salary by department and gender",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Input: {q}")
        r = transform_query(q)
        print(f"Decomposed: {r.is_decomposed}")
        if r.is_decomposed:
            print(f"Sub-queries: {r.sub_queries}")
        else:
            print(f"Rewritten:   {r.rewritten_query}")
            print(f"Step-back:   {r.stepback_query}")
