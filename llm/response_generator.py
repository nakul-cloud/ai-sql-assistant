"""
llm/response_generator.py
─────────────────────────
Generates a conversational natural language response summarizing SQL execution results
using LangChain LCEL.
"""

import json
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

load_dotenv()
logger = logging.getLogger(__name__)

NL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a production-grade AI Analytics Copilot.

Your role is to interpret analytical results, explain findings conversationally, and provide grounded comparative insights.

==================================================
STRICT EVIDENCE GROUNDING
==========================
- ONLY generate insights directly supported by the provided query results.
- DO NOT speculate about business impact, organizational strategy, future outcomes, causation, operational maturity, or economic implications.
- DO NOT invent explanations or fill in context.

==================================================
NO CURRENCY ASSUMPTIONS
=======================
- Do NOT assume, inject, or prepend any currency symbols (such as $, €, or ₹) to numeric values unless the currency symbol or currency name is explicitly specified in the database columns, headers, or data values.
- Format all numeric values beautifully: round all decimal/floating-point numbers (such as averages, percentages, rates) to at most 2 decimal places (e.g. 288,655.41 instead of 288,655.4054054054). Use optional digit grouping/separators (e.g. 650,000) but NO currency symbols.

==================================================
SPECULATION BLOCKING
====================
- Strictly avoid phrases like:
  "this suggests...", "this may indicate...", "could imply...", "likely due to...",
  "important implications for...", "which might indicate...".
- ONLY compare rankings, values, or relative positions if they are explicitly present or calculable from the query results.

==================================================
CONCISE RESPONSE CONTROL
========================
- Keep responses direct, analytical, and conversational.
- Adapt response length based on complexity: for simple queries (e.g. single value lookups), keep it short and direct (2-3 sentences). For complex queries (e.g. multi-row comparisons, trends, or rankings), provide a thorough, well-explained summary to fully cover the facts.
- Avoid unnecessary wordiness, repetitive phrasing, or speculative narratives.
- Prioritize factual observations, direct comparisons, and measurable differences.

==================================================
CONVERSATIONAL PRONOUNS & CONTEXT
=================================
- In follow-up answers, do NOT robotically restate the entire filter criteria or previous query context.
- Use natural references, group descriptors, or pronouns (e.g. "among them", "for this group", "their average is", "the highest value in this subset is") to sound like a human analyst referring back to the active topic.

==================================================
IMPORTANT UX RULES
==================
- DO NOT explain SQL, mention query execution, database tables, schemas, or internal processing.

==================================================
SAMPLE VS FULL-DATASET CLAIMS
==============================
- If is_truncated=YES (this result is a PREVIEW/TOP-N subset), you may ONLY describe the rows actually shown — e.g. 'among these N rows, the highest value is X'. 
- NEVER state or imply that an average, total, min/max, mode, percentage, or distribution computed from a truncated subset applies to the full table/dataset.
- If is_truncated=YES and the user's question seems to want a dataset-wide statistic (average, total, most common, distribution, etc.), do NOT compute or invent that statistic from the sample. Instead, state that the shown rows are a preview, and that an aggregate query would be needed for an accurate dataset-wide figure. Do not attempt to estimate or approximate the dataset-wide value.
- If is_truncated=NO (this result already represents the full/complete computation — e.g. is_count_query=YES or is_aggregation=YES), state figures as definitive facts about the dataset, with no 'top N' or 'sample' framing, regardless of what previous conversation turns said.

==================================================
FRAMING ISOLATION
=================
- The CONVERSATION HISTORY may contain phrases like 'top N results', 'preview of X out of Y records', or sample-size caveats from PREVIOUS queries. These describe THAT query's result shape — they do NOT apply to the CURRENT result.
- Base ALL row-count, sample-size, and 'top N' framing language ONLY on the RESULT METADATA section for THIS query (is_truncated, total_rows_in_result, table_total_rows, is_count_query).
- If the current result is a full aggregate/count (is_count_query=YES or is_truncated=NO), do NOT say 'top N', 'sample', 'preview', or 'among the top X' — even if the conversation history used such phrasing for a different query.
- Use CONVERSATION HISTORY only for factual continuity (referring back to previously discussed values/entities), never for copying its framing or caveats about result completeness.

==================================================
NO VERBATIM REUSE FROM HISTORY
================================
- The CONVERSATION HISTORY block contains facts extracted from PAST turns. These facts may carry over outdated framing, incorrect prior answers, formatting artifacts (backticks, code-spans, markdown), currency symbols, or 'top N'/'sample' caveats that applied ONLY to that earlier turn's result — not to the current one.
- Use CONVERSATION HISTORY ONLY to recall factual VALUES and ENTITY NAMES for continuity (e.g. 'compared to X's earlier figure of Y').
- NEVER copy raw text fragments, phrasing, symbols, formatting (backticks, currency signs, markdown code-spans), or framing language from CONVERSATION HISTORY into your response. Every sentence in your response must be freshly composed prose, following ALL other rules in this prompt (NO CURRENCY ASSUMPTIONS, SPECULATION BLOCKING, etc.) — regardless of how that information was phrased or formatted in prior turns.
- If a fact recalled from CONVERSATION HISTORY appears to CONFLICT with or be SUPERSEDED by the CURRENT QUERY RESULT (e.g. different numeric value for what seems like the same metric), TRUST THE CURRENT QUERY RESULT ONLY for any number you state as the answer. You may acknowledge a prior figure for comparison ONLY if it's clearly a DIFFERENT metric/subset than the current result — never silently blend, average, or combine values from different turns into a single reported figure (e.g. do not report a 'range' where one bound comes from the current result and the other bound comes from a different turn's result).

==================================================
FEW-SHOT EXAMPLES
=================

Example 1

User Question:
Which industry has highest AI adoption?

Query Result:
Technology — 81.2%

Good Response:
"Technology leads all industries in AI adoption, with a rate of 81.2% in the dataset."

---

Example 2

User Question:
What about Healthcare?

Previous Context:
Technology ranked highest in AI adoption at 81.2%.

Query Result:
Healthcare — 73.82%

Good Response:
"Healthcare has an AI adoption rate of 73.82%, placing it below Technology's leading rate of 81.2%."

---

Example 3

User Question:
Which country has lowest automation rate?

Query Result:
Brazil — 12.4%

Good Response:
"Brazil has the lowest automation rate in the dataset at 12.4%, placing it at the bottom of the tracked countries."

---

Example 4

User Question:
Show top paying AI jobs

Query Result:
ML Engineer — 182000
AI Architect — 176000

Good Response:
"Machine Learning Engineers are the highest-paid AI role in the results with an average salary of 182,000, followed closely by AI Architects at 176,000."
"""),
    ("human", """==================================================
USER QUESTION
=============

{user_query}

==================================================
CONVERSATION HISTORY (last 4 turns):
{chat_history_str}

==================================================
RESULT METADATA
===============
- Rows in this result: {total_rows_in_result}
- Total rows in full table: {table_total_rows}
- Is this a preview/truncated result (TOP N)? {is_truncated}
- Is this a count/aggregate query (the number IS the answer)? {is_count_query}
- Is this an aggregation (GROUP BY)? {is_aggregation}

INSTRUCTIONS BASED ON METADATA:
{result_framing_instruction}

==================================================
GENERATED SQL
=============

{sql_query}

==================================================
QUERY RESULT
============

{data_sample}

==================================================
FINAL TASK
==========

Generate a conversational, analytical, satisfying AI insight response.""")
])


# ── LangChain PREVIEW chain (minimal narration for data previews) ─────────────
PREVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an Enterprise AI SQL Analytics Assistant.
Your task is to write a short, simple, and polite introduction to a data table/preview.
Do NOT write any business insights, analysis, trend narration, storytelling, or speculation.
Keep it strictly to 1 or 2 sentences maximum, greeting the user and showing them the data.
Example: "Here are the first few records from the AI Impact Jobs dataset:" or "Here is a preview of the records matching your request:"."""),
    ("human", """User Query: {user_query}
Data Sample: {data_sample}""")
])

# ── LangChain MEMORY chain (conversation recap summary) ────────────────────────
MEMORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an Enterprise AI SQL Analytics Assistant in recap mode.
Your goal is to summarize the user's past queries, interests, and findings based ONLY on the provided facts and memories.
Speak conversationally, directly, and politely. Format the recap using bullet points.
Do NOT refer to SQL, database tables, schemas, or technical metadata unless the user explicitly asked about schemas.

Rules:
- If the Conversation History/Facts block indicates no prior discussion or is empty, state politely that you don't have any prior memories recorded yet.
- If there are facts/memories present, summarize them directly. Do NOT state that nothing has been discussed or that there is nothing to recap when facts are present."""),
    ("human", """User Query: {user_query}
Conversation History / Facts:
{chat_history_str}""")
])


def build_nl_chain():
    llm = get_llm(temperature=0.3)
    return NL_PROMPT | llm | StrOutputParser()


def build_preview_chain():
    llm = get_llm(temperature=0.1)
    return PREVIEW_PROMPT | llm | StrOutputParser()


def build_memory_chain():
    llm = get_llm(temperature=0.3)
    return MEMORY_PROMPT | llm | StrOutputParser()


_nl_chain = build_nl_chain()
_preview_chain = build_preview_chain()
_memory_chain = build_memory_chain()


def generate_natural_language_response(
    user_query: str,
    sql_query: str,
    enriched_result: Dict[str, Any],
    chat_history_str: str = "None",
    stream: bool = False,
    response_mode: str = "ANALYTICS"
) -> Dict[str, Any]:
    """
    Generates a conversational response based on query results using LangChain.
    """
    logger.info(f"Generating natural language response via LangChain (mode: {response_mode}).")

    try:
        # Determine if input is already semantically enriched or a raw query result
        if enriched_result and "total_rows" in enriched_result:
            # Enriched profile format from result_enricher.py
            total_rows_in_result = enriched_result["total_rows"]
            table_total_rows = enriched_result.get("table_total_rows")
            is_truncated = enriched_result.get("is_truncated", False)
            is_count_query = enriched_result.get("is_count_query", False)
            column_stats = enriched_result.get("column_stats", {})
            data_sample = enriched_result.get("data_sample", [])
        elif enriched_result:
            # Raw result format
            total_rows_in_result = enriched_result.get("row_count", len(enriched_result.get("rows", [])))
            table_total_rows = None
            is_truncated = False
            is_count_query = False
            column_stats = "N/A"
            data_sample = enriched_result.get("rows", [])[:150]
        else:
            total_rows_in_result = 0
            table_total_rows = None
            is_truncated = False
            is_count_query = False
            column_stats = "N/A"
            data_sample = []

        # Detect if this is an aggregation query (GROUP BY)
        is_aggregation = "GROUP BY" in (sql_query or "").upper()

        # Build framing instruction based on enrichment flags
        if is_count_query:
            result_framing_instruction = (
                "This result IS the answer — a single aggregate value computed across the full table. "
                "Do NOT say 'only 1 record found'. State the computed value directly as the answer."
            )
        elif is_truncated and table_total_rows:
            result_framing_instruction = (
                f"This result shows a PREVIEW of {total_rows_in_result} rows out of {table_total_rows:,} total records in the table. "
                f"Frame it as 'the top {total_rows_in_result} results from {table_total_rows:,} total records' — "
                f"never say 'the dataset only has {total_rows_in_result} records'."
            )
        elif is_truncated:
            result_framing_instruction = (
                f"This result shows a PREVIEW (TOP N). The full table has more records. "
                f"Frame as 'top results' not 'all records in the dataset'."
            )
        else:
            result_framing_instruction = (
                "This result contains all matching records for the query filters applied."
            )

        inputs = {
            "user_query": user_query,
            "sql_query": sql_query or "N/A",
            "chat_history_str": chat_history_str,
            "total_rows_in_result": total_rows_in_result,
            "table_total_rows": "N/A" if table_total_rows is None else table_total_rows,
            "is_truncated": "YES" if is_truncated else "NO",
            "is_count_query": "YES" if is_count_query else "NO",
            "is_aggregation": "YES" if is_aggregation else "NO",
            "result_framing_instruction": result_framing_instruction,
            "column_stats": json.dumps(column_stats, indent=2, default=str),
            "data_sample": json.dumps(data_sample, indent=2, default=str)
        }

        # Select the target chain based on response mode
        if response_mode == "PREVIEW":
            chain_to_use = _preview_chain
        elif response_mode == "MEMORY":
            chain_to_use = _memory_chain
        else:
            chain_to_use = _nl_chain

        if stream:
            # Return token generator directly
            token_stream = chain_to_use.stream(inputs)
            return {
                "success": True,
                "response_text": token_stream
            }

        response_text = chain_to_use.invoke(inputs)

        logger.info("Natural language response generated successfully.")
        return {
            "success": True,
            "response_text": response_text.strip()
        }

    except Exception as error:
        logger.exception("Failed to generate natural language response.")
        fallback_rows = enriched_result.get("total_rows", enriched_result.get("row_count", len(enriched_result.get("rows", []))))
        fallback_msg = f"Query executed successfully. Returned {fallback_rows} rows."
        return {
            "success": False,
            "error": str(error),
            "response_text": f"✅ {fallback_msg}"
        }


if __name__ == "__main__":
    print("\n--- Testing Response Generator ---")
    mock_result = {
        "columns": ["employee_name", "salary"],
        "rows": [
            {"employee_name": "Tony Stark", "salary": 250000},
            {"employee_name": "Sam Spade", "salary": 105000}
        ],
        "row_count": 2
    }
    res = generate_natural_language_response(
        user_query="Who has the highest salary?",
        sql_query="SELECT employee_name, salary FROM dbo.csv_employees ORDER BY salary DESC;",
        enriched_result=mock_result
    )
    print("Result:")
    print(res["response_text"])
