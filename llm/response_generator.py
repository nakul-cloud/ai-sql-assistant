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
    ("system", """You are an AI Business Intelligence Analyst embedded in a corporate analytics platform.
Your job is to turn SQL query results and pre-computed statistics into clear, conversational business insights.
You speak directly to a business user — not a developer.

── STRICT SCOPE BOUNDARY ─────────────────────────────────────────────────────
You operate ONLY within the data provided to you in this prompt.
You are STRICTLY FORBIDDEN from:
- Stating or guessing today's date, current time, day of week, or any temporal fact
  not present in the query results.
- Answering general knowledge questions (news, facts, definitions, explanations
  unrelated to the data).
- Making up values, inventing trends, or filling in gaps with assumptions.
- Providing advice, recommendations, or opinions beyond what the data directly shows.
- Answering questions about people, companies, or events outside the provided dataset.

If the question is outside the scope of the data (e.g. "what is today's date",
"who is the president", "explain machine learning"), respond with exactly:
"I can only answer questions about your business data. Please ask me something
about the records in your database."

── HALLUCINATION PREVENTION ──────────────────────────────────────────────────
- If the data sample is empty (0 rows), say clearly: no matching records were found.
  Do not guess why. Do not suggest what the answer might be.
- If a statistic is missing or marked N/A, do not invent a substitute.
- If the data is ambiguous or incomplete, say so — do not fill the gap.
- Never say "typically", "usually", "generally", or "I believe" — only state
  what the data explicitly shows.

── TEMPORAL GUARDRAILS ───────────────────────────────────────────────────────
- You do not know what today's date is. Never state it.
- You do not know the current time or day of week. Never state it.
- If date values appear in the data, you may reference them as-is
  (e.g. "the most recent record is from 2024-11-01") but never
  calculate how long ago that was or what "today" minus that date equals.
- Never say "as of today", "currently", or "right now" unless the data
  explicitly contains a real-time timestamp.

── RESPONSE STYLE ────────────────────────────────────────────────────────────
- Conversational, professional, insight-focused. No SQL, no table names,
  no column names, no technical jargon.
- Scale length to complexity:
    * Single value answer (a count, a name) -> 1 sentence.
    * Small result set (2-10 rows) -> 2-3 sentences highlighting key observations.
    * Aggregated/grouped data -> 3-5 sentences covering pattern, top/bottom, anomaly.
    * Large result or trend data -> up to 6 sentences with a clear narrative arc.
- Always lead with the direct answer to the question, then add context.
- Format numbers naturally: commas for thousands (1,200 not 1200),
  currency symbols where relevant, percentages where applicable.
- If a ranking or comparison is present, call out the leader and the gap.
- If a trend is present in the statistics (increasing/decreasing/stable), name it.
- Do not start your response with "Based on", "According to", or "The data shows".
  Start directly with the insight.
- If the SQL used TOP N or a WHERE filter, frame as "the top results" or
  "matching records" — never imply these are all records that exist.

── EDGE CASES ────────────────────────────────────────────────────────────────
- 0 rows returned -> "No records were found matching your criteria. You may want
  to refine your search or check the filters."
- Question asks for date/time -> "I don't have access to the current date or time.
  I can only report on dates present in your data."
- Question is general knowledge -> redirect to scope boundary response above.
- Single numeric result -> answer in one sentence, no padding.
- Data has NULLs in key columns -> mention it if relevant
  (e.g. "some records have no value recorded for this field").
"""),
    ("human", """User question: {user_query}

SQL used: {sql_query}

Pre-analyzed context:
- Total matching records: {total_rows}
- Column statistics & breakdowns: {column_stats}

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
    enriched_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a conversational response based on query results using LangChain.
    enriched_result can be either a raw query result or an enriched profile.
    """
    logger.info("Generating natural language response via LangChain.")

    try:
        # Determine if input is already semantically enriched or a raw query result
        if "total_rows" in enriched_result:
            # Enriched profile format from result_enricher.py
            total_rows = enriched_result["total_rows"]
            column_stats = enriched_result.get("column_stats", {})
            data_sample = enriched_result.get("data_sample", [])
        else:
            # Raw result format
            total_rows = enriched_result.get("row_count", len(enriched_result.get("rows", [])))
            column_stats = "N/A"
            data_sample = enriched_result.get("rows", [])[:5]

        response_text = _nl_chain.invoke({
            "user_query": user_query,
            "sql_query": sql_query,
            "total_rows": total_rows,
            "column_stats": json.dumps(column_stats, indent=2, default=str),
            "data_sample": json.dumps(data_sample, indent=2, default=str)
        })

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
