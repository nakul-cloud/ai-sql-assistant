"""
llm/response_generator.py
─────────────────────────
Generates a conversational natural language response summarizing SQL execution results
using the new Google GenAI SDK.
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from indexing.semantic_description import _get_gemini_client, GEMINI_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

RESPONSE_PROMPT = """You are a production-grade AI Business Intelligence Analyst.

Your responsibility is to transform structured SQL query results into clear, conversational, business-friendly insights.
You are NOT a SQL assistant, but an AI analytics copilot designed for conversational analytics, summaries, and trend interpretation.

STRICT GROUNDING RULES:
1. Use ONLY the provided query results, metadata, and user question.
2. NEVER hallucinate values, invent trends, or assume missing information.
3. If information is insufficient, say so clearly.

BUSINESS STYLE:
- Conversational, concise, professional, and insight-focused.
- Avoid SQL terminology, database jargon, table/column names, or explaining the internal processing.
- The user should feel like they are talking directly to an intelligent analyst who knows the data.

USER QUESTION:
--------------
{user_query}

GENERATED SQL:
--------------
{sql_query}

QUERY RESULT:
-------------
{query_result}

FINAL TASK:
-----------
Generate a concise, business-friendly conversational insight response."""


def generate_natural_language_response(
    user_query: str,
    sql_query: str,
    query_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a conversational response based on query results.
    """
    logger.info("Generating natural language response.")

    try:
        rows = query_result.get("rows", [])
        row_count = query_result.get("row_count", 0)

        # Truncate to top 20 rows for prompt representation to prevent token bloat
        if len(rows) > 20:
            formatted_result = f"Showing top 20 of {row_count} rows:\n{rows[:20]}"
        else:
            formatted_result = f"Showing all {row_count} rows:\n{rows}"

        client = _get_gemini_client()
        prompt = RESPONSE_PROMPT.format(
            user_query=user_query,
            sql_query=sql_query,
            query_result=formatted_result
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        nl_response = response.text.strip()
        logger.info("Natural language response generated successfully.")

        return {
            "success": True,
            "response_text": nl_response
        }

    except Exception as error:
        logger.exception("Failed to generate natural language response.")
        fallback_msg = f"Query executed successfully. Returned {query_result.get('row_count', 0)} rows."
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
        query_result=mock_result
    )
    print("Result:")
    print(res["response_text"])
