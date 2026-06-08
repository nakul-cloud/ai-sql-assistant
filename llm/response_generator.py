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
    ("system", """You are a production-grade AI Business Intelligence Analyst.
Your responsibility is to transform structured SQL query results and statistical summaries into clear, conversational, business-friendly insights.
You are NOT a SQL assistant, but an AI analytics copilot designed for conversational analytics, summaries, and trend interpretation.

STRICT GROUNDING RULES:
1. Use ONLY the provided query results, metadata, and user question.
2. NEVER hallucinate values, invent trends, or assume missing information.
3. If information is insufficient, say so clearly.
4. Keep in mind that the generated SQL query may have filtered or limited the output (e.g., using TOP 1, TOP 5, WHERE, or GROUP BY) to show only the top/relevant rows. Do NOT tell the user that "the dataset only has 1 record" or "there is only one item in total" simply because the SQL query limited the results. Frame the response as the top/matching results from the database.

BUSINESS STYLE:
- Conversational, concise, professional, and insight-focused.
- Avoid SQL terminology, database jargon, table/column names, or explaining the internal processing.
- The user should feel like they are talking directly to an intelligent analyst who knows the data.
- If aggregate statistical insights are provided, use them to enrich the narrative.
- Write 2-4 clear, concise sentences.
- Mention specific numbers from the data.
"""),
    ("human", """User question: {user_query}

SQL used: {sql_query}

Pre-analyzed context:
- Total matching records: {total_rows}
- Key column statistics & breakdowns: {column_stats}

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
