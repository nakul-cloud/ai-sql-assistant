"""
llm/query_ai.py
───────────────
Production-grade AI SQL generation engine using LangChain LCEL.
"""

import os
import re
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

load_dotenv()
logger = logging.getLogger(__name__)

# Prompt for SQL Generation
SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Microsoft SQL Server (T-SQL) engineer working on a business intelligence system.
Your job is to generate a single, accurate, production-safe SELECT query from a natural language question and a database schema.

── SCOPE BOUNDARY ────────────────────────────────────────────────────────────
You operate ONLY within the provided database schema. You have no knowledge of:
- Today's date, current time, day of week, or any real-world temporal context.
- External facts, current events, general knowledge, or anything outside the schema.
- Data that is not explicitly present in the schema provided.

If the question requires knowledge outside the schema — including current date/time,
general facts, opinions, or anything not in the provided tables — return exactly:
OUT_OF_SCOPE

If the question cannot be answered from the schema columns/tables — return exactly:
CANNOT_GENERATE

Never guess, never hallucinate column names, table names, or data values.
If a column or table does not exist in the schema, do not invent it.

── OUTPUT RULES ──────────────────────────────────────────────────────────────
- Return ONLY the raw T-SQL query. No markdown, no code fences, no explanation.
- One output only: either a valid SQL query, CANNOT_GENERATE, or OUT_OF_SCOPE.

── SAFETY RULES ──────────────────────────────────────────────────────────────
- Only SELECT statements. Never generate: DROP, DELETE, TRUNCATE, ALTER, UPDATE,
  INSERT, MERGE, EXEC, EXECUTE, CREATE, GRANT, REVOKE, DENY, SHUTDOWN, BACKUP, RESTORE.
- No subquery injection, no dynamic SQL, no system stored procedures.
- One statement only. No semicolons mid-query.

── T-SQL SYNTAX RULES ────────────────────────────────────────────────────────
- Use TOP instead of LIMIT.
- DISTINCT must come before TOP: SELECT DISTINCT TOP 10 ... NOT SELECT TOP 10 DISTINCT ...
- Never use SELECT *. Always list required columns explicitly.
- Use proper table aliases (e.g. e for employees, o for orders).
- Wrap column/table names with spaces or reserved words in square brackets: [column name].
- For date filtering, use CAST or CONVERT to ensure correct type comparison.
- For NULL-safe comparisons, use IS NULL / IS NOT NULL. Never use = NULL.
- For current date, use GETDATE() — only when the schema has date columns AND
  the user's question explicitly involves "today", "this week", "this month" etc.
  Never assume a date range unless the user asked for one.

── QUERY DESIGN RULES ────────────────────────────────────────────────────────
- Text/name searches: use LIKE '%value%' across all relevant string columns.
- Aggregations: use GROUP BY with all non-aggregated columns in SELECT.
- Sorting: always add ORDER BY for top-N queries or when the question implies ranking.
- Multi-table queries: use explicit JOIN with ON conditions derived from schema PKs/FKs.
  Prefer INNER JOIN unless the question implies optional/missing records (then LEFT JOIN).
- If the question asks for a count, use COUNT(*) or COUNT(column) as appropriate.
- If the question asks for a total or sum, verify the column is numeric before using SUM().
- If schema has multiple tables that could answer the question, pick the most relevant one.
  If a JOIN is needed, include it.

── EDGE CASES ────────────────────────────────────────────────────────────────
- "What is today's date?" → OUT_OF_SCOPE
- "What time is it?" → OUT_OF_SCOPE
- "What day is it?" → OUT_OF_SCOPE
- "Who is the CEO of X company?" → OUT_OF_SCOPE
- "Tell me a joke / explain a concept / summarize news" → OUT_OF_SCOPE
- "Show data from a table not in the schema" → CANNOT_GENERATE
- Vague questions with no schema match (e.g. "show everything") → generate a
  reasonable bounded query (TOP 100) on the most relevant table, do not return
  CANNOT_GENERATE for ambiguous but answerable questions.
"""),
    ("human", """Database Schema:
─────────────────
{schema_context}

User Question:
──────────────
{user_query}

T-SQL Query:""")
])


def clean_sql_query(sql_text: str) -> str:
    """
    Cleans LLM SQL response by removing markdown blocks and whitespace.
    Also auto-corrects SQL Server specific syntax errors like TOP DISTINCT.
    """
    if not sql_text:
        return ""

    cleaned = sql_text.strip()
    # Remove markdown code blocks if present
    cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    
    # Auto-correct TOP DISTINCT syntax error for MS SQL Server
    # e.g., "SELECT TOP 100 DISTINCT col" -> "SELECT DISTINCT TOP 100 col"
    cleaned = re.sub(
        r"\bSELECT\s+TOP\s+(\d+|\(\d+\))\s+DISTINCT\b",
        r"SELECT DISTINCT TOP \1",
        cleaned,
        flags=re.IGNORECASE
    )
    return cleaned.strip()


def validate_sql_response(sql_query: str) -> None:
    """
    Validates that the generated SQL is safe and is a SELECT statement.
    """
    forbidden_keywords = {
        "DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT",
        "MERGE", "EXEC", "EXECUTE", "CREATE", "GRANT", "REVOKE",
        "DENY", "SHUTDOWN", "BACKUP", "RESTORE"
    }
    
    # Strip comments first to avoid false positives or bypasses
    clean_query = re.sub(r"--.*?\n", "", sql_query)
    clean_query = re.sub(r"/\*.*?\*/", "", clean_query, flags=re.DOTALL)
    
    upper_query = clean_query.upper().strip()
    
    # Check for multiple statements
    if ";" in upper_query:
        statements = [s.strip() for s in upper_query.split(";") if s.strip()]
        if len(statements) > 1:
            raise ValueError("Multiple SQL statements are not allowed.")
 
    for keyword in forbidden_keywords:
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, upper_query):
            raise ValueError(f"Unsafe SQL detected: Forbidden keyword '{keyword}' used.")

    # Check for forbidden functions
    forbidden_functions = {"XP_CMDSHELL", "SP_EXECUTE", "SP_EXECUTESQL", "OPENROWSET", "OPENDATASOURCE"}
    for func in forbidden_functions:
        pattern = rf"\b{func}\b"
        if re.search(pattern, upper_query):
            raise ValueError(f"Unsafe SQL detected: Forbidden function '{func}' used.")

    if not upper_query.startswith("SELECT"):
        raise ValueError("Generated SQL is not a SELECT query.")


def build_sql_chain():
    llm = get_llm(temperature=0.0)
    return SQL_PROMPT | llm | StrOutputParser()


_sql_chain = build_sql_chain()


def generate_sql_query(user_query: str, schema_context: str) -> Dict[str, Any]:
    """
    Main entry point for generating SQL from user query and schema context.
    """
    logger.info(f"Generating SQL for query: '{user_query}'")
    
    try:
        raw_output = _sql_chain.invoke({
            "user_query": user_query,
            "schema_context": schema_context
        })
        raw_output = raw_output.strip()
        
        if raw_output == "CANNOT_GENERATE":
            return {
                "success": False,
                "user_query": user_query,
                "error": "CANNOT_GENERATE"
            }

        elif raw_output == "OUT_OF_SCOPE":
            return {
                "success": False,
                "user_query": user_query,
                "error": "OUT_OF_SCOPE"
            }
            
        cleaned_sql = clean_sql_query(raw_output)
        
        # Validate the generated query
        validate_sql_response(cleaned_sql)
        
        logger.info("SQL generated and validated successfully.")
        return {
            "success": True,
            "user_query": user_query,
            "sql_query": cleaned_sql,
            "schema_context": schema_context,
            "raw_response": raw_output
        }
    except Exception as error:
        logger.exception("SQL generation or validation failed.")
        return {
            "success": False,
            "user_query": user_query,
            "error": str(error)
        }


if __name__ == "__main__":
    import sys
    print("\n--- Testing SQL Generator ---")
    test_context = """
Table: dbo.csv_employees
Row Count: 5
Columns:
  - employee_id (bigint, NULL)
  - employee_name (varchar(MAX), NULL)
  - salary (bigint, NULL)
"""
    test_query = "Show employee name and salary"
    result = generate_sql_query(test_query, test_context)
    if result["success"]:
        print("Success!")
        print(f"Generated SQL: {result['sql_query']}")
    else:
        print(f"Failed: {result['error']}")
        sys.exit(1)
