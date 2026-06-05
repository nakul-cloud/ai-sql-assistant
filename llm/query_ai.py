"""
llm/query_ai.py
───────────────
Production-grade AI SQL generation engine using the Google GenAI SDK.
"""

import os
import re
import logging
from typing import Dict, Any

from google import genai
from dotenv import load_dotenv

from indexing.semantic_description import _get_gemini_client, GEMINI_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

# System instructions/prompt for SQL Generation
SQL_GENERATION_PROMPT = """You are an expert Microsoft SQL Server (T-SQL) engineer.

Your task is to generate accurate, optimized, production-grade SELECT queries.

STRICT RULES:
1. Generate ONLY valid Microsoft SQL Server (T-SQL) syntax.
2. Never hallucinate table names or column names. Use ONLY the provided database schema context.
3. Only generate SELECT queries. Never generate destructive queries (DROP, DELETE, TRUNCATE, ALTER, UPDATE, INSERT, EXEC).
4. Never explain the SQL or wrap it in triple backticks. Return ONLY executable SQL.
5. Use proper table aliases.
6. Use TOP instead of LIMIT for limiting rows.
7. Avoid SELECT *. Explicitly list the columns required.
8. When filtering by specific text/names, use LIKE '%value%' and search across relevant string columns.

Database Schema Context:
------------------------
{schema_context}

User Question:
--------------
{user_query}

Generate a valid Microsoft SQL Server query:"""


def clean_sql_query(sql_text: str) -> str:
    """
    Cleans LLM SQL response by removing markdown blocks and whitespace.
    """
    if not sql_text:
        return ""

    cleaned = sql_text.strip()
    # Remove markdown code blocks if present
    cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
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


def generate_sql_query(user_query: str, schema_context: str) -> Dict[str, Any]:
    """
    Main entry point for generating SQL from user query and schema context.
    """
    logger.info(f"Generating SQL for query: '{user_query}'")
    
    try:
        client = _get_gemini_client()
        prompt = SQL_GENERATION_PROMPT.format(
            schema_context=schema_context,
            user_query=user_query
        )
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        
        raw_output = response.text.strip()
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
