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
    ("system", """You are an expert Microsoft SQL Server (T-SQL) engineer.
Generate an accurate, optimized, production-grade SELECT query.

Rules:
- Generate ONLY valid Microsoft SQL Server (T-SQL) syntax.
- Use table and column names exactly as provided in the schema.
- Only generate SELECT queries. Never generate destructive queries (DROP, DELETE, TRUNCATE, ALTER, UPDATE, INSERT, EXEC).
- Return ONLY the SQL query, no explanation, no markdown fences.
- Use proper table aliases.
- Use TOP instead of LIMIT for limiting rows.
- Avoid SELECT *. Explicitly list the columns required.
- When filtering by specific text/names, use LIKE '%value%' and search across relevant string columns.
- If combining TOP and DISTINCT, you MUST write DISTINCT before TOP (e.g. SELECT DISTINCT TOP 10 ... instead of SELECT TOP 10 DISTINCT ...).
- If the question cannot be answered from the schema, say: CANNOT_GENERATE
"""),
    ("human", """Database Schema Context:
------------------------
{schema_context}

User Question:
--------------
{user_query}

Generate a valid Microsoft SQL Server query:""")
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
                "error": "I couldn't find relevant data to answer that question."
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
