"""
workflow/process_query.py
─────────────────────────
Unified production-grade execution pipeline for the Enterprise AI SQL Assistant.
Orchestrates: Router -> Cache (Check) -> Retriever -> Schema Builder -> SQL Gen -> Executor -> NL Response -> Cache (Store).
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv

from retrieval.query_router import route_query
from retrieval.query_cache import check_query_cache, store_in_query_cache
from retrieval.table_retriever import retrieve_relevant_tables
from analysis.schema_context import generate_schema_context
from llm.query_ai import generate_sql_query
from workflow.query_executor import execute_sql_query
from llm.response_generator import generate_natural_language_response
from analysis.result_enricher import enrich_sql_result
from database.schema_manager import fetch_database_metadata
from indexing.semantic_description import generate_content_with_retry

load_dotenv()
logger = logging.getLogger(__name__)


def handle_chat_query(user_query: str) -> Dict[str, Any]:
    """
    Handle general chat and conversational queries using Groq.
    """
    logger.info(f"Handling CHAT query: '{user_query}'")
    prompt = f"""You are a helpful and professional Enterprise AI SQL Analytics Assistant.
Answer the user's conversational message politely, briefly, and guide them on how they can query the database.
Mention that you can answer analytical queries about employees, departments, and sales data.

User Message: "{user_query}"
Response:"""
    try:
        client = None
        response = generate_content_with_retry(
            client,
            model=None,
            contents=prompt,
        )
        return {
            "success": True,
            "intent": "CHAT",
            "nl_response": response.text.strip(),
            "cache_hit": False
        }
    except Exception as e:
        logger.exception("Failed to generate chat response.")
        return {
            "success": False,
            "intent": "CHAT",
            "error": str(e),
            "nl_response": "Hello! I am your AI SQL Analytics Assistant. How can I help you query the database today?"
        }


def handle_schema_info_query(user_query: str) -> Dict[str, Any]:
    """
    Handle queries asking about the database tables, schemas, or structure.
    """
    logger.info(f"Handling SCHEMA_INFO query: '{user_query}'")
    try:
        metadata_list = fetch_database_metadata()
        
        # Format a summary of the schemas
        meta_lines = []
        for meta in metadata_list:
            cols = [col["name"] for col in meta["columns"]]
            meta_lines.append(
                f"- Table: {meta['table_name']}\n"
                f"  Columns: {', '.join(cols)}\n"
                f"  Approximate Row Count: {meta['row_count']}"
            )
        formatted_meta = "\n".join(meta_lines)
        
        prompt = f"""You are a database structure expert and documentation assistant.
Answer the user's question about the database structure, tables, columns, or schemas using the metadata below.
Be concise, accurate, and direct.

Database Metadata:
------------------
{formatted_meta}

User Question: "{user_query}"
Response:"""

        client = None
        response = generate_content_with_retry(
            client,
            model=None,
            contents=prompt,
        )
        return {
            "success": True,
            "intent": "SCHEMA_INFO",
            "nl_response": response.text.strip(),
            "cache_hit": False,
            "metadata_summary": formatted_meta
        }
    except Exception as e:
        logger.exception("Failed to generate schema info response.")
        return {
            "success": False,
            "intent": "SCHEMA_INFO",
            "error": str(e),
            "nl_response": "I'm sorry, I could not retrieve the database schema information at this time."
        }


def process_user_query(user_query: str, focus_tables: list = None) -> Dict[str, Any]:
    """
    Unified query processing pipeline.
    Routes the query and performs caching, retrieval, generation, and execution as appropriate.
    """
    logger.info(f"Processing query: '{user_query}'")
    
    # 1. Determine Intent
    intent = route_query(user_query)
    logger.info(f"Routed intent: {intent}")
    
    if intent == "CHAT":
        return handle_chat_query(user_query)
        
    elif intent == "SCHEMA_INFO":
        return handle_schema_info_query(user_query)
        
    # 2. SQL Query Execution Pipeline
    # Check cache first
    cached_result = check_query_cache(user_query)
    if cached_result:
        return {
            "success": True,
            "intent": "SQL_QUERY",
            "cache_hit": True,
            "generated_sql": "Served from Cache",
            "query_result": {
                "columns": cached_result.get("columns", []),
                "rows": cached_result.get("rows", []),
                "row_count": len(cached_result.get("rows", []))
            },
            "nl_response": cached_result.get("nl_response")
        }
        
    # Cache Miss -> Hybrid Retrieval (only retrieve if not explicitly provided)
    if not focus_tables:
        focus_tables = retrieve_relevant_tables(user_query, top_k=2)
    logger.info(f"Retrieved tables for context: {focus_tables}")
    
    # Generate Schema Context
    schema_context = generate_schema_context(user_query, focus_tables=focus_tables)
    
    # Generate SQL
    sql_res = generate_sql_query(user_query, schema_context)
    if not sql_res["success"]:
        return {
            "success": False,
            "intent": "SQL_QUERY",
            "error_type": "SQL_GENERATION_FAILED",
            "error": sql_res.get("error", "Failed to generate SQL query."),
            "nl_response": "I was unable to generate a valid SQL query for your request."
        }
        
    sql_query = sql_res["sql_query"]
    
    # Execute SQL Query
    exec_res = execute_sql_query(sql_query)
    if not exec_res["success"]:
        return {
            "success": False,
            "intent": "SQL_QUERY",
            "error_type": exec_res.get("error_type", "SQL_EXECUTION_FAILED"),
            "error": exec_res.get("error", "Database execution error."),
            "generated_sql": sql_query,
            "nl_response": "The generated SQL query failed to execute against the database."
        }
        
    query_result = exec_res["result"]
    
    # Enrich the result for the LLM
    enriched_result = enrich_sql_result(query_result)
    
    # Generate Natural Language Insights using the enriched profile
    nl_res = generate_natural_language_response(user_query, sql_query, enriched_result)
    nl_response = nl_res["response_text"]
    
    # Store result in cache
    store_in_query_cache(user_query, nl_response, query_result)
    
    return {
        "success": True,
        "intent": "SQL_QUERY",
        "cache_hit": False,
        "generated_sql": sql_query,
        "query_result": query_result,
        "nl_response": nl_response
    }


if __name__ == "__main__":
    print("\n--- Testing Integrated Query Pipeline ---")
    
    # Clear cache for isolated test
    from retrieval.query_cache import clear_query_cache
    clear_query_cache()
    
    queries = [
        ("Hello there!", "CHAT"),
        ("What tables exist in our database?", "SCHEMA_INFO"),
        ("Show departments and their managers", "SQL_QUERY"),
        ("Show departments and their managers", "SQL_QUERY (Cache Hit Check)"),
    ]
    
    for q_text, expected_type in queries:
        print(f"\n=======================================================")
        print(f"Query: '{q_text}' ({expected_type})")
        print(f"=======================================================")
        res = process_user_query(q_text)
        print(f"Success    : {res.get('success')}")
        print(f"Intent     : {res.get('intent')}")
        print(f"Cache Hit  : {res.get('cache_hit')}")
        if "generated_sql" in res:
            print(f"SQL        : {res['generated_sql']}")
        if "nl_response" in res:
            print(f"Answer     : {res['nl_response']}")
        if "query_result" in res:
            print(f"Rows count : {res['query_result'].get('row_count')}")
            
    clear_query_cache()
