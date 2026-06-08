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


def contextualize_query(user_query: str, chat_history: list) -> str:
    """
    Contextualize the user's query based on recent chat history to handle follow-up questions.
    """
    if not chat_history:
        return user_query

    # Extract user/assistant turns
    turns = []
    # Only use the last 4 messages to keep it fast and prevent token bloat
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        turns.append(f"{role}: {content}")
        
    history_str = "\n".join(turns)

    prompt = f"""You are a conversational database query contextualization assistant.
Given the following conversation history and the user's latest follow-up question, rephrase the question to be a self-contained, standalone query (in natural language) that contains all necessary context from the history.
If the latest question is already completely standalone and needs no context, return it exactly as is.
DO NOT answer the question. DO NOT write SQL. Return ONLY the self-contained question text.

Conversation History:
{history_str}

Latest Follow-up Question: "{user_query}"
Standalone Question:"""

    try:
        from llm.llm_client import generate_text
        rephrased = generate_text(prompt)
        rephrased_clean = rephrased.strip().strip('"\'')
        if rephrased_clean:
            logger.info(f"Rephrased user query: '{user_query}' -> '{rephrased_clean}'")
            return rephrased_clean
    except Exception as e:
        logger.warning(f"Failed to contextualize query: {e}")
    return user_query


def process_user_query(user_query: str, focus_tables: list = None, chat_history: list = None) -> Dict[str, Any]:
    """
    Unified query processing pipeline.
    Routes the query and performs caching, retrieval, generation, and execution as appropriate.
    Supports conversational query contextualization if chat_history is provided.
    """
    # 0. Contextualize user query using chat history
    active_query = user_query
    if chat_history:
        active_query = contextualize_query(user_query, chat_history)
        
    logger.info(f"Processing query: '{active_query}' (Original: '{user_query}')")
    
    # 1. Determine Intent
    intent = route_query(active_query)
    logger.info(f"Routed intent: {intent}")
    
    if intent == "CHAT":
        res = handle_chat_query(active_query)
        res["rephrased_query"] = active_query
        return res
        
    elif intent == "SCHEMA_INFO":
        res = handle_schema_info_query(active_query)
        res["rephrased_query"] = active_query
        return res
        
    # 2. SQL Query Execution Pipeline
    # Check cache first
    cached_result = check_query_cache(active_query)
    if cached_result:
        return {
            "success": True,
            "intent": "SQL_QUERY",
            "cache_hit": True,
            "generated_sql": "Served from Cache",
            "rephrased_query": active_query,
            "query_result": {
                "columns": cached_result.get("columns", []),
                "rows": cached_result.get("rows", []),
                "row_count": len(cached_result.get("rows", []))
            },
            "nl_response": cached_result.get("nl_response")
        }
        
    # Cache Miss -> Hybrid Retrieval (only retrieve if not explicitly provided)
    if not focus_tables:
        focus_tables = retrieve_relevant_tables(active_query, top_k=2)
    logger.info(f"Retrieved tables for context: {focus_tables}")
    
    # Generate Schema Context
    schema_context = generate_schema_context(active_query, focus_tables=focus_tables)
    
    # Generate SQL
    sql_res = generate_sql_query(active_query, schema_context)
    if not sql_res["success"]:
        return {
            "success": False,
            "intent": "SQL_QUERY",
            "error_type": "SQL_GENERATION_FAILED",
            "rephrased_query": active_query,
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
            "rephrased_query": active_query,
            "nl_response": "The generated SQL query failed to execute against the database."
        }
        
    query_result = exec_res["result"]
    
    # Enrich the result for the LLM
    enriched_result = enrich_sql_result(query_result)
    
    # Generate Natural Language Insights using the enriched profile
    nl_res = generate_natural_language_response(active_query, sql_query, enriched_result)
    nl_response = nl_res["response_text"]
    
    # Store result in cache
    store_in_query_cache(active_query, nl_response, query_result)
    
    return {
        "success": True,
        "intent": "SQL_QUERY",
        "cache_hit": False,
        "generated_sql": sql_query,
        "query_result": query_result,
        "rephrased_query": active_query,
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
