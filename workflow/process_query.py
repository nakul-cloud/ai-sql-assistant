import os
# Warm up PyTorch & embedding model BEFORE loading SQL Server drivers / pyodbc
# to prevent the Windows OpenMP/pyodbc thread collision crash.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from indexing.embedder import embed_text
_ = embed_text("warmup")

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval.query_router import route_query
from retrieval.query_cache import check_query_cache, store_in_query_cache
from retrieval.table_retriever import retrieve_relevant_tables
from analysis.schema_context import generate_schema_context
from llm.query_ai import generate_sql_query
from workflow.query_executor import execute_sql_query
from llm.response_generator import generate_natural_language_response
from analysis.result_enricher import enrich_sql_result
from database.schema_manager import fetch_database_metadata
from llm.llm_client import get_llm

load_dotenv()
logger = logging.getLogger(__name__)

# ── Hardcoded refusals (never reach LLM) ─────────────────────────────────────
TEMPORAL_REFUSAL = (
    "I don't have access to the current date or time — "
    "I can only report on dates and years present in your data. "
    "You can ask things like: 'What is the earliest year in the dataset?' "
    "or 'How many records are from 2015?'"
)

OUT_OF_SCOPE_REFUSAL = (
    "I can only answer questions about your business data. "
    "Please ask me something about the records in your database."
)

GENERAL_KNOWLEDGE_REFUSAL = (
    "I can only answer questions about the data in your database. "
    "For general knowledge questions like current events, people, or politics, "
    "please use a search engine or general assistant."
)

# ── LangChain CHAT chain ──────────────────────────────────────────────────────
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and professional Enterprise AI SQL Analytics Assistant.
Answer the user's conversational message politely, briefly, and guide them on how they can query the database.
You currently have access to data regarding: {available_tables}.
Briefly mention some of these topics to let the user know what they can ask about."""),
    ("human", "{user_query}")
])

_chat_chain = CHAT_PROMPT | get_llm(temperature=0.7) | StrOutputParser()


def handle_chat_query(user_query: str, stream: bool = False) -> Dict[str, Any]:
    """
    Handle general chat and conversational queries using LangChain.
    """
    logger.info(f"Handling CHAT query via LangChain: '{user_query}'")
    try:
        # Dynamically fetch available tables so the greeting is always accurate
        try:
            metadata_list = fetch_database_metadata()
            # Extract clean table names (e.g., 'dbo.csv_employees' -> 'employees')
            tables = [meta["table_name"].split('.')[-1].replace('csv_', '').replace('_', ' ') for meta in metadata_list]
            available_tables = ", ".join(tables) if tables else "various analytical datasets"
        except Exception as meta_err:
            logger.warning(f"Failed to fetch metadata for chat prompt: {meta_err}")
            available_tables = "various analytical datasets"

        if stream:
            token_stream = _chat_chain.stream({
                "user_query": user_query,
                "available_tables": available_tables
            })
            return {
                "success": True,
                "intent": "CHAT",
                "nl_response": token_stream,
                "cache_hit": False
            }

        response_text = _chat_chain.invoke({
            "user_query": user_query,
            "available_tables": available_tables
        })
        return {
            "success": True,
            "intent": "CHAT",
            "nl_response": response_text.strip(),
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


# ── LangChain SCHEMA_INFO chain ──────────────────────────────────────────────
SCHEMA_INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a database structure expert and documentation assistant.
Answer the user's question about the database structure, tables, columns, or schemas using the metadata provided.
Be concise, accurate, and direct."""),
    ("human", """Database Metadata:
------------------
{formatted_meta}

User Question: "{user_query}"
Response:""")
])

_schema_info_chain = SCHEMA_INFO_PROMPT | get_llm(temperature=0.2) | StrOutputParser()


def handle_schema_info_query(user_query: str, stream: bool = False) -> Dict[str, Any]:
    """
    Handle queries asking about the database tables, schemas, or structure.
    """
    logger.info(f"Handling SCHEMA_INFO query via LangChain: '{user_query}'")
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

        if stream:
            token_stream = _schema_info_chain.stream({
                "formatted_meta": formatted_meta,
                "user_query": user_query
            })
            return {
                "success": True,
                "intent": "SCHEMA_INFO",
                "nl_response": token_stream,
                "cache_hit": False,
                "metadata_summary": formatted_meta
            }

        response_text = _schema_info_chain.invoke({
            "formatted_meta": formatted_meta,
            "user_query": user_query
        })
        
        return {
            "success": True,
            "intent": "SCHEMA_INFO",
            "nl_response": response_text.strip(),
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


def handle_describe_intent(user_query: str, focus_tables: list = None, stream: bool = False) -> Dict[str, Any]:
    """
    Handles 'explain this dataset / what is this table about' type questions.
    Pulls schema metadata and generates a plain-English description.
    No SQL execution needed.
    """
    logger.info(f"Handling DESCRIBE query: '{user_query}'")
    try:
        from analysis.schema_context import get_all_table_summaries, generate_schema_context
        from llm.describe_generator import generate_dataset_description, generate_database_overview
        from retrieval.table_retriever import retrieve_relevant_tables

        # 0. Check if the user is asking for a general overview or list of tables
        q_lower = user_query.lower()
        is_general_query = any(w in q_lower for w in [
            "database", "all datasets", "all tables", "what datasets", "what tables",
            "available datasets", "available tables", "list of datasets", "list of tables",
            "show all datasets", "show all tables", "what data do we have", "what is in the database",
            "explain the database", "overview of the database", "summarize the database", "show me all"
        ])

        # 1. Determine target tables semantically if not explicitly provided and NOT a general query
        tables = focus_tables
        if not tables and not is_general_query:
            tables = retrieve_relevant_tables(user_query, top_k=1)

        # 2. Get the schema context for the identified tables
        if tables:
            logger.info(f"DESCRIBE intent resolved to tables: {tables}")
            relevant_schema = generate_schema_context(user_query, focus_tables=tables)
            description = generate_dataset_description(user_query, relevant_schema, stream=stream)
        else:
            logger.info("DESCRIBE intent fell back to check general overview.")
            if is_general_query:
                relevant_schema = get_all_table_summaries()
                description = generate_database_overview(user_query, relevant_schema, stream=stream)
            else:
                description = "I couldn't identify which specific dataset you want me to explain. Could you please specify one of the available tables? (e.g., Sales, Employees, Departments, Customer Support Tickets, Corporate AI Adoption, AI Impact on Jobs, or Placement Data)."

        return {
            "success": True,
            "intent": "DESCRIBE",
            "cache_hit": False,
            "nl_response": description,
            "generated_sql": None,
            "query_result": {}
        }
    except Exception as e:
        logger.exception("Failed to generate dataset description.")
        return {
            "success": False,
            "intent": "DESCRIBE",
            "error": str(e),
            "nl_response": "I'm sorry, I could not generate a description of the dataset at this time."
        }


# ── LangChain contextualization chain ─────────────────────────────────────────
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a conversational database query contextualization assistant.
Given the conversation history and the user's latest follow-up question, rephrase the question to be a self-contained, standalone query (in natural language) that contains all necessary context from the history.
If the latest question is already completely standalone and needs no context, return it exactly as is.
DO NOT answer the question. DO NOT write SQL. Return ONLY the self-contained question text."""),
    ("human", """Conversation History:
{history_str}

Latest Follow-up Question: "{user_query}"
Standalone Question:""")
])

_contextualize_chain = CONTEXTUALIZE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()


def needs_contextualization(query: str) -> bool:
    """
    Dynamically determines if a follow-up query is context-dependent or self-contained.
    Uses regex checks for pronouns, relative terms, and comparative keywords.
    """
    import re
    q = query.strip().lower()
    
    # 1. Extremely short queries (<= 3 words) are highly likely to be fragments
    words = q.split()
    if len(words) <= 3:
        return True
        
    # 2. Context-dependent pronouns/determiners/adverbs/comparatives
    context_indicators = [
        r"\b(it|they|them|their|its|he|she|him|her|his|this|that|these|those)\b",
        r"\b(here|there|then|so|previous|above|below|former|latter|another|other)\b",
        r"\b(compare|comparison|vs|versus|difference|diff)\b",
        r"\b(higher|lower|more|less|better|worse|earliest|latest|highest|lowest|most|least)\b",
        r"\b(also|and|but|or|what\s+about|how\s+about|what\s+else|anything\s+else)\b"
    ]
    
    for pattern in context_indicators:
        if re.search(pattern, q):
            return True
            
    # 3. If it explicitly mentions one of the known table names, it is highly likely to be self-contained
    try:
        from retrieval.query_router import _load_schema_terms
        terms = _load_schema_terms()
        for table in terms.get("table_names", []):
            if table in q:
                return False
    except Exception:
        pass
        
    return False


def contextualize_query(user_query: str, chat_history: list) -> str:
    """
    Contextualize the user's query based on recent chat history to handle follow-up questions.
    """
    if not chat_history:
        return user_query

    # Dynamically check if query actually needs history context
    if not needs_contextualization(user_query):
        logger.info(f"Query '{user_query}' determined to be self-contained. Bypassing LLM contextualization.")
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

    try:
        rephrased = _contextualize_chain.invoke({
            "history_str": history_str,
            "user_query": user_query
        })
        rephrased_clean = rephrased.strip().strip('"\'')
        if rephrased_clean:
            logger.info(f"Rephrased user query: '{user_query}' -> '{rephrased_clean}'")
            return rephrased_clean
    except Exception as e:
        logger.warning(f"Failed to contextualize query: {e}")
    return user_query


def process_user_query(
    user_query: str,
    focus_tables: list = None,
    chat_history: list = None,
    stream: bool = False
) -> Dict[str, Any]:
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
        res = handle_chat_query(active_query, stream=stream)
        res["rephrased_query"] = active_query
        return res
        
    elif intent == "SCHEMA_INFO":
        res = handle_schema_info_query(active_query, stream=stream)
        res["rephrased_query"] = active_query
        return res

    elif intent == "DESCRIBE":
        res = handle_describe_intent(active_query, focus_tables=focus_tables, stream=stream)
        res["rephrased_query"] = active_query
        return res

    elif intent == "TEMPORAL":
        logger.info("Temporal question intercepted — returning hardcoded refusal.")
        return {
            "success": True,
            "intent": "TEMPORAL",
            "cache_hit": False,
            "rephrased_query": active_query,
            "generated_sql": None,
            "query_result": {},
            "nl_response": TEMPORAL_REFUSAL
        }

    elif intent == "GENERAL_KNOWLEDGE":
        logger.info("General knowledge question intercepted — returning refusal.")
        return {
            "success": True,
            "intent": "GENERAL_KNOWLEDGE",
            "cache_hit": False,
            "rephrased_query": active_query,
            "generated_sql": None,
            "query_result": {},
            "nl_response": GENERAL_KNOWLEDGE_REFUSAL
        }
        
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
        # Handle the two failure modes with different user messages
        if sql_res.get("error") == "OUT_OF_SCOPE":
            return {
                "success": True,   # not a system error — a valid handled response
                "intent": "OUT_OF_SCOPE",
                "cache_hit": False,
                "rephrased_query": active_query,
                "generated_sql": None,
                "query_result": {},
                "nl_response": OUT_OF_SCOPE_REFUSAL
            }
        else:
            return {
                "success": False,
                "intent": "SQL_QUERY",
                "error_type": "SQL_GENERATION_FAILED",
                "rephrased_query": active_query,
                "error": sql_res.get("error", "Failed to generate SQL query."),
                "nl_response": "I couldn't find relevant data to answer that question."
            }
        
    sql_query = sql_res["sql_query"]
    
    # Execute SQL Query
    exec_res = execute_sql_query(sql_query)
    if not exec_res["success"]:
        logger.warning(f"Standard SQL execution failed: {exec_res.get('error')}. Falling back to autonomous LangChain agent...")
        try:
            from llm.langchain_agent import run_autonomous_sql_agent
            agent_res = run_autonomous_sql_agent(active_query)
            if agent_res["success"]:
                return {
                    "success": True,
                    "intent": "SQL_QUERY",
                    "cache_hit": False,
                    "generated_sql": agent_res.get("generated_sql", "LangChain Agent Execution"),
                    "query_result": {
                        "columns": ["Message"],
                        "rows": [[agent_res["nl_response"]]],
                        "row_count": 1
                    },
                    "rephrased_query": active_query,
                    "nl_response": agent_res["nl_response"]
                }
        except Exception as agent_err:
            logger.error(f"LangChain autonomous agent fallback failed: {agent_err}")
 
        return {
            "success": False,
            "intent": "SQL_QUERY",
            "error_type": exec_res.get("error_type", "SQL_EXECUTION_FAILED"),
            "error": exec_res.get("error", "Database execution error."),
            "generated_sql": sql_query,
            "rephrased_query": active_query,
            "nl_response": f"The generated SQL query failed to execute: {exec_res.get('error')}"
        }
        
    query_result = exec_res["result"]
    
    # Enrich the result for the LLM
    enriched_result = enrich_sql_result(query_result)
    
    # Generate Natural Language Insights using the enriched profile
    if stream:
        nl_res = generate_natural_language_response(active_query, sql_query, enriched_result, stream=True)
        raw_stream = nl_res["response_text"]

        def cached_stream_wrapper(strm, q, q_res):
            accumulated = []
            try:
                for chunk in strm:
                    accumulated.append(chunk)
                    yield chunk
            finally:
                full_txt = "".join(accumulated).strip()
                if full_txt:
                    store_in_query_cache(q, full_txt, q_res)

        return {
            "success": True,
            "intent": "SQL_QUERY",
            "cache_hit": False,
            "generated_sql": sql_query,
            "query_result": query_result,
            "rephrased_query": active_query,
            "nl_response": cached_stream_wrapper(raw_stream, active_query, query_result)
        }

    nl_res = generate_natural_language_response(active_query, sql_query, enriched_result, stream=False)
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
        ("How long ago was 2015?", "TEMPORAL"),
        ("What is today's date?", "TEMPORAL"),
        ("who is pm ?", "GENERAL_KNOWLEDGE"),
        ("who is President of USA ?", "GENERAL_KNOWLEDGE"),
        ("Who is the CEO of Microsoft?", "OUT_OF_SCOPE"),
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
