import os
import re
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
from memory.mem0_manager import contextualize, get_context_for_prompt, store as store_memory, get_all_memories
from analysis.schema_context import generate_schema_context
from llm.query_ai import generate_sql_query
from workflow.query_executor import execute_sql_query
from llm.response_generator import generate_natural_language_response
from analysis.result_enricher import enrich_sql_result
from database.schema_manager import fetch_database_metadata
from llm.llm_client import get_llm
from llm.query_transformer import transform_query

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

# ── Currency-symbol post-processor for agent fallback ────────────────────────
_CURRENCY_SYMBOLS = {"$", "€", "£", "₹", "¥"}
_CURRENCY_WORDS = {"usd", "eur", "gbp", "inr", "currency", "dollar", "rupee", "euro", "pound", "yen"}


def _strip_ungrounded_currency_symbols(text: str, schema_context: str) -> str:
    """
    Remove currency symbols that immediately precede a digit when the schema
    contains no evidence that any column carries currency data.
    Applied ONLY to the autonomous-agent fallback output (the standard
    pipeline's NL_PROMPT already has its own NO CURRENCY ASSUMPTIONS rule).
    """
    if not text:
        return text

    schema_lower = (schema_context or "").lower()

    # If the schema mentions any currency symbol or keyword, leave text as-is
    if any(sym in schema_lower for sym in _CURRENCY_SYMBOLS):
        return text
    if any(word in schema_lower for word in _CURRENCY_WORDS):
        return text

    # Strip currency symbols immediately before a digit
    return re.sub(r'[$€£₹¥](\d)', r'\1', text)

# ── LangChain CHAT chain ──────────────────────────────────────────────────────
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and professional Enterprise AI Analytics Assistant.
Answer the user's conversational message politely.

STRICT RULES:
- NEVER mention SQL, queries, commands, SELECT, FROM, WHERE, GROUP BY, ORDER BY, JOIN, or any database/technical terminology.
- NEVER suggest writing SQL or "querying the database". The user interacts in plain English only.
- Do NOT mention table names, schemas, column names, or internal processing.

You have access to the following relevant facts/memories from the conversation history:
{chat_history_str}

If the user asks about what was discussed, what they asked, or references previous topics, use this history to answer their question.
If they just greeted you, greet them back warmly and briefly mention the datasets you can help with: {available_tables}. Then suggest 2-3 example questions they could ask in plain English (e.g. "What is the average salary of placed candidates?", "Which industry has the highest AI adoption?")."""),
    ("human", "{user_query}")
])

_chat_chain = CHAT_PROMPT | get_llm(temperature=0.7) | StrOutputParser()


def handle_chat_query(user_query: str, chat_history_str: str = "None", stream: bool = False) -> Dict[str, Any]:
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

        inputs = {
            "user_query": user_query,
            "available_tables": available_tables,
            "chat_history_str": chat_history_str
        }

        if stream:
            token_stream = _chat_chain.stream(inputs)
            return {
                "success": True,
                "intent": "CHAT",
                "nl_response": token_stream,
                "cache_hit": False
            }

        response_text = _chat_chain.invoke(inputs)
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

        # 1. Determine target tables semantically if not a general query
        tables = None
        if not is_general_query:
            retrieved = retrieve_relevant_tables(user_query, top_k=1)
            if retrieved:
                # If focus_tables is specified, ensure the retrieved table is in it
                if not focus_tables or retrieved[0] in focus_tables:
                    tables = retrieved
        
        # Fallback to focus_tables if it has exactly 1 table
        if not tables and focus_tables and len(focus_tables) == 1:
            tables = focus_tables

        # 2. Get the schema context for the identified tables
        if tables:
            logger.info(f"DESCRIBE intent resolved to tables: {tables}")
            relevant_schema = generate_schema_context(user_query, focus_tables=tables)
            description = generate_dataset_description(user_query, relevant_schema, stream=stream)
        else:
            logger.info("DESCRIBE intent fell back to check general overview.")
            if is_general_query or not focus_tables:
                relevant_schema = get_all_table_summaries()
                description = generate_database_overview(user_query, relevant_schema, stream=stream)
            else:
                # Describe the first table in focus_tables or all of them if general query
                relevant_schema = generate_schema_context(user_query, focus_tables=focus_tables)
                description = generate_database_overview(user_query, relevant_schema, stream=stream)

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





def handle_schema_explanation_query(user_query: str) -> Dict[str, Any]:
    """
    Handles 'what is column_name?' type questions.
    Looks up column metadata and semantic description — no SQL execution.
    """
    logger.info(f"Handling SCHEMA_EXPLANATION query: '{user_query}'")
    try:
        from database.schema_manager import fetch_database_metadata
        from indexing.semantic_description import get_column_descriptions

        metadata_list = fetch_database_metadata()

        # Find which column the user is asking about
        q_lower = user_query.lower()
        matched_col = None
        matched_table = None
        col_semantic_desc = None

        for meta in metadata_list:
            col_descs = get_column_descriptions(meta["table_name"])
            for col in meta["columns"]:
                col_name = col["name"].lower()
                col_name_spaced = col_name.replace("_", " ")
                # Check for exact word matching to avoid false positives (e.g. matching 'ssc_p' inside 'hsc_p')
                col_pattern = r"\b" + re.escape(col_name) + r"\b"
                col_spaced_pattern = r"\b" + re.escape(col_name_spaced) + r"\b"
                if re.search(col_pattern, q_lower) or re.search(col_spaced_pattern, q_lower):
                    matched_col = col
                    matched_table = meta["table_name"]
                    col_semantic_desc = col_descs.get(col["name"], "")
                    break
            if matched_col:
                break

        if matched_col:
            # Build a plain English explanation using metadata
            from llm.llm_client import get_llm
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
                ("system", """You are a data analyst explaining a database column to a business user.
Explain what the column means in plain English. Be concise — 2-3 sentences max.
Use the column name, data type, and semantic description provided.
Do not mention SQL, databases, or technical implementation details.
If sample values are provided, use them to make the explanation concrete."""),
                ("human", """Column name: {col_name}
Data type: {col_type}
Table it belongs to: {table_name}
Semantic description: {semantic_desc}
Sample values: {sample_values}

Explain this column in plain English:""")
            ])

            sample_vals = ""
            for meta in metadata_list:
                if meta["table_name"] == matched_table:
                    sv = meta.get("sample_values", {}).get(matched_col["name"], [])
                    sample_vals = str(sv[:5]) if sv else "Not available"
                    break

            chain = EXPLAIN_PROMPT | get_llm(temperature=0.2) | StrOutputParser()
            explanation = chain.invoke({
                "col_name": matched_col["name"],
                "col_type": matched_col.get("display_type", matched_col.get("type", "unknown")),
                "table_name": matched_table,
                "semantic_desc": col_semantic_desc or "No description available",
                "sample_values": sample_vals
            })

            return {
                "success": True,
                "intent": "SCHEMA_EXPLANATION",
                "cache_hit": False,
                "nl_response": explanation.strip(),
                "generated_sql": None,
                "query_result": {}
            }

        else:
            # Column not found — fall back to schema info handler
            logger.info("Column not found for SCHEMA_EXPLANATION — falling back to SCHEMA_INFO.")
            return handle_schema_info_query(user_query)

    except Exception as e:
        logger.exception("Failed to handle schema explanation.")
        return {
            "success": False,
            "intent": "SCHEMA_EXPLANATION",
            "error": str(e),
            "nl_response": "I couldn't find information about that column. Try asking 'what columns are available?' to see what's in the database."
        }





def handle_conversation_summary(user_query: str, user_id: str, stream: bool = False) -> Dict[str, Any]:
    """
    Summarize stored conversational memories for the user using response_mode="MEMORY".
    """
    logger.info(f"Handling CONVERSATION_SUMMARY query for user '{user_id}'")
    
    memories = get_all_memories(user_id)
    if not memories:
        history_str = "No prior discussion has been recorded yet. Feel free to ask about our datasets!"
    else:
        lines = []
        for m in memories:
            text = m.get("memory", "") if isinstance(m, dict) else str(m)
            if text:
                lines.append(f"- {text}")
        history_str = "\n".join(lines)
        
    res = generate_natural_language_response(
        user_query=user_query,
        sql_query="",
        enriched_result={},
        chat_history_str=history_str,
        stream=stream,
        response_mode="MEMORY"
    )
    
    nl_response = res["response_text"]
    
    if stream:
        def summary_stream_wrapper(strm):
            accumulated = []
            try:
                for chunk in strm:
                    accumulated.append(chunk)
                    yield chunk
            finally:
                full_txt = "".join(accumulated).strip()
                if full_txt:
                    store_memory(user_id, user_query, full_txt, {"intent": "CONVERSATION_SUMMARY"})
        return {
            "success": True,
            "intent": "CONVERSATION_SUMMARY",
            "cache_hit": False,
            "nl_response": summary_stream_wrapper(nl_response)
        }
        
    store_memory(user_id, user_query, nl_response, {"intent": "CONVERSATION_SUMMARY"})
    return {
        "success": True,
        "intent": "CONVERSATION_SUMMARY",
        "cache_hit": False,
        "nl_response": nl_response
    }


def process_user_query(
    user_query: str,
    focus_tables: list = None,
    chat_history: list = None,  # kept for backward compat, no longer used internally
    stream: bool = False,
    user_id: str = None         # mem0 user identifier
) -> Dict[str, Any]:
    """
    Unified query processing pipeline.
    Routes the query and performs caching, retrieval, generation, and execution as appropriate.
    Uses mem0 Cloud for conversational memory and follow-up contextualization.
    """
    _user_id = user_id or os.getenv("MEM0_USER_ID", "default_user")

    # 1. Determine Intent on the RAW query first (bypassing contextualization for conversational intents)
    intent = route_query(user_query)
    logger.info(f"Raw query routed intent: {intent}")
    
    if intent == "CHAT":
        # Get chat history for conversational facts context
        chat_hist_str = get_context_for_prompt(_user_id, user_query)
        res = handle_chat_query(user_query, chat_history_str=chat_hist_str, stream=stream)
        res["rephrased_query"] = user_query
        if res.get("success") and isinstance(res.get("nl_response"), str):
            store_memory(_user_id, user_query, res["nl_response"], {"intent": "CHAT"})
        return res
        
    elif intent == "CONVERSATION_SUMMARY":
        res = handle_conversation_summary(user_query, user_id=_user_id, stream=stream)
        res["rephrased_query"] = user_query
        return res

    elif intent == "SCHEMA_INFO":
        res = handle_schema_info_query(user_query, stream=stream)
        res["rephrased_query"] = user_query
        if res.get("success") and isinstance(res.get("nl_response"), str):
            store_memory(_user_id, user_query, res["nl_response"], {"intent": "SCHEMA_INFO"})
        return res

    elif intent == "SCHEMA_EXPLANATION":
        res = handle_schema_explanation_query(user_query)
        res["rephrased_query"] = user_query
        if res.get("success") and isinstance(res.get("nl_response"), str):
            store_memory(_user_id, user_query, res["nl_response"], {"intent": "SCHEMA_EXPLANATION"})
        return res

    elif intent == "DESCRIBE":
        res = handle_describe_intent(user_query, focus_tables=focus_tables, stream=stream)
        res["rephrased_query"] = user_query
        if res.get("success") and isinstance(res.get("nl_response"), str):
            store_memory(_user_id, user_query, res["nl_response"], {"intent": "DESCRIBE"})
        return res

    elif intent == "TEMPORAL":
        logger.info("Temporal question intercepted — returning hardcoded refusal.")
        return {
            "success": True,
            "intent": "TEMPORAL",
            "cache_hit": False,
            "rephrased_query": user_query,
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
            "rephrased_query": user_query,
            "generated_sql": None,
            "query_result": {},
            "nl_response": GENERAL_KNOWLEDGE_REFUSAL
        }

    # 2. Contextualize only if intent is SQL/Preview related
    active_query = contextualize(user_query, _user_id)
    logger.info(f"Processing query: '{active_query}' (Original: '{user_query}')")
    
    # Re-route the contextualized query
    resolved_intent = route_query(active_query)
    logger.info(f"Contextualized routed intent: {resolved_intent}")

    # Check for DATA_PREVIEW
    if resolved_intent == "DATA_PREVIEW":
        # Resolve table to preview: retrieve if focus_tables has multiple tables or is empty
        if focus_tables and len(focus_tables) == 1:
            target_table = focus_tables[0]
        else:
            retrieved = retrieve_relevant_tables(active_query, top_k=1)
            if retrieved:
                target_table = retrieved[0]
            elif focus_tables:
                target_table = focus_tables[0]
            else:
                target_table = None

        if not target_table:
            return {
                "success": False,
                "intent": "DATA_PREVIEW",
                "error": "No table matched for data preview.",
                "nl_response": "I couldn't identify which table or dataset you would like to see."
            }
        
        sql_query = f"SELECT * FROM {target_table}"
        logger.info(f"Executing data preview query: {sql_query}")
        
        # Execute query
        exec_res = execute_sql_query(sql_query)
        if not exec_res["success"]:
            return {
                "success": False,
                "intent": "DATA_PREVIEW",
                "error": exec_res.get("error"),
                "nl_response": f"Failed to retrieve data preview: {exec_res.get('error')}"
            }
            
        query_result = exec_res["result"]
        enriched_result = enrich_sql_result(query_result, sql_query=sql_query)
        chat_hist_str = get_context_for_prompt(_user_id, active_query)
        
        # Generate minimal response using PREVIEW mode
        nl_res = generate_natural_language_response(
            active_query,
            sql_query,
            enriched_result,
            chat_history_str=chat_hist_str,
            stream=stream,
            response_mode="PREVIEW"
        )
        nl_response = nl_res["response_text"]
        
        if stream:
            def preview_stream_wrapper(strm, q, q_res):
                accumulated = []
                try:
                    for chunk in strm:
                        accumulated.append(chunk)
                        yield chunk
                finally:
                    full_txt = "".join(accumulated).strip()
                    if full_txt:
                        store_in_query_cache(q, full_txt, q_res)
                        store_memory(
                            user_id=_user_id,
                            user_query=q,
                            assistant_response=full_txt,
                            metadata={"intent": "DATA_PREVIEW", "tables": [target_table]},
                            sql_query=sql_query,
                            query_result=q_res
                        )
            return {
                "success": True,
                "intent": "DATA_PREVIEW",
                "cache_hit": False,
                "generated_sql": sql_query,
                "query_result": query_result,
                "rephrased_query": active_query,
                "nl_response": preview_stream_wrapper(nl_response, active_query, query_result)
            }
            
        store_in_query_cache(active_query, nl_response, query_result)
        store_memory(
            user_id=_user_id,
            user_query=active_query,
            assistant_response=nl_response,
            metadata={"intent": "DATA_PREVIEW", "tables": [target_table]},
            sql_query=sql_query,
            query_result=query_result
        )
        return {
            "success": True,
            "intent": "DATA_PREVIEW",
            "cache_hit": False,
            "generated_sql": sql_query,
            "query_result": query_result,
            "rephrased_query": active_query,
            "nl_response": nl_response
        }

    # 3. Apply Advanced Query Transformations
    transform_result = transform_query(active_query)

    if transform_result.is_decomposed:
        # ── Multi-part query: execute each sub-query independently ────────
        logger.info(f"Executing {len(transform_result.sub_queries)} decomposed sub-queries")
        all_nl_parts = []
        all_sql_parts = []
        combined_rows = []
        combined_cols = []

        for i, sq in enumerate(transform_result.sub_queries, 1):
            logger.info(f"Sub-query {i}/{len(transform_result.sub_queries)}: '{sq}'")
            try:
                sq_tables = retrieve_relevant_tables(sq, top_k=2)
                sq_schema = generate_schema_context(sq, focus_tables=sq_tables)
                sq_sql_res = generate_sql_query(sq, sq_schema)

                if not sq_sql_res["success"]:
                    all_nl_parts.append(f"**Part {i}:** I couldn't generate an answer for: \"{sq}\"")
                    continue

                sq_exec = execute_sql_query(sq_sql_res["sql_query"])
                if not sq_exec["success"]:
                    all_nl_parts.append(f"**Part {i}:** The query for \"{sq}\" failed to execute.")
                    continue

                sq_enriched = enrich_sql_result(sq_exec["result"], sql_query=sq_sql_res["sql_query"])
                sq_chat_hist = get_context_for_prompt(_user_id, sq)
                sq_nl = generate_natural_language_response(
                    sq, sq_sql_res["sql_query"], sq_enriched,
                    chat_history_str=sq_chat_hist, stream=False
                )
                all_nl_parts.append(sq_nl["response_text"])
                all_sql_parts.append(sq_sql_res["sql_query"])
                combined_rows.extend(sq_exec["result"].get("rows", [])[:25])
                if not combined_cols:
                    combined_cols = sq_exec["result"].get("columns", [])
            except Exception as sub_err:
                logger.warning(f"Sub-query {i} failed: {sub_err}")
                all_nl_parts.append(f"**Part {i}:** Could not process: \"{sq}\"")

        combined_nl = "\n\n---\n\n".join(all_nl_parts) if all_nl_parts else "I couldn't process any part of your question."
        combined_sql = " ; ".join(all_sql_parts) if all_sql_parts else None
        combined_result = {"columns": combined_cols, "rows": combined_rows, "row_count": len(combined_rows)}

        store_in_query_cache(active_query, combined_nl, combined_result)
        store_memory(
            user_id=_user_id, user_query=active_query,
            assistant_response=combined_nl,
            metadata={"intent": "SQL_QUERY", "decomposed": True}
        )
        return {
            "success": True,
            "intent": "SQL_QUERY",
            "cache_hit": False,
            "generated_sql": combined_sql,
            "query_result": combined_result,
            "rephrased_query": active_query,
            "nl_response": combined_nl
        }

    # ── Single query: apply rewriting and step-back ───────────────────────
    active_query = transform_result.rewritten_query
    if transform_result.stepback_query:
        logger.info(f"Step-back query for enhanced retrieval: '{transform_result.stepback_query}'")

    # 4. Standard SQL Query Execution Pipeline
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
        # Enhance retrieval with step-back query if available
        if transform_result.stepback_query:
            stepback_tables = retrieve_relevant_tables(transform_result.stepback_query, top_k=2)
            for t in stepback_tables:
                if t not in focus_tables:
                    focus_tables.append(t)
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
                cleaned_nl = _strip_ungrounded_currency_symbols(
                    agent_res["nl_response"], schema_context
                )
                return {
                    "success": True,
                    "intent": "SQL_QUERY",
                    "cache_hit": False,
                    "generated_sql": agent_res.get("generated_sql", "LangChain Agent Execution"),
                    "query_result": {},
                    "rephrased_query": active_query,
                    "nl_response": cleaned_nl
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
    enriched_result = enrich_sql_result(query_result, sql_query=sql_query)
    
    # Generate Natural Language Insights using the enriched profile
    chat_hist_str = get_context_for_prompt(_user_id, active_query)
    if stream:
        nl_res = generate_natural_language_response(
            active_query,
            sql_query,
            enriched_result,
            chat_history_str=chat_hist_str,
            stream=True
        )
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
                    store_memory(
                        user_id=_user_id,
                        user_query=q,
                        assistant_response=full_txt,
                        metadata={"intent": "SQL_QUERY", "tables": focus_tables or []},
                        sql_query=sql_query,
                        query_result=q_res
                    )

        return {
            "success": True,
            "intent": "SQL_QUERY",
            "cache_hit": False,
            "generated_sql": sql_query,
            "query_result": query_result,
            "rephrased_query": active_query,
            "nl_response": cached_stream_wrapper(raw_stream, active_query, query_result)
        }

    nl_res = generate_natural_language_response(
        active_query,
        sql_query,
        enriched_result,
        chat_history_str=chat_hist_str,
        stream=False
    )
    nl_response = nl_res["response_text"]
    
    # Store result in cache
    store_in_query_cache(active_query, nl_response, query_result)
    store_memory(
        user_id=_user_id,
        user_query=active_query,
        assistant_response=nl_response,
        metadata={"intent": "SQL_QUERY", "tables": focus_tables or []},
        sql_query=sql_query,
        query_result=query_result
    )
    
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
