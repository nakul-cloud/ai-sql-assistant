import time
import sys
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from workflow.process_query import process_user_query
from retrieval.query_router import route_query
from retrieval.table_retriever import retrieve_relevant_tables
from analysis.schema_context import generate_schema_context
from llm.query_ai import generate_sql_query
from workflow.query_executor import execute_sql_query
from analysis.result_enricher import enrich_sql_result
from llm.response_generator import generate_natural_language_response

def profile_query(user_query: str):
    logger.info(f"--- Profiling Query: '{user_query}' ---")
    
    t0 = time.time()
    logger.info("1. Routing intent...")
    intent = route_query(user_query)
    logger.info(f"[{(time.time()-t0):.2f}s] Intent routed: {intent}")
    
    t0 = time.time()
    logger.info("2. Retrieving tables...")
    tables = retrieve_relevant_tables(user_query, top_k=2)
    logger.info(f"[{(time.time()-t0):.2f}s] Retrieved tables: {tables}")
    
    t0 = time.time()
    logger.info("3. Generating schema context...")
    schema_context = generate_schema_context(user_query, focus_tables=tables)
    logger.info(f"[{(time.time()-t0):.2f}s] Generated schema context")
    
    t0 = time.time()
    logger.info("4. Generating SQL...")
    sql_res = generate_sql_query(user_query, schema_context)
    logger.info(f"[{(time.time()-t0):.2f}s] Generated SQL.")
    if not sql_res['success']:
        logger.error(f"  Error: {sql_res.get('error')}")
        return
        
    t0 = time.time()
    logger.info(f"5. Executing SQL: {sql_res['sql_query']}")
    exec_res = execute_sql_query(sql_res['sql_query'])
    logger.info(f"[{(time.time()-t0):.2f}s] Executed SQL.")
    if not exec_res['success']:
        logger.error(f"  Error: {exec_res.get('error')}")
        return
        
    t0 = time.time()
    logger.info("6. Enriching SQL Result...")
    enriched = enrich_sql_result(exec_res['result'])
    logger.info(f"[{(time.time()-t0):.2f}s] Enriched SQL Result")
    
    t0 = time.time()
    logger.info("7. Generating NL response...")
    nl_res = generate_natural_language_response(user_query, sql_res['sql_query'], enriched)
    logger.info(f"[{(time.time()-t0):.2f}s] Generated NL response.")
    logger.info(f"  Response: {nl_res.get('response_text', '')[:100]}...")

if __name__ == "__main__":
    profile_query("what is ticket description for Billy George ?")
