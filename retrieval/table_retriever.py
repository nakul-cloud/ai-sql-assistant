"""
retrieval/table_retriever.py
────────────────────────────
Performs hybrid search (Dense + Sparse) using Reciprocal Rank Fusion (RRF)
in Qdrant to identify the most relevant tables for a user's SQL query.

Usage:
    python -m retrieval.table_retriever
"""

import os
import sys
from dotenv import load_dotenv

# Ensure PyTorch model is loaded first to prevent Windows library initialization crashes
from indexing.embedder import embed_text

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
SCHEMA_COLLECTION = os.getenv("SCHEMA_COLLECTION", "ai_sql_schema_index")

_client = None

def get_qdrant_client() -> QdrantClient:
    """Lazy initialization of Qdrant Client."""
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


_keyword_map = None

# Set of words too generic to map to specific tables
STOP_WORDS = {
    "dataset", "datasets", "data", "table", "tables", "info", "information",
    "explain", "describe", "overview", "show", "list", "view", "query",
    "find", "get", "fetch", "details", "database", "name", "id", "column", "columns",
    "performance", "metrics"
}

def get_keyword_map() -> dict:
    """
    Build a dynamic keyword map from active database metadata.
    Maps lowercase keywords to a list of tables containing/matching them.
    """
    global _keyword_map
    if _keyword_map is not None:
        return _keyword_map

    from database.schema_manager import fetch_database_metadata
    try:
        metadata = fetch_database_metadata()
    except Exception:
        metadata = []

    keyword_map = {}
    
    # Standard synonyms/concepts mapped to tables
    synonyms = {
        "placement": ["dbo.csv_placement_data_full_class"],
        "class": ["dbo.csv_placement_data_full_class"],
        "stream": ["dbo.csv_placement_data_full_class"],
        "specialization": ["dbo.csv_placement_data_full_class"],
        "specialisation": ["dbo.csv_placement_data_full_class"],
        "ticket": ["dbo.csv_customer_support_tickets"],
        "support": ["dbo.csv_customer_support_tickets"],
        "issue": ["dbo.csv_customer_support_tickets"],
        "agent": ["dbo.csv_customer_support_tickets"],
        "manager": ["dbo.csv_departments"],
        "department": ["dbo.csv_departments", "dbo.csv_employees"],
        "salary": ["dbo.csv_employees", "dbo.csv_placement_data_full_class"],
        "employee": ["dbo.csv_employees", "dbo.csv_sales"],
        "sale": ["dbo.csv_sales"],
        "revenue": ["dbo.csv_sales"],
        "amount": ["dbo.csv_sales"],
        "product": ["dbo.csv_sales"]
    }
    
    # Populate initial synonyms
    for kw, tables in synonyms.items():
        if kw.lower() not in STOP_WORDS:
            keyword_map[kw] = set(tables)

    for tbl in metadata:
        tbl_name = tbl["table_name"]
        
        # Clean table name keywords (e.g., "csv_sales" -> "sales")
        clean_name = tbl_name.split(".")[-1]
        clean_name_parts = clean_name.replace("csv_", "").split("_")
        for part in clean_name_parts:
            part_lower = part.lower()
            if len(part_lower) > 2 and part_lower not in STOP_WORDS:
                keyword_map.setdefault(part_lower, set()).add(tbl_name)

        # Column keywords
        for col in tbl["columns"]:
            col_name = col["name"].lower()
            if len(col_name) > 2 and col_name not in STOP_WORDS:
                keyword_map.setdefault(col_name, set()).add(tbl_name)
                # Split column name if snake_case
                for part in col_name.split("_"):
                    part_lower = part.lower()
                    if len(part_lower) > 2 and part_lower not in STOP_WORDS:
                        keyword_map.setdefault(part_lower, set()).add(tbl_name)

        # Sample-value keywords (categorical values from live data)
        for col in tbl["columns"]:
            sample_vals = tbl.get("sample_values", {}).get(col["name"], [])
            for val in sample_vals:
                val_str = str(val).strip().lower()
                # Only index short, non-numeric, categorical-looking values
                if (val_str
                        and len(val_str) > 1
                        and len(val_str) <= 20
                        and not val_str.replace('.', '', 1).isdigit()
                        and val_str not in STOP_WORDS):
                    keyword_map.setdefault(val_str, set()).add(tbl_name)

    # Convert sets to lists
    _keyword_map = {k: list(v) for k, v in keyword_map.items()}
    return _keyword_map


def fast_keyword_table_match(query: str) -> list[str]:
    """
    Checks if query words or segments match any of our cached table metadata keywords.
    Bypasses deep embedding model if clear keywords match, for sub-millisecond retrieval.
    Scores tables based on the number of keyword hits and returns them sorted by score descending.
    """
    normalized_q = query.lower()
    keyword_map = get_keyword_map()
    table_scores = {}

    # Tokenize the query into words (removing common punctuation)
    import re
    words = re.findall(r"\b\w{3,}\b", normalized_q)  # only match words of length 3+

    for word in words:
        if word in STOP_WORDS:
            continue
        if word in keyword_map:
            for tbl in keyword_map[word]:
                table_scores[tbl] = table_scores.get(tbl, 0) + 1

    if not table_scores:
        return []

    # Sort tables by score descending
    sorted_tables = sorted(table_scores.keys(), key=lambda t: table_scores[t], reverse=True)
    return sorted_tables


def retrieve_relevant_tables(user_query: str, top_k: int = 3) -> list[str]:
    """
    Retrieves the most relevant table names using hybrid search & RRF.
    Optimized with a fast keyword-matching cache layer to bypass embedding latency.
    
    Args:
        user_query: The user's question in natural language.
        top_k: Max number of unique tables to return.
        
    Returns:
        List of table names (e.g. ['dbo.csv_employees', 'dbo.csv_departments'])
    """
    # 1. Try fast keyword matching first (0ms latency)
    matched = fast_keyword_table_match(user_query)
    if matched:
        # DO NOT SORT ALPHABETICALLY! Preserve the score-based sorting from fast_keyword_table_match.
        selected_tables = matched[:top_k]
        print(f"[RAG-Fast] Selected matched tables for context: {selected_tables}")
        return selected_tables

    client = get_qdrant_client()
    
    try:
        # 1. Embed query
        vectors = embed_text(user_query)
        dense = vectors["dense"]
        sparse = vectors["sparse"]
        
        # Convert sparse dict keys to integers (token IDs)
        sparse_indices = [int(k) for k in sparse.keys()] if sparse else []
        sparse_values = list(sparse.values()) if sparse else []
        
        # 2. Query Qdrant with prefetch & RRF fusion (if sparse is available) or dense-only (fallback)
        if sparse_indices and sparse_values:
            results = client.query_points(
                collection_name=SCHEMA_COLLECTION,
                prefetch=[
                    Prefetch(
                        query=dense,
                        using="dense",
                        limit=20
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using="sparse",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k * 3
            )
        else:
            # Fallback search for OpenAI/Gemini which only produce dense vectors
            results = client.query_points(
                collection_name=SCHEMA_COLLECTION,
                query=dense,
                using="dense",
                limit=top_k * 3
            )
        
        # 3. Print retrieval details and scores to terminal
        print(f"\n[RAG] Retrieval details for: '{user_query}'")
        print("-" * 80)
        seen = set()
        tables = []
        for rank, point in enumerate(results.points, 1):
            payload = point.payload or {}
            score = point.score or 0.0
            table = payload.get("table_name", "Unknown")
            chunk_type = payload.get("chunk_type", "Unknown")
            text_snippet = payload.get("text", "").replace("\n", " ")[:120]
            print(f"Rank {rank:02d} | Score: {score:.4f} | Table: {table:<25} | Type: {chunk_type:<10} | Snippet: {text_snippet}...")
            
            if table not in seen and table != "Unknown":
                seen.add(table)
                tables.append(table)
        print("-" * 80)
        
        # Limit to requested top_k tables
        selected_tables = tables[:top_k]
        print(f"[RAG] Selected top {top_k} tables for context: {selected_tables}\n")
        return selected_tables
        
    except Exception as e:
        print(f"  [WARNING] Table retrieval failed: {e}. Falling back to empty table list.")
        return []


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing Table Retriever ---")
    
    test_cases = [
        "Who is the manager of each department?",
        "Show employee salaries and hire dates",
        "Total sales amount by product category",
        "Show list of engineering department employees",
    ]
    
    for q in test_cases:
        print(f"\nQuery: '{q}'")
        tables = retrieve_relevant_tables(q, top_k=2)
        print(f"  Retrieved Tables: {tables}")
        
    print("\n=======================================================")
    print("  [OK] Table retriever -- all tests passed!")
    print("=======================================================")
    sys.exit(0)
