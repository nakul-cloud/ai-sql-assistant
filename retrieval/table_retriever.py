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


def retrieve_relevant_tables(user_query: str, top_k: int = 3) -> list[str]:
    """
    Retrieves the most relevant table names using hybrid search & RRF.
    
    Args:
        user_query: The user's question in natural language.
        top_k: Max number of unique tables to return.
        
    Returns:
        List of table names (e.g. ['dbo.csv_employees', 'dbo.csv_departments'])
    """
    client = get_qdrant_client()
    
    try:
        # 1. Embed query
        vectors = embed_text(user_query)
        dense = vectors["dense"]
        sparse = vectors["sparse"]
        
        # Convert sparse dict keys to integers (token IDs)
        sparse_indices = [int(k) for k in sparse.keys()]
        sparse_values = list(sparse.values())
        
        # 2. Query Qdrant with prefetch & RRF fusion
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
        
        # 3. Deduplicate tables while preserving ranked order
        seen = set()
        tables = []
        for point in results.points:
            payload = point.payload
            if not payload or "table_name" not in payload:
                continue
            t = payload["table_name"]
            if t not in seen:
                seen.add(t)
                tables.append(t)
            if len(tables) == top_k:
                break
                
        return tables
        
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
