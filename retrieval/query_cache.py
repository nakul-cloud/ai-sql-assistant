"""
retrieval/query_cache.py
────────────────────────
Manages caching of previous user queries and natural language answers in Qdrant.

Uses a separate Qdrant collection (ai_sql_query_cache) with dense vector cosine similarity.
A match with similarity >= QUERY_CACHE_THRESHOLD (default: 0.97) is served immediately,
bypassing retriever, SQL generation, execution, and response writing.

Usage:
    python -m retrieval.query_cache
"""

import os
import sys
import uuid
from dotenv import load_dotenv

# Ensure PyTorch model is loaded first to prevent Windows library initialization crashes
from indexing.embedder import embed_text

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# Configurations
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
CACHE_COLLECTION = os.getenv("QUERY_CACHE_COLLECTION", "ai_sql_query_cache")
THRESHOLD = float(os.getenv("QUERY_CACHE_THRESHOLD", "0.97"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

_client = None

def get_qdrant_client() -> QdrantClient:
    """Lazy initialization of Qdrant Client."""
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_cache_collection() -> None:
    """Creates the cache collection in Qdrant if it does not exist."""
    client = get_qdrant_client()
    try:
        if not client.collection_exists(CACHE_COLLECTION):
            print(f"[INFO] Creating query cache collection: {CACHE_COLLECTION}")
            client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f"  [OK] Collection {CACHE_COLLECTION} created.")
    except Exception as e:
        print(f"  [FAIL] Failed to ensure collection {CACHE_COLLECTION}: {e}")


def check_query_cache(user_query: str) -> dict | None:
    """
    Search Qdrant cache collection for similar queries.
    Returns:
        A dict with the cached results if score >= THRESHOLD, else None.
    """
    ensure_cache_collection()
    client = get_qdrant_client()
    
    try:
        # Embed the query
        dense_vector = embed_text(user_query)["dense"]
        
        # Search Qdrant
        results = client.query_points(
            collection_name=CACHE_COLLECTION,
            query=dense_vector,
            limit=1,
            score_threshold=THRESHOLD
        )
        
        if results.points:
            match = results.points[0]
            payload = match.payload
            print(f"  [OK] Cache HIT (similarity score: {match.score:.4f})")
            return {
                "cache_hit": True,
                "original_query": payload.get("original_query"),
                "nl_response": payload.get("nl_response"),
                "rows": payload.get("rows", []),
                "columns": payload.get("columns", []),
                "similarity_score": match.score
            }
            
        print("  [INFO] Cache MISS")
        return None
        
    except Exception as e:
        print(f"  [WARNING] Query cache lookup failed: {e}")
        return None


def store_in_query_cache(user_query: str, nl_response: str, query_result: dict) -> bool:
    """
    Embed the user query and store the query result and NL response in the cache.
    """
    ensure_cache_collection()
    client = get_qdrant_client()
    
    try:
        # Embed the query
        dense_vector = embed_text(user_query)["dense"]
        
        point_id = str(uuid.uuid4())
        
        # Limit rows saved to cache to avoid enormous payloads
        rows_to_cache = query_result.get("rows", [])[:50]
        
        client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=dense_vector,
                    payload={
                        "original_query": user_query,
                        "nl_response": nl_response,
                        "rows": rows_to_cache,
                        "columns": query_result.get("columns", [])
                    }
                )
            ]
        )
        print("  [OK] Query result stored in cache.")
        return True
        
    except Exception as e:
        print(f"  [WARNING] Failed to store query in cache: {e}")
        return False


def clear_query_cache() -> bool:
    """Drop and recreate the query cache collection (clearing it completely)."""
    client = get_qdrant_client()
    try:
        if client.collection_exists(CACHE_COLLECTION):
            print(f"[INFO] Dropping cache collection: {CACHE_COLLECTION}")
            client.delete_collection(CACHE_COLLECTION)
        ensure_cache_collection()
        print("  [OK] Cache cleared.")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to clear query cache: {e}")
        return False


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing Query Cache Layer ---")
    
    # Clear cache first to isolate test
    clear_query_cache()
    
    test_query = "Who is the manager of sales?"
    mock_result = {
        "columns": ["department_name", "manager_name"],
        "rows": [["Sales", "Bob Smith"]]
    }
    mock_nl = "The manager of the Sales department is Bob Smith."
    
    # 1. First search: should be a MISS
    print("\n[INFO] Query 1 (Expected MISS):")
    res1 = check_query_cache(test_query)
    assert res1 is None, "Expected cache MISS"
    
    # 2. Store in cache
    print("\n[INFO] Storing result in cache:")
    stored = store_in_query_cache(test_query, mock_nl, mock_result)
    assert stored is True, "Failed to store in cache"
    
    # 3. Second search: should be a HIT
    print("\n[INFO] Query 2 (Expected HIT - exact match):")
    res2 = check_query_cache(test_query)
    assert res2 is not None, "Expected cache HIT"
    assert res2["nl_response"] == mock_nl
    
    # 4. Third search: semantic similarity check (very similar phrasing)
    similar_query = "Who is the manager for sales?"
    print(f"\n[INFO] Query 3 (Expected HIT - semantic match: '{similar_query}'):")
    res3 = check_query_cache(similar_query)
    assert res3 is not None, f"Expected cache HIT for similar query: {similar_query}"
    print(f"  Returned Answer: {res3['nl_response']}")
    print(f"  Similarity Score: {res3['similarity_score']:.4f}")
    
    # Cleanup after test
    print("\n[INFO] Cleaning up:")
    clear_query_cache()
    
    print("\n=======================================================")
    print("  [OK] Query cache layer -- all tests passed!")
    print("=======================================================")
    sys.exit(0)
