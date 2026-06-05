"""
indexing/qdrant_uploader.py
───────────────────────────
Embeds chunks and upserts them into Qdrant's sql_table_schemas collection.

Handles:
  • Collection creation (dense + sparse vector config)
  • Embedding chunks via BAAI/bge-m3
  • Upserting points with metadata payloads
  • Deleting chunks for a specific table (for incremental re-index)

Provides:
  • ensure_collection()              → create collection if not exists
  • upload_chunks(chunks)            → embed + upsert to Qdrant
  • delete_table_chunks(table_name)  → remove all chunks for a table
  • get_collection_info()            → point count + collection status

Usage:
    python -m indexing.qdrant_uploader
"""

# Import embedder first to initialize PyTorch/OpenMP runtimes before networking client libraries load
from indexing.embedder import embed_text, embed_texts

import os
import sys
import time
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

# ── Config from .env ─────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
SCHEMA_COLLECTION = os.getenv("SCHEMA_COLLECTION", "ai_sql_schema_index")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# ── Qdrant client (singleton) ───────────────────────────────────────
_client = None


def _get_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection() -> None:
    """
    Create the schema collection if it doesn't exist.
    Configured with both dense and sparse vector indexes.
    """
    client = _get_client()

    # Check if collection already exists
    collections = [c.name for c in client.get_collections().collections]
    if SCHEMA_COLLECTION in collections:
        info = client.get_collection(SCHEMA_COLLECTION)
        print(f"  Collection '{SCHEMA_COLLECTION}' exists ({info.points_count} points)")
        return

    # Create with dense + sparse vector config
    client.create_collection(
        collection_name=SCHEMA_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(),
            )
        },
    )
    print(f"  [OK] Created collection: {SCHEMA_COLLECTION}")
    print(f"     Dense: {EMBEDDING_DIMENSION}-dim, Cosine")
    print(f"     Sparse: BM25 (auto-indexed)")


def upload_chunks(chunks: list) -> int:
    """
    Embed and upsert chunks to Qdrant.

    Args:
        chunks: List of Chunk objects (from chunk_builder).

    Returns:
        Number of points successfully upserted.
    """
    client = _get_client()
    ensure_collection()

    # Batch embed all chunk texts
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    # Build Qdrant points
    points = []
    for chunk, vectors in zip(chunks, embeddings):
        sparse_data = vectors["sparse"]
        # Convert sparse dict keys to integers (token IDs)
        sparse_indices = [int(k) for k in sparse_data.keys()]
        sparse_values = list(sparse_data.values())

        point = PointStruct(
            id=str(uuid4()),
            vector={
                "dense": vectors["dense"],
                "sparse": SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            },
            payload={
                "table_name": chunk.metadata["table_name"],
                "chunk_type": chunk.metadata["chunk_type"],
                "text": chunk.text,
                "columns": chunk.metadata["columns"],
            },
        )
        points.append(point)

    # Upsert in batches of 64
    batch_size = 64
    upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=SCHEMA_COLLECTION,
            points=batch,
        )
        upserted += len(batch)

    return upserted


def delete_table_chunks(table_name: str) -> int:
    """
    Delete all chunks for a specific table from Qdrant.
    Used before incremental re-indexing.

    Returns:
        Number of points deleted (approximate).
    """
    client = _get_client()

    # Count before delete
    before = client.count(
        collection_name=SCHEMA_COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="table_name", match=MatchValue(value=table_name))]
        ),
    ).count

    # Delete matching points
    client.delete(
        collection_name=SCHEMA_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="table_name", match=MatchValue(value=table_name))]
        ),
    )

    return before


def get_collection_info() -> dict:
    """Return basic collection stats."""
    client = _get_client()

    collections = [c.name for c in client.get_collections().collections]
    if SCHEMA_COLLECTION not in collections:
        return {"exists": False, "points_count": 0}

    info = client.get_collection(SCHEMA_COLLECTION)
    return {
        "exists": True,
        "points_count": info.points_count,
        "status": str(info.status),
    }


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing Qdrant uploader ---\n")

    # Test 0: Connection check
    print("--- Test 0: Qdrant connection ---")
    try:
        client = _get_client()
        collections = client.get_collections()
        print(f"  [OK] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"  Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"  [FAIL] Cannot connect to Qdrant: {e}")
        print("  Is Docker running? Try: docker start qdrant")
        sys.exit(1)

    # Test 1: Ensure collection
    print("\n--- Test 1: Ensure collection ---")
    ensure_collection()

    # Test 2: Upload test chunks
    print("\n--- Test 2: Upload test chunks ---")
    from indexing.chunk_builder import Chunk

    test_chunks = [
        Chunk(
            text="Table: dbo.TestOrders\nColumns:\n  - OrderID (int) [PK]\n  - Total (decimal)\nRow Count: 100",
            metadata={
                "table_name": "dbo.TestOrders",
                "chunk_type": "structural",
                "columns": ["OrderID", "Total"],
            },
        ),
        Chunk(
            text="Table: dbo.TestOrders\nThis table tracks all customer orders including totals and dates.",
            metadata={
                "table_name": "dbo.TestOrders",
                "chunk_type": "semantic",
                "columns": ["OrderID", "Total"],
            },
        ),
    ]

    start = time.time()
    count = upload_chunks(test_chunks)
    elapsed = time.time() - start
    print(f"  Upserted : {count} points")
    print(f"  Time     : {elapsed:.2f}s")

    # Test 3: Check collection info
    print("\n--- Test 3: Collection info ---")
    info = get_collection_info()
    print(f"  Exists   : {info['exists']}")
    print(f"  Points   : {info['points_count']}")
    print(f"  Status   : {info.get('status', 'N/A')}")
    assert info["points_count"] >= 2, f"Expected >= 2 points, got {info['points_count']}"
    print(f"  [OK] Points in collection")

    # Test 4: Delete test chunks
    print("\n--- Test 4: Delete test table chunks ---")
    deleted = delete_table_chunks("dbo.TestOrders")
    print(f"  Deleted  : {deleted} points for 'dbo.TestOrders'")

    # Verify deletion
    import time as _t
    _t.sleep(0.5)  # Brief wait for Qdrant to process
    info = get_collection_info()
    print(f"  Remaining: {info['points_count']} points")
    print(f"  [OK] Cleanup complete")

    print("\n" + "=" * 55)
    print("  [OK] Qdrant uploader -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
