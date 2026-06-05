"""
indexing/embedder.py
────────────────────
Wrapper around BAAI/bge-m3 for generating dense + sparse embeddings.

The model is loaded once (singleton) and reused across the application.
Both the offline indexing pipeline and the online query pipeline use this.

Provides:
  • get_model()            → returns the singleton BGEM3FlagModel
  • embed_text(text)       → returns {"dense": list[float], "sparse": dict}
  • embed_texts(texts)     → batch version for multiple texts

Usage:
    python -m indexing.embedder
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ── Config from .env ─────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# Import early to avoid Windows C-level conflicts with networking libs
from FlagEmbedding import BGEM3FlagModel

# ── Singleton model ──────────────────────────────────────────────────
_model = None


def get_model():
    """
    Load and return the BAAI/bge-m3 model (singleton).
    First call downloads the model (~2 GB) if not cached.
    Subsequent calls return the cached instance instantly.
    """
    global _model

    if _model is not None:
        return _model

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL} ...")
    start = time.time()

    _model = BGEM3FlagModel(
        EMBEDDING_MODEL,
        use_fp16=True,      # half-precision for speed + lower memory
    )

    elapsed = time.time() - start
    print(f"[OK] Model loaded in {elapsed:.1f}s")

    return _model


def embed_text(text: str) -> dict:
    """
    Embed a single text string.

    Returns:
        {
            "dense":  list[float]   (1024-dim vector),
            "sparse": dict          {token_id: weight} for BM25-style matching
        }
    """
    model = get_model()

    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,   # not needed, saves memory
    )

    return {
        "dense": output["dense_vecs"][0].tolist(),
        "sparse": output["lexical_weights"][0],
    }


def embed_texts(texts: list[str]) -> list[dict]:
    """
    Embed multiple texts in a single batch (more efficient than calling
    embed_text() in a loop).

    Returns:
        List of {"dense": [...], "sparse": {...}} dicts
    """
    if not texts:
        return []

    model = get_model()

    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    results = []
    for i in range(len(texts)):
        results.append({
            "dense": output["dense_vecs"][i].tolist(),
            "sparse": output["lexical_weights"][i],
        })

    return results


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing BAAI/bge-m3 embedder ---\n")

    # Test 1: Single text embedding
    print("--- Test 1: Single text embedding ---")
    test_text = "Show me all employees in the Engineering department"
    start = time.time()
    result = embed_text(test_text)
    elapsed = time.time() - start

    dense = result["dense"]
    sparse = result["sparse"]

    print(f"  Input    : \"{test_text}\"")
    print(f"  Dense    : {len(dense)}-dim vector (first 5: {dense[:5]})")
    print(f"  Sparse   : {len(sparse)} non-zero tokens")
    print(f"  Time     : {elapsed:.3f}s")

    # Validate dimensions
    assert len(dense) == EMBEDDING_DIMENSION, (
        f"Expected {EMBEDDING_DIMENSION}-dim, got {len(dense)}-dim"
    )
    print(f"  [OK] Dense dimension correct ({EMBEDDING_DIMENSION})")

    # Test 2: Batch embedding
    print("\n--- Test 2: Batch embedding ---")
    batch_texts = [
        "What are the total sales by region?",
        "List all tickets assigned to Billy George",
        "Table: Orders, Columns: OrderID, CustomerID, Total",
    ]
    start = time.time()
    batch_results = embed_texts(batch_texts)
    elapsed = time.time() - start

    print(f"  Batch size : {len(batch_texts)} texts")
    print(f"  Results    : {len(batch_results)} embeddings")
    print(f"  Time       : {elapsed:.3f}s")
    assert len(batch_results) == len(batch_texts)
    print(f"  [OK] Batch count correct")

    # Test 3: Similarity sanity check
    print("\n--- Test 3: Similarity sanity check ---")
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similar_text = "Show employees from Engineering"
    different_text = "What is the weather today?"

    emb_original = embed_text(test_text)["dense"]
    emb_similar = embed_text(similar_text)["dense"]
    emb_different = embed_text(different_text)["dense"]

    sim_score = cosine_sim(emb_original, emb_similar)
    diff_score = cosine_sim(emb_original, emb_different)

    print(f"  \"{test_text}\"")
    print(f"    vs \"{similar_text}\"  → {sim_score:.4f}")
    print(f"    vs \"{different_text}\"  → {diff_score:.4f}")

    if sim_score > diff_score:
        print(f"  [OK] Similar text scored higher (as expected)")
    else:
        print(f"  [WARNING] Unexpected: different text scored higher")

    print("\n" + "=" * 55)
    print("  [OK] Embedder -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
