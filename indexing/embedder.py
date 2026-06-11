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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace Offline Mode ────────────────────────────────────────
# Disable all internet checks — model is already cached locally.
# This eliminates the 10-15s of HTTP requests to HuggingFace on every load.
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.getenv("HF_DATASETS_OFFLINE", "1"))

# ── Config from .env ─────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Limit CPU threads for PyTorch to avoid Windows scheduling contention
import torch
torch.set_num_threads(1)

# ── Singleton model ──────────────────────────────────────────────────
_model = None


def _load_model_resource():
    """Actually load BGE-M3 model and run warmup."""
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(
        EMBEDDING_MODEL,
        use_fp16=False,
        devices="cpu"
    )
    # Warm up the PyTorch compilation and thread pool for CPU execution
    model.encode(["warmup"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    return model


try:
    import streamlit as st
    @st.cache_resource
    def _get_streamlit_cached_model():
        print(f"[INFO] Loading BAAI/bge-m3 embedding model inside Streamlit cache...")
        return _load_model_resource()
except (ImportError, AttributeError):
    pass


def get_model(force: bool = False):
    """
    Load and return the BAAI/bge-m3 model (singleton).
    First call downloads the model (~2 GB) if not cached.
    Subsequent calls return the cached instance instantly.
    """
    global _model

    if EMBEDDING_PROVIDER != "local" and not force:
        return None

    # Try to use Streamlit's cache if running under Streamlit
    try:
        import streamlit as st
        if st.runtime.exists():
            return _get_streamlit_cached_model()
    except (ImportError, AttributeError):
        pass

    if _model is not None:
        return _model

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL} ...")
    start = time.time()

    _model = _load_model_resource()

    elapsed = time.time() - start
    print(f"[OK] Model loaded and warmed up in {elapsed:.1f}s")

    return _model


from functools import lru_cache

@lru_cache(maxsize=128)
def embed_text(text: str) -> dict:
    """
    Embed a single text string with fallback support: BGE -> OpenAI -> Gemini.

    Returns:
        {
            "dense":  list[float]   (1024-dim vector),
            "sparse": dict          {token_id: weight} for BM25-style matching
        }
    """
    # Helper to call OpenAI
    def _embed_openai(t):
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from langchain_openai import OpenAIEmbeddings
            openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            emb_fn = OpenAIEmbeddings(
                model=openai_model,
                api_key=openai_key,
                dimensions=1024
            )
            return {
                "dense": emb_fn.embed_query(t),
                "sparse": {}
            }
        raise ValueError("OPENAI_API_KEY not set")

    # Helper to call Gemini
    def _embed_gemini(t):
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            gemini_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
            emb_fn = GoogleGenerativeAIEmbeddings(
                model=gemini_model,
                google_api_key=gemini_key
            )
            vector = emb_fn.embed_query(t)
            if len(vector) < 1024:
                vector = vector + [0.0] * (1024 - len(vector))
            elif len(vector) > 1024:
                vector = vector[:1024]
            return {
                "dense": vector,
                "sparse": {}
            }
        raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY not set")

    # Helper to call Local BGE
    def _embed_local(t, force_load=False):
        model = get_model(force=force_load)
        if model is None:
            raise ValueError("Local embedding model not loaded")
        output = model.encode(
            [t],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": output["dense_vecs"][0].tolist(),
            "sparse": output["lexical_weights"][0],
        }

    # Execute based on preferred provider
    if EMBEDDING_PROVIDER == "openai":
        try:
            return _embed_openai(text)
        except Exception as e:
            print(f"[WARNING] Primary OpenAI embedding failed: {e}")
    elif EMBEDDING_PROVIDER == "gemini":
        try:
            return _embed_gemini(text)
        except Exception as e:
            print(f"[WARNING] Primary Gemini embedding failed: {e}")
    else:
        try:
            return _embed_local(text, force_load=False)
        except Exception as e:
            print(f"[WARNING] Primary Local embedding failed: {e}")

    # Fallback Cascade
    for provider in ["local", "openai", "gemini"]:
        if provider == EMBEDDING_PROVIDER:
            continue
        try:
            if provider == "local":
                return _embed_local(text, force_load=True)
            elif provider == "openai":
                return _embed_openai(text)
            elif provider == "gemini":
                return _embed_gemini(text)
        except Exception:
            pass

    raise ValueError("All embedding options (BGE, OpenAI, Gemini) failed or are unconfigured.")


def embed_texts(texts: list[str]) -> list[dict]:
    """
    Embed multiple texts in a single batch with fallback support: BGE -> OpenAI -> Gemini.

    Returns:
        List of {"dense": [...], "sparse": {...}} dicts
    """
    if not texts:
        return []

    # Helper for OpenAI
    def _batch_openai(ts):
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from langchain_openai import OpenAIEmbeddings
            openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            emb_fn = OpenAIEmbeddings(
                model=openai_model,
                api_key=openai_key,
                dimensions=1024
            )
            vectors = emb_fn.embed_documents(ts)
            return [{"dense": vec, "sparse": {}} for vec in vectors]
        raise ValueError("OPENAI_API_KEY not set")

    # Helper for Gemini
    def _batch_gemini(ts):
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            gemini_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
            emb_fn = GoogleGenerativeAIEmbeddings(
                model=gemini_model,
                google_api_key=gemini_key
            )
            vectors = emb_fn.embed_documents(ts)
            results = []
            for vec in vectors:
                if len(vec) < 1024:
                    vec = vec + [0.0] * (1024 - len(vec))
                elif len(vec) > 1024:
                    vec = vec[:1024]
                results.append({"dense": vec, "sparse": {}})
            return results
        raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY not set")

    # Helper for Local BGE
    def _batch_local(ts, force_load=False):
        get_model(force=force_load)
        results = []
        for text in ts:
            results.append(embed_text(text))
        return results

    # Execute based on preferred provider
    if EMBEDDING_PROVIDER == "openai":
        try:
            return _batch_openai(texts)
        except Exception as e:
            print(f"[WARNING] Primary OpenAI batch embedding failed: {e}")
    elif EMBEDDING_PROVIDER == "gemini":
        try:
            return _batch_gemini(texts)
        except Exception as e:
            print(f"[WARNING] Primary Gemini batch embedding failed: {e}")
    else:
        try:
            return _batch_local(texts, force_load=False)
        except Exception as e:
            print(f"[WARNING] Primary Local batch embedding failed: {e}")

    # Fallback Cascade
    for provider in ["local", "openai", "gemini"]:
        if provider == EMBEDDING_PROVIDER:
            continue
        try:
            if provider == "local":
                return _batch_local(texts, force_load=True)
            elif provider == "openai":
                return _batch_openai(texts)
            elif provider == "gemini":
                return _batch_gemini(texts)
        except Exception:
            pass

    raise ValueError("All embedding options (BGE, OpenAI, Gemini) failed or are unconfigured.")


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
    print(f"    vs \"{similar_text}\"  -> {sim_score:.4f}")
    print(f"    vs \"{different_text}\"  -> {diff_score:.4f}")

    if sim_score > diff_score:
        print(f"  [OK] Similar text scored higher (as expected)")
    else:
        print(f"  [WARNING] Unexpected: different text scored higher")

    print("\n" + "=" * 55)
    print("  [OK] Embedder -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
