"""
indexing/chunk_builder.py
─────────────────────────
Builds two chunks per table for the Qdrant index:

  1. Structural Chunk — column names, types, PKs, row count
     (good for exact column-name matching)

  2. Semantic Chunk — LLM-generated business description
     (good for concept/intent matching)

Both chunks are embedded and stored in Qdrant with metadata
so the retriever can find the right tables for a user question.

Provides:
  • build_table_chunks(table_meta) → list of Chunk dicts
  • build_all_chunks(all_meta)     → list of all Chunk dicts

Usage:
    python -m indexing.chunk_builder
"""

import sys
import time

from indexing.semantic_description import get_semantic_description


class Chunk:
    """Simple container for a text chunk + its metadata."""

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk(type={self.metadata.get('chunk_type')}, table={self.metadata.get('table_name')}, len={len(self.text)})"


def _build_structural_chunk(table_meta: dict) -> Chunk:
    """
    Build a structural chunk from table metadata.
    This chunk contains exact column names, types, PKs — optimized
    for matching queries that mention specific column names.
    """
    table_name = table_meta["table_name"]
    columns = table_meta["columns"]
    primary_keys = table_meta.get("primary_keys", [])
    row_count = table_meta.get("row_count", 0)

    # Build column listing
    col_lines = []
    for c in columns:
        pk_marker = " [PK]" if c["name"] in primary_keys else ""
        col_lines.append(f"  - {c['name']} ({c['type']}){pk_marker}")

    col_text = "\n".join(col_lines)

    text = f"""Table: {table_name}
Columns:
{col_text}
Primary Keys: {', '.join(primary_keys) if primary_keys else 'None'}
Row Count: {row_count}"""

    # Add sample values if available
    sample_values = table_meta.get("sample_values", {})
    if sample_values:
        sample_lines = []
        for col_name, vals in list(sample_values.items())[:5]:
            sample_lines.append(f"  - {col_name}: {', '.join(str(v) for v in vals[:3])}")
        text += "\nSample Values:\n" + "\n".join(sample_lines)

    metadata = {
        "table_name": table_name,
        "chunk_type": "structural",
        "columns": [c["name"] for c in columns],
    }

    return Chunk(text=text, metadata=metadata)


def _build_semantic_chunk(table_meta: dict) -> Chunk:
    """
    Build a semantic chunk using the LLM-generated description.
    This chunk captures business meaning — optimized for matching
    queries that describe intent without mentioning column names.
    """
    table_name = table_meta["table_name"]

    # Get description (from cache or Gemini)
    description = get_semantic_description(table_meta)

    text = f"Table: {table_name}\n{description}"

    metadata = {
        "table_name": table_name,
        "chunk_type": "semantic",
        "columns": [c["name"] for c in table_meta["columns"]],
    }

    return Chunk(text=text, metadata=metadata)


def build_table_chunks(table_meta: dict) -> list[Chunk]:
    """
    Build both chunks (structural + semantic) for a single table.

    Args:
        table_meta: Dict from schema_extractor with keys:
            table_name, columns, primary_keys, row_count, sample_values

    Returns:
        List of 2 Chunk objects.
    """
    structural = _build_structural_chunk(table_meta)
    semantic = _build_semantic_chunk(table_meta)
    return [structural, semantic]


def build_all_chunks(all_table_meta: list[dict]) -> list[Chunk]:
    """
    Build chunks for ALL tables. Used during full index.

    Args:
        all_table_meta: List of table metadata dicts.

    Returns:
        List of Chunk objects (2 per table).
    """
    all_chunks = []
    total = len(all_table_meta)

    for i, meta in enumerate(all_table_meta, 1):
        try:
            chunks = build_table_chunks(meta)
            all_chunks.extend(chunks)
            print(f"  [{i}/{total}] [OK] {meta['table_name']} -> {len(chunks)} Chunks")
        except Exception as e:
            print(f"  [{i}/{total}] [FAIL] {meta['table_name']} -- Error: {e}")
            continue

    print(f"\n  Built {len(all_chunks)} chunks from {total} tables.")
    return all_chunks


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Testing chunk builder...\n")

    # Create fake table metadata
    test_meta = {
        "table_name": "dbo.Orders",
        "columns": [
            {"name": "OrderID", "type": "int"},
            {"name": "CustomerID", "type": "int"},
            {"name": "OrderDate", "type": "datetime"},
            {"name": "TotalAmount", "type": "decimal(10,2)"},
            {"name": "Status", "type": "varchar(20)"},
        ],
        "primary_keys": ["OrderID"],
        "row_count": 12500,
        "sample_values": {
            "Status": ["Shipped", "Pending", "Cancelled"],
            "TotalAmount": ["149.99", "32.50", "899.00"],
        },
    }

    # Test 1: Build chunks for a single table
    print("--- Test 1: Build chunks for one table ---")
    start = time.time()
    chunks = build_table_chunks(test_meta)
    elapsed = time.time() - start
    print(f"  Chunks   : {len(chunks)}")
    print(f"  Time     : {elapsed:.2f}s")

    for chunk in chunks:
        print(f"\n  --- {chunk.metadata['chunk_type'].upper()} CHUNK ---")
        print(f"  Table    : {chunk.metadata['table_name']}")
        print(f"  Columns  : {chunk.metadata['columns']}")
        # Show first 300 chars of text
        preview = chunk.text[:300].replace("\n", "\n  ")
        print(f"  Text     :\n  {preview}...")

    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert chunks[0].metadata["chunk_type"] == "structural"
    assert chunks[1].metadata["chunk_type"] == "semantic"
    print(f"\n  [OK] Correct: 2 chunks (structural + semantic)")

    # Test 2: Build chunks for multiple tables (batch)
    print("\n--- Test 2: Batch chunk building ---")
    test_meta_2 = {
        "table_name": "dbo.Customers",
        "columns": [
            {"name": "CustomerID", "type": "int"},
            {"name": "FullName", "type": "varchar(100)"},
            {"name": "Email", "type": "varchar(255)"},
            {"name": "City", "type": "varchar(50)"},
        ],
        "primary_keys": ["CustomerID"],
        "row_count": 3200,
        "sample_values": {
            "City": ["New York", "London", "Mumbai"],
        },
    }

    all_chunks = build_all_chunks([test_meta, test_meta_2])
    assert len(all_chunks) == 4, f"Expected 4 chunks, got {len(all_chunks)}"
    print(f"  [OK] Batch: 4 chunks from 2 tables")

    # Cleanup description cache from test
    from indexing.semantic_description import clear_cache
    print("\n--- Cleanup ---")
    clear_cache()

    print("\n" + "=" * 55)
    print("  [OK]  Chunk builder -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
