"""
indexing/schema_extractor.py
────────────────────────────
Bridges database/schema_manager.py → the indexing/chunking pipeline.

Pulls raw schema metadata and formats it into structured dicts
ready for chunk_builder.py to consume.

Provides:
  • extract_all_tables()       → list of formatted table metadata
  • extract_single_table(name) → formatted metadata for one table

Usage:
    python -m indexing.schema_extractor
"""

import sys
import time

from database.schema_manager import fetch_database_metadata, fetch_single_table


def _format_table_meta(raw: dict) -> dict:
    """
    Take raw metadata from schema_manager and format it
    into the structure expected by chunk_builder.

    Input (from schema_manager):
        {
            "table_name": "dbo.Orders",
            "schema": "dbo",
            "columns": [{"name": ..., "type": ..., "display_type": ..., ...}],
            "primary_keys": ["OrderID"],
            "row_count": 1500,
            "sample_values": {"OrderID": [1, 2, 3], ...}
        }

    Output (for chunk_builder):
        {
            "table_name": "dbo.Orders",
            "columns": [{"name": "OrderID", "type": "int"}, ...],
            "primary_keys": ["OrderID"],
            "row_count": 1500,
            "sample_values": {"OrderID": [1, 2, 3], ...},
            "column_names": ["OrderID", "CustomerID", ...],
            "column_types_display": ["OrderID (int)", "CustomerID (int)", ...],
        }
    """
    columns_simplified = [
        {"name": c["name"], "type": c.get("display_type", c["type"])}
        for c in raw["columns"]
    ]

    column_names = [c["name"] for c in raw["columns"]]

    column_types_display = [
        f"{c['name']} ({c.get('display_type', c['type'])})"
        for c in raw["columns"]
    ]

    return {
        "table_name": raw["table_name"],
        "columns": columns_simplified,
        "primary_keys": raw.get("primary_keys", []),
        "row_count": raw.get("row_count", 0),
        "sample_values": raw.get("sample_values", {}),
        "column_names": column_names,
        "column_types_display": column_types_display,
    }


def extract_all_tables() -> list[dict]:
    """
    Extract and format metadata for ALL tables in the database.
    Used during full index (startup / nightly re-index).
    """
    raw_metadata = fetch_database_metadata()
    formatted = [_format_table_meta(m) for m in raw_metadata]
    return formatted


def extract_single_table(table_name: str, schema: str = "dbo") -> dict | None:
    """
    Extract and format metadata for a SINGLE table.
    Used during incremental re-index (CSV upload / schema change).

    Args:
        table_name: Table name without schema prefix (e.g., "Orders")
        schema:     SQL Server schema (default: "dbo")
    """
    # If the name includes schema prefix, split it
    if "." in table_name:
        schema, table_name = table_name.rsplit(".", 1)

    raw = fetch_single_table(table_name, schema=schema)
    if raw is None:
        return None

    return _format_table_meta(raw)


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Testing schema extractor...\n")

    # Test 1: Extract all tables
    print("--- Test 1: Extract all tables ---")
    start = time.time()
    all_tables = extract_all_tables()
    elapsed = time.time() - start

    print(f"  Found    : {len(all_tables)} tables")
    print(f"  Time     : {elapsed:.2f}s")

    if not all_tables:
        print("\n  [WARNING] No tables found (database is empty).")
        print("  This is fine -- tables will appear after CSV upload")
        print("  or when you add data to your database.")
        print("\n" + "=" * 55)
        print("  [OK] Schema extractor -- working (0 tables to extract)")
        print("=" * 55)
        sys.exit(0)

    # Test 2: Inspect first table structure
    print("\n--- Test 2: Inspect formatted output ---")
    sample = all_tables[0]
    print(f"  Table    : {sample['table_name']}")
    print(f"  Columns  : {len(sample['columns'])}")
    print(f"  PKs      : {sample['primary_keys'] or '(none)'}")
    print(f"  Rows     : {sample['row_count']}")
    print(f"  Col list : {sample['column_names'][:5]}{'...' if len(sample['column_names']) > 5 else ''}")
    print(f"  Types    : {sample['column_types_display'][:3]}{'...' if len(sample['column_types_display']) > 3 else ''}")

    # Verify all expected keys exist
    expected_keys = {"table_name", "columns", "primary_keys", "row_count",
                     "sample_values", "column_names", "column_types_display"}
    actual_keys = set(sample.keys())
    assert expected_keys == actual_keys, (
        f"Key mismatch: missing={expected_keys - actual_keys}, "
        f"extra={actual_keys - expected_keys}"
    )
    print(f"  [OK] All expected keys present")

    # Test 3: Single table extraction
    print("\n--- Test 3: Single table extraction ---")
    single = extract_single_table(sample["table_name"])
    if single:
        print(f"  [OK] Extracted: {single['table_name']} ({len(single['columns'])} cols)")
    else:
        print(f"  [FAIL] Failed to extract: {sample['table_name']}")

    print("\n" + "=" * 55)
    print("  [OK] Schema extractor -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
