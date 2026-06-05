"""
database/schema_manager.py
──────────────────────────
Extracts schema metadata from SQL Server for the indexing pipeline.

Provides:
  • fetch_database_metadata()  → list of table metadata dicts
  • fetch_single_table()       → metadata for one specific table

Each table dict has:
  {
    "table_name":   "dbo.Orders",
    "schema":       "dbo",
    "columns":      [{"name": "OrderID", "type": "int", "nullable": False}, ...],
    "primary_keys": ["OrderID"],
    "row_count":    1500,
    "sample_values": {"OrderID": [1, 2, 3], "Status": ["Open", "Closed", "Pending"]}
  }

Usage:
    python -m database.schema_manager
"""

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import text

from database.sql_server import get_engine

load_dotenv()

# How many sample values to pull per column (for LLM context)
SAMPLE_SIZE = 5


def _get_all_table_names(conn) -> list[dict]:
    """Return all user tables with their schema names."""
    query = text("""
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """)
    rows = conn.execute(query).fetchall()
    return [{"schema": r[0], "table_name": r[1]} for r in rows]


def _get_columns(conn, schema: str, table: str) -> list[dict]:
    """Return column metadata for a specific table."""
    query = text("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            CHARACTER_MAXIMUM_LENGTH,
            COLUMN_DEFAULT,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
    """)
    rows = conn.execute(query, {"schema": schema, "table": table}).fetchall()

    columns = []
    for r in rows:
        col = {
            "name": r[0],
            "type": r[1],
            "nullable": r[2] == "YES",
            "max_length": r[3],
            "default": r[4],
            "position": r[5],
        }
        # Build a display type like "varchar(255)" or "int"
        if r[3] and r[1] in ("varchar", "nvarchar", "char", "nchar"):
            length = "MAX" if r[3] == -1 else str(r[3])
            col["display_type"] = f"{r[1]}({length})"
        else:
            col["display_type"] = r[1]

        columns.append(col)

    return columns


def _get_primary_keys(conn, schema: str, table: str) -> list[str]:
    """Return primary key column names for a specific table."""
    query = text("""
        SELECT ccu.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
            ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
            AND tc.TABLE_SCHEMA = ccu.TABLE_SCHEMA
        WHERE tc.TABLE_SCHEMA = :schema
            AND tc.TABLE_NAME = :table
            AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY ccu.COLUMN_NAME
    """)
    rows = conn.execute(query, {"schema": schema, "table": table}).fetchall()
    return [r[0] for r in rows]


def _get_row_count(conn, schema: str, table: str) -> int:
    """Return approximate row count using system metadata (fast, no table scan)."""
    query = text("""
        SELECT SUM(p.rows)
        FROM sys.partitions p
        JOIN sys.tables t ON p.object_id = t.object_id
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name = :schema
            AND t.name = :table
            AND p.index_id IN (0, 1)
    """)
    result = conn.execute(query, {"schema": schema, "table": table}).fetchone()
    return int(result[0]) if result and result[0] else 0


def _get_sample_values(conn, schema: str, table: str, columns: list[dict]) -> dict:
    """
    Return a few distinct sample values per column for LLM context.
    Uses TOP N with DISTINCT to get representative values.
    """
    sample_values = {}
    quoted_table = f"[{schema}].[{table}]"

    for col in columns:
        col_name = col["name"]
        try:
            # Use dynamic SQL safely — schema/table/column are from INFORMATION_SCHEMA
            query = text(
                f"SELECT DISTINCT TOP {SAMPLE_SIZE} [{col_name}] "
                f"FROM {quoted_table} "
                f"WHERE [{col_name}] IS NOT NULL"
            )
            rows = conn.execute(query).fetchall()
            values = [str(r[0]) for r in rows]
            if values:
                sample_values[col_name] = values
        except Exception:
            # Skip columns that can't be sampled (e.g., image, xml types)
            continue

    return sample_values


def fetch_single_table(table_name: str, schema: str = "dbo") -> dict | None:
    """
    Fetch metadata for a single table.
    Returns a metadata dict or None if the table doesn't exist.
    """
    engine = get_engine()

    with engine.connect() as conn:
        columns = _get_columns(conn, schema, table_name)
        if not columns:
            return None

        primary_keys = _get_primary_keys(conn, schema, table_name)
        row_count = _get_row_count(conn, schema, table_name)
        sample_values = _get_sample_values(conn, schema, table_name, columns)

        return {
            "table_name": f"{schema}.{table_name}",
            "schema": schema,
            "columns": columns,
            "primary_keys": primary_keys,
            "row_count": row_count,
            "sample_values": sample_values,
        }


def fetch_database_metadata(force_refresh: bool = False) -> list[dict]:
    """
    Fetch metadata for ALL user tables in the database.

    Args:
        force_refresh: If True, bypasses any caching. (Caching will be
                       added at the Streamlit layer via st.cache_data later.)

    Returns:
        List of table metadata dicts.
    """
    engine = get_engine()
    metadata_list = []

    with engine.connect() as conn:
        tables = _get_all_table_names(conn)

        print(f"[INFO] Found {len(tables)} tables in database.")

        for i, tbl in enumerate(tables, 1):
            schema = tbl["schema"]
            table = tbl["table_name"]
            full_name = f"{schema}.{table}"

            try:
                columns = _get_columns(conn, schema, table)
                primary_keys = _get_primary_keys(conn, schema, table)
                row_count = _get_row_count(conn, schema, table)
                sample_values = _get_sample_values(conn, schema, table, columns)

                metadata_list.append({
                    "table_name": full_name,
                    "schema": schema,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "row_count": row_count,
                    "sample_values": sample_values,
                })

                print(f"  [{i}/{len(tables)}] [OK] {full_name} -- "
                      f"{len(columns)} cols, {row_count} rows")

            except Exception as e:
                print(f"  [{i}/{len(tables)}] [FAIL] {full_name} -- Error: {e}")
                continue

    print(f"\n[OK] Extracted metadata for {len(metadata_list)}/{len(tables)} tables.")
    return metadata_list


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Extracting database schema metadata...\n")

    metadata = fetch_database_metadata()

    if not metadata:
        print("\n[WARNING] No tables found. Is the database empty?")
        print("   You may need to create some tables first.")
        sys.exit(1)

    # Print a summary of the first table as a sample
    print("\n" + "=" * 60)
    print("Sample -- first table metadata:")
    print("=" * 60)

    sample = metadata[0]
    print(f"  Table        : {sample['table_name']}")
    print(f"  Columns      : {len(sample['columns'])}")
    print(f"  Primary Keys : {sample['primary_keys'] or '(none)'}")
    print(f"  Row Count    : {sample['row_count']}")
    print(f"  Columns list :")
    for col in sample["columns"]:
        pk_marker = " [PK]" if col["name"] in sample["primary_keys"] else ""
        nullable = "NULL" if col["nullable"] else "NOT NULL"
        print(f"    - {col['name']} ({col['display_type']}, {nullable}){pk_marker}")

    if sample["sample_values"]:
        print(f"  Sample values:")
        for col_name, vals in list(sample["sample_values"].items())[:3]:
            print(f"    - {col_name}: {vals}")

    print("\n[OK] Schema extraction complete.")
    sys.exit(0)
