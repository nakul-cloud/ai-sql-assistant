"""
database/csv_uploader.py
────────────────────────
Handles CSV file parsing, validation, and upload to SQL Server.

Provides:
  • process_csv(file_path)                → parse & validate CSV, return DataFrame
  • upload_df_to_sql(df, table_name, ...) → write DataFrame to SQL Server table
  • upload_csv_and_index(file_path, ...)   → full pipeline (parse → SQL → index)

The indexing step (Qdrant re-index) is wired in later once the
indexing pipeline is built. For now it's a placeholder.

Usage:
    python -m database.csv_uploader
"""

import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv

from database.sql_server import get_engine

load_dotenv()

# From .env
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")


def _sanitize_table_name(name: str) -> str:
    """
    Clean a user-provided table name so it's safe for SQL Server.
    - Strips file extensions
    - Replaces spaces/special chars with underscores
    - Lowercases
    - Prefixes with 'csv_' to distinguish uploaded tables
    """
    # Remove file extension if present
    name = os.path.splitext(name)[0]
    # Replace non-alphanumeric chars (except underscores) with underscore
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_")
    # Lowercase
    name = name.lower()
    # Prefix with csv_ if not already
    if not name.startswith("csv_"):
        name = f"csv_{name}"
    return name


def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to be SQL Server-safe.
    - Strip whitespace
    - Replace spaces/special chars with underscores
    - Ensure no duplicate column names
    """
    new_cols = []
    seen = {}

    for col in df.columns:
        clean = str(col).strip()
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")

        if not clean:
            clean = "column"

        # Handle duplicates by appending _2, _3, etc.
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 1

        new_cols.append(clean)

    df.columns = new_cols
    return df


def process_csv(file_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Parse and validate a CSV file.

    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    # ── Check file exists ────────────────────────────────────────
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    # ── Check file size ──────────────────────────────────────────
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return None, (
            f"File too large: {size_mb:.1f} MB "
            f"(max: {MAX_UPLOAD_SIZE_MB} MB)"
        )

    # ── Parse CSV ────────────────────────────────────────────────
    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")
    except Exception as e:
        return None, f"Failed to parse CSV: {e}"

    # ── Validate ─────────────────────────────────────────────────
    if df.empty:
        return None, "CSV file is empty (no rows)."

    if len(df.columns) == 0:
        return None, "CSV file has no columns."

    # ── Clean column names ───────────────────────────────────────
    df = _sanitize_column_names(df)

    return df, None


def upload_df_to_sql(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    schema: str = "dbo",
) -> tuple[bool, str]:
    """
    Write a DataFrame to a SQL Server table.

    Args:
        df:         The DataFrame to upload.
        table_name: Target table name (will be sanitized).
        if_exists:  'replace' (drop & recreate) or 'append' (add rows).
        schema:     SQL Server schema (default: dbo).

    Returns:
        (True, success_message) or (False, error_message)
    """
    table_name = _sanitize_table_name(table_name)
    engine = get_engine()

    try:
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=1000,     # batch inserts for performance
        )

        row_count = len(df)
        col_count = len(df.columns)
        return True, (
            f"Uploaded {row_count} rows x {col_count} columns "
            f"to [{schema}].[{table_name}]"
        )

    except Exception as e:
        return False, f"SQL upload failed: {e}"


def upload_csv_and_index(
    file_path: str,
    table_name: str,
    if_exists: str = "replace",
) -> tuple[bool, str]:
    """
    Full CSV pipeline: parse → upload to SQL → trigger Qdrant re-index.

    This is the function called from the Streamlit upload page.
    """
    # Step 1: Parse CSV
    df, err = process_csv(file_path)
    if err:
        return False, err

    # Step 2: Upload to SQL Server
    success, msg = upload_df_to_sql(df, table_name, if_exists=if_exists)
    if not success:
        return False, msg

    sanitized = _sanitize_table_name(table_name)

    # Step 3: Trigger incremental Qdrant re-index
    from indexing.index_manager import index_single_table
    index_single_table("dbo", sanitized)

    # Step 4: Bust schema metadata cache
    from database.schema_manager import fetch_database_metadata
    fetch_database_metadata(force_refresh=True)

    return True, f"[OK] {msg} — table '{sanitized}' is ready to query!"


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFO] Testing CSV uploader...\n")

    # Create a small test CSV in memory
    test_csv_path = os.path.join(UPLOAD_DIR, "_test_upload.csv")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Generate sample data
    test_df = pd.DataFrame({
        "Employee Name": ["Alice Johnson", "Bob Smith", "Charlie Brown"],
        "Department": ["Engineering", "Sales", "Marketing"],
        "Salary": [95000, 72000, 68000],
        "Hire Date": ["2023-01-15", "2022-06-01", "2024-03-10"],
    })
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Created test CSV: {test_csv_path}")

    # Step 1: Test parsing
    print("\n--- Step 1: Parsing CSV ---")
    df, err = process_csv(test_csv_path)
    if err:
        print(f"  [FAIL] Parse failed: {err}")
        sys.exit(1)
    print(f"  [OK] Parsed: {len(df)} rows x {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Step 2: Test upload to SQL Server
    print("\n--- Step 2: Uploading to SQL Server ---")
    success, msg = upload_df_to_sql(df, "_test_upload", if_exists="replace")
    if success:
        print(f"  [OK] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        sys.exit(1)

    # Step 3: Verify by reading back
    print("\n--- Step 3: Verifying upload ---")
    from sqlalchemy import text
    engine = get_engine()
    sanitized_name = _sanitize_table_name("_test_upload")
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM [dbo].[{sanitized_name}]"))
        count = result.fetchone()[0]
        print(f"  [OK] Read back {count} rows from [dbo].[{sanitized_name}]")

    # Step 4: Clean up test table
    print("\n--- Step 4: Cleaning up ---")
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS [dbo].[{sanitized_name}]"))
        conn.commit()
        print(f"  [OK] Dropped test table [{sanitized_name}]")

    # Clean up test CSV
    os.remove(test_csv_path)
    print(f"  [OK] Deleted test CSV: {test_csv_path}")

    print("\n" + "=" * 55)
    print("  [OK] CSV uploader -- all tests passed!")
    print("=" * 55)
    sys.exit(0)
