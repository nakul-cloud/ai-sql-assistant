"""
indexing/index_manager.py
─────────────────────────
The master orchestrator for the database schema indexing pipeline.

Coordinates:
  1. schema_manager     → Pull raw metadata from SQL Server
  2. schema_extractor   → Format it for indexing
  3. chunk_builder      → Build structural + semantic chunks (using LLM descriptions)
  4. qdrant_uploader    → Upsert chunks to Qdrant (with automatic deletion of old table chunks)

Provides:
  • index_all_tables()         → Run full DB schema index
  • index_single_table()       → Safely index/re-index a specific table

Usage:
    python -m indexing.index_manager
"""

import sys
import time

# Import embedder first to prevent Windows OpenMP/networking library conflicts
from indexing.embedder import get_model, embed_text

# Warm up PyTorch & embedding model BEFORE loading SQL Server drivers / pyodbc
print("[INFO] Warming up local embedding model to avoid runtime thread conflicts...")
_ = embed_text("warmup")

from indexing.chunk_builder import build_all_chunks, build_table_chunks
from indexing.qdrant_uploader import (
    delete_table_chunks,
    ensure_collection,
    get_collection_info,
    upload_chunks,
)
from indexing.schema_extractor import extract_all_tables, extract_single_table


def index_single_table(schema_name: str, table_name: str) -> bool:
    """
    Extract, chunk, embed, and upload a single table schema.
    Safely deletes any existing chunks for this table in Qdrant first.

    Args:
        schema_name: Database schema (e.g. 'dbo')
        table_name: Table name (e.g. 'Customers')

    Returns:
        True if indexing succeeded, False otherwise.
    """
    full_table_name = f"{schema_name}.{table_name}"
    print(f"\n[INFO] Starting indexing for table: {full_table_name}")
    start_time = time.time()

    try:
        # 1. Fetch and format metadata directly
        formatted_meta = extract_single_table(table_name, schema=schema_name)
        if not formatted_meta:
            print(f"  [FAIL] Table {full_table_name} not found in database.")
            return False

        # 2. Build chunks (triggers LLM description with cache verification)
        chunks = build_table_chunks(formatted_meta)
        print(f"  [OK] Built {len(chunks)} chunks (structural + semantic)")

        # 3. Remove existing chunks for this table from Qdrant (idempotency)
        deleted_count = delete_table_chunks(full_table_name)
        if deleted_count > 0:
            print(f"  [INFO] Cleared {deleted_count} stale points from Qdrant.")

        # 4. Upload new chunks
        upserted_count = upload_chunks(chunks)
        elapsed = time.time() - start_time
        print(f"  [OK] Uploaded {upserted_count} points to Qdrant in {elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"  [FAIL] Failed to index {full_table_name}: {e}")
        return False


def index_all_tables() -> dict:
    """
    Run the full indexing pipeline for the entire database.
    Extracts, chunks, embeds, and uploads all active user tables.

    Returns:
        Summary dictionary with counts of tables processed, chunks built, and status.
    """
    print("\n=======================================================")
    print("      [INFO] Starting Full Database Indexing Job")
    print("=======================================================")
    start_time = time.time()

    # Ensure collection exists
    ensure_collection()

    summary = {
        "status": "success",
        "tables_processed": 0,
        "chunks_built": 0,
        "points_upserted": 0,
        "errors": [],
    }

    try:
        # 1. Fetch and format all tables
        print("[INFO] Fetching and formatting schema metadata...")
        formatted_db_meta = extract_all_tables()
        if not formatted_db_meta:
            print("  [WARNING] No tables found in database. Indexing aborted.")
            summary["status"] = "no_tables"
            return summary

        total_tables = len(formatted_db_meta)
        print(f"  [OK] Found {total_tables} tables to index.")

        # 2. Process each table individually (safely isolates errors per table)
        all_chunks = []
        for i, table_meta in enumerate(formatted_db_meta, 1):
            table_name = table_meta["table_name"]
            print(f"\n  [{i}/{total_tables}] Processing: {table_name}")

            try:
                # Build chunks (using LLM description with cache verification)
                table_chunks = build_table_chunks(table_meta)
                all_chunks.extend(table_chunks)
                summary["tables_processed"] += 1

                # Safely delete existing points for this table from Qdrant
                delete_table_chunks(table_name)

            except Exception as table_err:
                err_msg = f"Failed to build chunks for {table_name}: {table_err}"
                print(f"    [FAIL] {err_msg}")
                summary["errors"].append(err_msg)

        summary["chunks_built"] = len(all_chunks)
        if not all_chunks:
            print("\n[WARNING] No chunks built. Nothing to upload.")
            return summary

        # 3. Upload all chunks to Qdrant
        print(f"\n[INFO] Embedding and uploading {len(all_chunks)} chunks to Qdrant...")
        upserted = upload_chunks(all_chunks)
        summary["points_upserted"] = upserted

        elapsed = time.time() - start_time
        print("\n=======================================================")
        print("      [OK] Database Indexing Completed Successfully")
        print(f"      Tables: {summary['tables_processed']}/{total_tables}")
        print(f"      Points: {summary['points_upserted']} upserted")
        print(f"      Time  : {elapsed:.2f}s")
        print("=======================================================")

    except Exception as e:
        print(f"\n[CRITICAL] Full database indexing job failed: {e}")
        summary["status"] = "failed"
        summary["errors"].append(str(e))

    return summary


# ── Standalone test / execution ──────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing index manager ---")

    # Fetch collection stats before running
    info_before = get_collection_info()
    print(f"  Collection Status : {info_before.get('status', 'N/A')}")
    print(f"  Points Before     : {info_before['points_count']}")

    # Run full database indexing
    result = index_all_tables()

    # Fetch collection stats after running
    info_after = get_collection_info()
    print(f"\n  Points After      : {info_after['points_count']}")
    print(f"  Job Status        : {result['status'].upper()}")
    print("-----------------------------")
    sys.exit(0)
