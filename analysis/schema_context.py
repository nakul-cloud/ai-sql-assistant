"""
analysis/schema_context.py
───────────────────────────
Generates schema context for selected tables to insert into the LLM prompt.
"""

from typing import List, Dict, Any
from database.schema_manager import fetch_single_table

def generate_schema_context(user_query: str, focus_tables: List[str] = None) -> str:
    """
    Builds LLM-friendly schema text for retrieved tables.
    """
    if not focus_tables:
        return "No relevant tables identified."

    lines = []
    lines.append("Here is the relevant database schema metadata for the query:")
    lines.append("")

    for table in focus_tables:
        schema = "dbo"
        table_name = table
        if "." in table:
            schema, table_name = table.split(".", 1)
        
        meta = fetch_single_table(table_name, schema)
        if not meta:
            continue
            
        lines.append(f"Table: {meta['table_name']}")
        lines.append(f"Row Count: {meta['row_count']}")
        
        pk_list = meta.get("primary_keys", [])
        if pk_list:
            lines.append(f"Primary Keys: {', '.join(pk_list)}")
            
        lines.append("Columns:")
        for col in meta["columns"]:
            pk_marker = " [PK]" if col["name"] in pk_list else ""
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            lines.append(f"  - {col['name']} ({col['display_type']}, {nullable}){pk_marker}")
            
        sample_vals = meta.get("sample_values", {})
        if sample_vals:
            lines.append("Sample Values:")
            for col_name, vals in sample_vals.items():
                lines.append(f"  - {col_name}: {vals}")
                
        lines.append("") # blank line between tables

    return "\n".join(lines)


def get_all_table_summaries() -> str:
    """
    Builds a summary of all tables and columns in the database for high-level description prompts.
    """
    from database.schema_manager import fetch_database_metadata
    metadata_list = fetch_database_metadata()
    if not metadata_list:
        return "No tables available in the database."

    lines = []
    for meta in metadata_list:
        lines.append(f"Table: {meta['table_name']}")
        lines.append(f"Approximate Row Count: {meta['row_count']}")
        lines.append("Columns:")
        for col in meta["columns"]:
            pk_marker = " [PK]" if col["name"] in meta.get("primary_keys", []) else ""
            lines.append(f"  - {col['name']} ({col['display_type']}){pk_marker}")
        lines.append("") # blank line between tables
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n--- Testing Schema Context Builder ---")
    test_tables = ["dbo.csv_departments", "dbo.csv_employees"]
    ctx = generate_schema_context("Show department managers", test_tables)
    print(ctx)
    print("\n--- Testing All Table Summaries ---")
    print(get_all_table_summaries())
