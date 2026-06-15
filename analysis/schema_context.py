"""
analysis/schema_context.py
───────────────────────────
Generates schema context for selected tables to insert into the LLM prompt.
"""

from typing import List, Dict, Any
from database.schema_manager import fetch_single_table

def infer_semantic_description(col_name: str, col_type: str = "", sample_values: list = None) -> str:
    """
    Dynamic fallback semantic inference — used only when the LLM-generated
    cache (get_column_descriptions) has no entry for this column yet.

    Uses generic patterns (suffixes, data types, value shapes) instead of
    a fixed keyword dictionary, so it generalizes to any domain.
    """
    c = col_name.lower().replace("_", " ").strip()
    sample_values = sample_values or []

    # ── Type-based inference (works for ANY domain) ──────────────────────────
    type_lower = (col_type or "").lower()

    if "date" in type_lower or "time" in type_lower:
        return "TEMPORAL_VALUE - represents a date or timestamp"

    # ── Naming-pattern inference (suffix/prefix based, domain-agnostic) ───────
    if c.endswith((" id", " key", " code")) or c in ("id", "key", "code"):
        return "IDENTIFIER - unique reference value, not used for aggregation"

    if c.endswith((" name", " title")):
        return "LABEL - descriptive name or title field"

    if c.startswith(("is ", "has ")) or c.endswith((" flag", " indicator")):
        return "BOOLEAN_FLAG - represents a yes/no or true/false condition"

    if any(c.endswith(suffix) for suffix in [
        " amount", " cost", " price", " value", " usd", " revenue", " expense", " salary", " pay", " income", " compensation", " wage", " earning"
    ]) or c in ["salary", "pay", "income", "compensation", "wage", "earning"]:
        return "MONETARY_VALUE - represents a financial/monetary amount"

    if any(c.endswith(suffix) for suffix in [
        " rate", " ratio", " percent", " percentage", " score", " level", " quality", " probability"
    ]) or c in ["rate", "ratio", "percent", "percentage", "score", "level"]:
        return "METRIC_SCORE - represents a calculated rate, ratio, score, or level"

    if any(c.endswith(suffix) for suffix in [
        " count", " quantity", " number", " total", " qty", " sum"
    ]) or c in ["count", "quantity", "number", "total", "qty", "sum"]:
        return "COUNT_METRIC - represents a quantity or count"

    # ── Value-shape inference (sample-based, domain-agnostic) ─────────────────
    if sample_values:
        non_null = [v for v in sample_values if v is not None]
        if non_null:
            try:
                unique_ratio = len(set(map(str, non_null))) / len(non_null)
                # Low cardinality + short strings → likely a category
                if unique_ratio < 0.5 and all(len(str(v)) < 30 for v in non_null):
                    return "CATEGORY - represents a classification or grouping value"
            except Exception:
                pass

    return ""


def generate_schema_context(user_query: str, focus_tables: List[str] = None) -> str:
    """
    Builds LLM-friendly schema text for retrieved tables.
    """
    if not focus_tables:
        return "No relevant tables identified."

    from indexing.semantic_description import get_column_descriptions

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
            
        sample_vals = meta.get("sample_values", {}) or {}
        lines.append("Columns:")
        cached_cols = get_column_descriptions(meta['table_name'])
        for col in meta["columns"]:
            pk_marker = " [PK]" if col["name"] in pk_list else ""
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            
            # Try dynamic cache first, then fall back to static rules
            sem_desc = cached_cols.get(col["name"])
            if not sem_desc:
                col_samples = sample_vals.get(col["name"], []) if isinstance(sample_vals, dict) else []
                sem_desc = infer_semantic_description(
                    col["name"],
                    col_type=col.get("display_type", col.get("type", "")),
                    sample_values=col_samples
                )
                
            sem_str = f" [Semantic: {sem_desc}]" if sem_desc else ""
            lines.append(f"  - {col['name']} ({col['display_type']}, {nullable}){pk_marker}{sem_str}")
            
        if sample_vals:
            lines.append("Sample Values:")
            for col_name, vals in sample_vals.items():
                if isinstance(vals, list):
                    short_vals = []
                    for v in vals[:3]:
                        v_str = str(v)
                        if len(v_str) > 50:
                            v_str = v_str[:47] + "..."
                        short_vals.append(v_str)
                    lines.append(f"  - {col_name}: {short_vals}")
                else:
                    v_str = str(vals)
                    if len(v_str) > 50:
                        v_str = v_str[:47] + "..."
                    lines.append(f"  - {col_name}: {v_str}")
                
        lines.append("") # blank line between tables

    return "\n".join(lines)


def get_all_table_summaries() -> str:
    """
    Builds a summary of all tables and columns in the database for high-level description prompts.
    """
    from database.schema_manager import fetch_database_metadata
    from indexing.semantic_description import get_column_descriptions

    metadata_list = fetch_database_metadata()
    if not metadata_list:
        return "No tables available in the database."

    lines = []
    for meta in metadata_list:
        lines.append(f"Table: {meta['table_name']}")
        lines.append(f"Approximate Row Count: {meta['row_count']}")
        sample_vals = meta.get("sample_values", {}) or {}
        lines.append("Columns:")
        cached_cols = get_column_descriptions(meta['table_name'])
        for col in meta["columns"]:
            pk_marker = " [PK]" if col["name"] in meta.get("primary_keys", []) else ""
            
            # Try dynamic cache first, then fall back to static rules
            sem_desc = cached_cols.get(col["name"])
            if not sem_desc:
                col_samples = sample_vals.get(col["name"], []) if isinstance(sample_vals, dict) else []
                sem_desc = infer_semantic_description(
                    col["name"],
                    col_type=col.get("display_type", col.get("type", "")),
                    sample_values=col_samples
                )
                
            sem_str = f" [Semantic: {sem_desc}]" if sem_desc else ""
            lines.append(f"  - {col['name']} ({col['display_type']}){pk_marker}{sem_str}")
        lines.append("") # blank line between tables
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n--- Testing Schema Context Builder ---")
    test_tables = ["dbo.csv_departments", "dbo.csv_employees"]
    ctx = generate_schema_context("Show department managers", test_tables)
    print(ctx)
    print("\n--- Testing All Table Summaries ---")
    print(get_all_table_summaries())
