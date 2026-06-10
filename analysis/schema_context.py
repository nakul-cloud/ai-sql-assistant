"""
analysis/schema_context.py
───────────────────────────
Generates schema context for selected tables to insert into the LLM prompt.
"""

from typing import List, Dict, Any
from database.schema_manager import fetch_single_table

def infer_semantic_description(col_name: str) -> str:
    c = col_name.lower().replace("_", " ").strip()
    
    # Salary / Money
    if any(k in c for k in ["salary", "pay", "income", "compensation", "wage", "earning"]):
        return "SALARY_METRIC - represents monetary pay/compensation"
        
    # Job Title / Role
    if any(k == c for k in ["job title", "role", "title", "position", "occupation", "job"]):
        return "ROLE_NAME/JOB_TITLE - represents the specific occupation or job title"
    if any(k in c for k in ["job title", "role name", "occupation title"]):
        return "ROLE_NAME/JOB_TITLE - represents the specific occupation or job title"
        
    # Seniority Level
    if any(k in c for k in ["seniority", "experience level", "experience tier", "experience group", "seniority level", "level"]):
        if "education" not in c and "skill" not in c:
            return "SENIORITY_LEVEL - represents experience tier (e.g. Senior, Lead, Junior, Entry)"
            
    # Temporal / Date
    if any(k in c for k in ["date", "year", "month", "time", "timestamp", "created at", "updated at"]):
        return "TEMPORAL_VALUE - represents date, year, or timestamp"
        
    # Industry
    if any(k in c for k in ["industry", "sector", "field of work", "business area"]):
        return "INDUSTRY_CATEGORY - represents the industry sector or business area"
        
    # Location
    if any(k in c for k in ["country", "location", "region", "state", "city", "hq", "headquarters"]):
        return "LOCATION_CATEGORY - represents geographical country or region"
        
    # Company
    if any(k in c for k in ["company", "organization", "employer", "firm"]):
        return "COMPANY_NAME - represents company or organization name"
        
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
            
        lines.append("Columns:")
        cached_cols = get_column_descriptions(meta['table_name'])
        for col in meta["columns"]:
            pk_marker = " [PK]" if col["name"] in pk_list else ""
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            
            # Try dynamic cache first, then fall back to static rules
            sem_desc = cached_cols.get(col["name"])
            if not sem_desc:
                sem_desc = infer_semantic_description(col["name"])
                
            sem_str = f" [Semantic: {sem_desc}]" if sem_desc else ""
            lines.append(f"  - {col['name']} ({col['display_type']}, {nullable}){pk_marker}{sem_str}")
            
        sample_vals = meta.get("sample_values", {})
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
        lines.append("Columns:")
        cached_cols = get_column_descriptions(meta['table_name'])
        for col in meta["columns"]:
            pk_marker = " [PK]" if col["name"] in meta.get("primary_keys", []) else ""
            
            # Try dynamic cache first, then fall back to static rules
            sem_desc = cached_cols.get(col["name"])
            if not sem_desc:
                sem_desc = infer_semantic_description(col["name"])
                
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
