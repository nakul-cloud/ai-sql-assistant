"""
analysis/result_enricher.py
───────────────────────────
Semantic enrichment layer for SQL results.
Instead of sending 1000+ raw records to the LLM (which is slow and token-heavy),
this layer analyzes the dataset using pandas to generate a lightweight statistical profile 
and metadata summary. This allows the LLM to easily generate natural language insights 
without the latency of processing thousands of rows.
"""

import pandas as pd
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

def enrich_sql_result(query_result: Dict[str, Any], sql_query: str = None) -> Dict[str, Any]:
    """
    Enriches the raw SQL query result with semantic insights and statistical summaries.
    Reduces payload size to speed up LLM processing and prevent token overflow.
    """
    logger.info("Applying semantic enrichment to query results.")
    
    rows = query_result.get("rows", [])
    if not rows:
        return {
            "total_rows": 0,
            "table_total_rows": 0,
            "is_truncated": False,
            "is_count_query": False,
            "summary": "The query returned no results.",
            "columns": query_result.get("columns", []),
            "column_stats": {},
            "data_sample": []
        }

    df = pd.DataFrame(rows)
    row_count = len(df)

    # ── Detect query type from SQL ────────────────────────────────────────────
    sql_upper = (sql_query or "").upper()

    # Is this a COUNT/SUM/AVG/MAX/MIN aggregation with no GROUP BY?
    # Single-row result where the row IS the answer, not a sample
    AGG_FUNCTIONS = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
    has_agg = any(fn in sql_upper for fn in AGG_FUNCTIONS)
    has_group_by = "GROUP BY" in sql_upper
    has_top = bool(re.search(r"\bTOP\s+\d+\b", sql_upper))

    # Pure aggregation = single aggregate, no GROUP BY
    # e.g. SELECT COUNT(*) → 1 row, that row IS the full answer
    is_count_query = has_agg and not has_group_by and row_count == 1

    # Truncated = TOP N was used OR result has exactly N rows suggesting a limit
    is_truncated = has_top and not is_count_query

    # ── Get actual table row count for truncated results ──────────────────────
    table_total_rows = None
    if is_truncated and sql_query:
        try:
            match = re.search(r"FROM\s+([\[\]`\w\.]+)", sql_query, re.IGNORECASE)
            if match:
                table_name = match.group(1).strip()
                from database.sql_server import get_engine
                from sqlalchemy import text
                engine = get_engine()
                with engine.connect() as conn:
                    res = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    ).fetchone()
                    if res and res[0] is not None:
                        table_total_rows = int(res[0])
        except Exception as e:
            logger.warning(f"Could not fetch table total rows: {e}")

    # Dynamic sample limit: send more rows (up to 150) for GROUP BY or small datasets
    sample_limit = 150 if (has_group_by or row_count <= 150) else 10

    # ── Build enrichment ──────────────────────────────────────────────────────
    enrichment = {
        "total_rows": row_count,                          # rows in THIS result
        "table_total_rows": table_total_rows,             # rows in full table (if fetched)
        "is_truncated": is_truncated,                     # True if TOP N was used
        "is_count_query": is_count_query,                 # True if result IS the aggregate answer
        "columns": query_result.get("columns", []),
        "column_stats": {},
        "data_sample": df.head(sample_limit).to_dict(orient="records")
    }

    # Numeric stats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        col_series = df[col].dropna()
        enrichment["column_stats"][col] = {
            "sum": float(col_series.sum()) if not col_series.empty else 0.0,
            "mean": float(col_series.mean()) if not col_series.empty else 0.0,
            "min": float(col_series.min()) if not col_series.empty else 0.0,
            "max": float(col_series.max()) if not col_series.empty else 0.0,
        }

    # Global averages for single-row filter results (not count queries)
    if row_count == 1 and not is_count_query and sql_query and len(numeric_cols) > 0:
        try:
            match = re.search(r"FROM\s+([\[\]`\w\.]+)", sql_query, re.IGNORECASE)
            if match:
                table_name = match.group(1).strip()
                from database.sql_server import get_engine
                from sqlalchemy import text
                engine = get_engine()
                with engine.connect() as conn:
                    for col in numeric_cols:
                        try:
                            res = conn.execute(
                                text(f"SELECT AVG(CAST([{col}] AS FLOAT)) FROM {table_name}")
                            ).fetchone()
                            if res and res[0] is not None:
                                enrichment["column_stats"][col]["global_average"] = float(res[0])
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Failed to fetch global averages: {e}")

    # Categorical stats
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        col_series = df[col].dropna()
        unique_count = int(col_series.nunique())
        enrichment["column_stats"][col] = {
            "unique_values": unique_count,
            "top_values": col_series.value_counts().head(3).to_dict() if unique_count > 0 else {}
        }

    logger.info(f"Enrichment: {row_count} rows returned, truncated={is_truncated}, "
                f"count_query={is_count_query}, table_total={table_total_rows}")
    return enrichment
