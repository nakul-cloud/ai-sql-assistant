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
from typing import Dict, Any

logger = logging.getLogger(__name__)

def enrich_sql_result(query_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches the raw SQL query result with semantic insights and statistical summaries.
    Reduces payload size to speed up LLM processing and prevent token overflow.
    """
    logger.info("Applying semantic enrichment to query results.")
    
    rows = query_result.get("rows", [])
    if not rows:
        return {"summary": "The query returned no results.", "enriched_data": None}
        
    df = pd.DataFrame(rows)
    row_count = len(df)
    
    enrichment = {
        "total_rows": row_count,
        "columns": query_result.get("columns", []),
        "column_stats": {},
        "data_sample": df.head(5).to_dict(orient="records") # Only send top 5 rows to LLM
    }
    
    # 1. Generate statistical summaries for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Avoid NaN errors
        col_series = df[col].dropna()
        enrichment["column_stats"][col] = {
            "sum": float(col_series.sum()) if not col_series.empty else 0.0,
            "mean": float(col_series.mean()) if not col_series.empty else 0.0,
            "min": float(col_series.min()) if not col_series.empty else 0.0,
            "max": float(col_series.max()) if not col_series.empty else 0.0,
        }
        
    # 2. Generate basic summaries for categorical/date columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        col_series = df[col].dropna()
        unique_count = int(col_series.nunique())
        enrichment["column_stats"][col] = {
            "unique_values": unique_count,
            "top_values": col_series.value_counts().head(3).to_dict() if unique_count > 0 else {}
        }
        
    logger.info("Semantic enrichment completed successfully.")
    return enrichment
