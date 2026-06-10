"""
workflow/query_executor.py
──────────────────────────
Production-grade SQL execution engine using SQLAlchemy.
"""

import time
import logging
from typing import Dict, Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.sql_server import get_engine

logger = logging.getLogger(__name__)

MAX_RESULT_ROWS = 10000
QUERY_TIMEOUT_SECONDS = 30


def enforce_row_limit(sql_query: str) -> str:
    """
    Enforces row limiting safely for large result prevention.
    In T-SQL, this means injecting TOP if not already present.
    """
    upper_query = sql_query.upper().strip()
    if "TOP" in upper_query:
        return sql_query

    # Simple regex injection for SELECT queries
    if upper_query.startswith("SELECT"):
        # Replace the first SELECT with SELECT TOP {MAX_RESULT_ROWS}
        return re.sub(
            r"^SELECT",
            f"SELECT TOP {MAX_RESULT_ROWS}",
            sql_query,
            count=1,
            flags=re.IGNORECASE
        )
    return sql_query


import re


def serialize_dataframe(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Converts dataframe into structured response: columns, rows (as dicts), row_count.
    """
    # Convert datetime/Timestamp columns to string representation for easy JSON serialization
    for col in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].astype(str)

    return {
        "columns": dataframe.columns.tolist(),
        "rows": dataframe.to_dict(orient="records"),
        "row_count": len(dataframe)
    }


def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """
    Executes T-SQL query safely.
    """
    logger.info("Executing SQL query.")
    start_time = time.time()

    try:
        # Enforce safe row limit
        safe_query = enforce_row_limit(sql_query)
        logger.info(f"Final SQL Query: {safe_query}")

        engine = get_engine()

        with engine.connect() as connection:
            # Set lock timeout
            connection.execute(text(f"SET LOCK_TIMEOUT {QUERY_TIMEOUT_SECONDS * 1000}"))
            dataframe = pd.read_sql(text(safe_query), connection)

        execution_time = round(time.time() - start_time, 3)
        logger.info(f"Query executed successfully in {execution_time}s")

        serialized_results = serialize_dataframe(dataframe)
        return {
            "success": True,
            "sql_query": safe_query,
            "execution_time_seconds": execution_time,
            "result": serialized_results
        }

    except SQLAlchemyError as error:
        logger.exception("Database execution failed.")
        return {
            "success": False,
            "sql_query": sql_query,
            "error_type": "DATABASE_ERROR",
            "error": str(error)
        }
    except Exception as error:
        logger.exception("Unexpected execution error.")
        return {
            "success": False,
            "sql_query": sql_query,
            "error_type": "SYSTEM_ERROR",
            "error": str(error)
        }


if __name__ == "__main__":
    print("\n--- Testing Query Executor ---")
    res = execute_sql_query("SELECT * FROM dbo.csv_departments")
    if res["success"]:
        print("Success!")
        print(f"Columns: {res['result']['columns']}")
        print(f"Row count: {res['result']['row_count']}")
        print(f"First row: {res['result']['rows'][0] if res['result']['rows'] else 'None'}")
    else:
        print(f"Failed: {res['error']}")
