"""
database/sql_server.py
──────────────────────
SQL Server connection module using SQLAlchemy + pyodbc.

Reads all connection parameters from .env and exposes:
  • get_engine()       → returns a singleton SQLAlchemy Engine
  • test_connection()  → runs SELECT 1 and prints success / failure

Usage:
    python -m database.sql_server
"""

import os
import sys
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── Load .env from project root ──────────────────────────────────────
load_dotenv()

# ── Module-level singleton ───────────────────────────────────────────
_engine: Engine | None = None


def _build_connection_string() -> str:
    """
    Build the ODBC connection string from .env variables.
    Supports both Windows Authentication (Trusted_Connection)
    and SQL Server Authentication (UID/PWD).
    """
    server   = os.getenv("SQL_SERVER", "localhost")
    database = os.getenv("SQL_DATABASE", "master")
    driver   = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    trusted  = os.getenv("SQL_TRUSTED_CONNECTION", "no").lower()
    encrypt  = os.getenv("SQL_ENCRYPT", "yes")
    trust_cert = os.getenv("SQL_TRUST_CERTIFICATE", "yes")

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Encrypt={encrypt}",
        f"TrustServerCertificate={trust_cert}",
    ]

    if trusted in ("yes", "true", "1"):
        parts.append("Trusted_Connection=yes")
    else:
        uid = os.getenv("SQL_USERNAME", "")
        pwd = os.getenv("SQL_PASSWORD", "")
        parts.append(f"UID={uid}")
        parts.append(f"PWD={pwd}")

    return ";".join(parts)


def get_engine() -> Engine:
    """
    Return a singleton SQLAlchemy Engine with connection pooling.
    The engine is created once and reused for the lifetime of the process.
    """
    global _engine

    if _engine is not None:
        return _engine

    conn_str = _build_connection_string()
    connection_url = f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}"

    # Pool settings from .env (with safe defaults)
    pool_size    = int(os.getenv("SQL_POOL_SIZE", "10"))
    max_overflow = int(os.getenv("SQL_MAX_OVERFLOW", "20"))
    echo         = os.getenv("SQL_ECHO", "False").lower() in ("true", "1", "yes")

    _engine = create_engine(
        connection_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,       # verify connections before checkout
        pool_recycle=3600,         # recycle connections every hour
        echo=echo,
    )

    return _engine


def test_connection() -> bool:
    """
    Run a lightweight SELECT 1 against SQL Server.
    Prints a clear success or failure message.
    Returns True on success, False on failure.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()

            if row and row[0] == 1:
                print("=" * 55)
                print("  ✅  SQL Server connection successful!")
                print(f"  Server   : {os.getenv('SQL_SERVER')}")
                print(f"  Database : {os.getenv('SQL_DATABASE')}")
                print(f"  Driver   : {os.getenv('SQL_DRIVER')}")
                print(f"  Auth     : {'Windows (Trusted)' if os.getenv('SQL_TRUSTED_CONNECTION', 'no').lower() in ('yes', 'true', '1') else 'SQL Server'}")
                print("=" * 55)
                return True

        # Should not reach here, but just in case
        print("❌  SELECT 1 returned unexpected result.")
        return False

    except Exception as e:
        print("=" * 55)
        print("  ❌  SQL Server connection FAILED!")
        print(f"  Error: {e}")
        print("=" * 55)
        print("\nTroubleshooting tips:")
        print("  1. Is SQL Server running? Check Windows Services → 'SQL Server (SQLEXPRESS)'")
        print("  2. Is the database name correct? Currently set to:", os.getenv("SQL_DATABASE"))
        print("  3. Is the ODBC driver installed? Currently set to:", os.getenv("SQL_DRIVER"))
        print("  4. Check your .env file for typos in SQL_SERVER, SQL_DATABASE, SQL_DRIVER")
        return False


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔌 Testing SQL Server connection...\n")
    success = test_connection()
    sys.exit(0 if success else 1)
