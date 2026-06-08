import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
from database.sql_server import get_engine
from database.schema_manager import fetch_database_metadata

# Load .env
load_dotenv(override=True)

# Initialize background indexing scheduler once
from indexing.scheduler import start_scheduler
@st.cache_resource
def init_background_scheduler():
    start_scheduler()

init_background_scheduler()

# Page configuration
st.set_page_config(
    page_title="AI SQL Assistant Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
    }
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 8px;
    }
    .status-connected {
        color: #10b981;
        font-weight: bold;
    }
    .status-disconnected {
        color: #ef4444;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #3b82f6 !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to test DB connection
def test_db_connection() -> tuple[bool, str]:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True, "Connected successfully."
    except Exception as e:
        return False, str(e)

# Header
st.title("✨ AI SQL Assistant Dashboard")
st.markdown("""
Welcome to the **Enterprise AI SQL Assistant**. This system translates natural language into Microsoft SQL Server queries, executes them safely, and generates natural language responses.

Use the sidebar navigation to switch between pages:
* **Dashboard (Home)**: System health and schema structure.
* **Chat Assistant**: Talk to your database.
* **Upload Data**: Ingest new CSV files and index them.
""")

st.divider()

# System Health Dashboard
st.header("🎛️ System Health")

db_connected, db_msg = test_db_connection()
metadata_list = fetch_database_metadata() if db_connected else []
table_count = len(metadata_list)

api_key = os.getenv("GROQ_API_KEY")
groq_ready = bool(api_key and api_key != "your_groq_api_key_here")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if groq_ready:
        st.markdown('<div class="metric-value status-connected">Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-value status-disconnected">Missing Key</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Groq AI Status</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if db_connected:
        st.markdown('<div class="metric-value status-connected">Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-value status-disconnected">Offline</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Database Status</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{table_count}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Tables Indexed</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Schema Explorer
st.header("📊 Database Schema Explorer")

if not db_connected:
    st.error(f"Database connection failed: {db_msg}")
elif not metadata_list:
    st.info("No tables found in the database. Navigate to the **Upload Data** page to ingest data.")
else:
    # Display tables in an expander grid
    cols = st.columns(3)
    col_idx = 0
    
    for meta in metadata_list:
        table_name = meta["table_name"]
        columns = meta["columns"]
        row_count = meta.get("row_count", 0)
        
        with cols[col_idx % 3]:
            with st.expander(f"📋 {table_name} ({row_count} rows)", expanded=False):
                for col in columns:
                    pk_icon = "🔑 " if col.get("primary_key") else "• "
                    col_type = col.get("type", "Unknown")
                    st.markdown(f"`{pk_icon}{col['name']}` *({col_type})*")
        col_idx += 1
