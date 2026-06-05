import os
import streamlit as st
import pandas as pd
from sqlalchemy import text
from database.sql_server import get_engine
from database.csv_uploader import process_csv, upload_csv_and_index
from database.schema_manager import fetch_database_metadata

# Custom Premium Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
        color: #ffffff;
    }
    .card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    h1, h2, h3 {
        color: #3b82f6 !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📥 Upload CSV & Index Schema")
st.markdown("Easily upload CSV files to load them into your SQL Server database and automatically index their schema for natural language queries.")

# Connection Check
def test_db_connection() -> tuple[bool, str]:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True, "Connected to SQL Server successfully."
    except Exception as e:
        return False, str(e)

db_connected, db_msg = test_db_connection()
if not db_connected:
    st.error(f"❌ Cannot connect to SQL Server database. Please check your `.env` settings.\n\nError: {db_msg}")
    st.stop()

# File Upload Container
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Ensure uploads directory exists
    os.makedirs("data/uploads", exist_ok=True)
    
    # Save the file temporarily
    temp_path = os.path.join("data/uploads", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    df, err = process_csv(temp_path)
    
    if err or df is None:
        st.error(f"❌ {err}")
    else:
        st.success(f"✅ Loaded '{uploaded_file.name}' successfully!")
        
        # Display Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            file_size_kb = os.path.getsize(temp_path) / 1024
            st.metric("File Size", f"{file_size_kb:.2f} KB")
            
        # Preview Data
        st.subheader("🔍 Data Preview (First 5 rows)")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Upload Options
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Configure Target Table")
        
        # Get existing tables
        metadata = fetch_database_metadata()
        existing_tables = [meta["table_name"] for meta in metadata]
        
        dest_type = st.radio(
            "Destination Table Type",
            ["Create New Table", "Append to Existing Table"]
        )
        
        if dest_type == "Create New Table":
            default_table_name = os.path.splitext(uploaded_file.name)[0].lower()
            default_table_name = "".join(c if c.isalnum() else "_" for c in default_table_name)
            
            table_name = st.text_input(
                "Table Name",
                value=default_table_name,
                help="Specify the name of the new database table. Only alphanumeric characters and underscores."
            )
            if_exists = 'replace'
        else:
            if not existing_tables:
                st.warning("⚠️ No existing tables found in the database. Please create a new table first.")
                table_name = None
            else:
                table_name = st.selectbox("Select Existing Table", existing_tables)
            if_exists = 'append'
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Upload trigger button
        if table_name:
            if st.button("🚀 Upload & Index Table"):
                with st.spinner("Processing CSV, saving to SQL Server, and building semantic chunks..."):
                    success, msg = upload_csv_and_index(temp_path, table_name, if_exists)
                    if success:
                        st.balloons()
                        st.success(f"🎉 {msg}")
                    else:
                        st.error(f"❌ {msg}")
                        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
