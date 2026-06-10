import streamlit as st
import pandas as pd
from workflow.process_query import process_user_query
from database.schema_manager import fetch_database_metadata

# Page config
st.set_page_config(
    page_title="AI SQL Analytics Assistant",
    page_icon="💬",
    layout="wide"
)

# Dark theme styling
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
    }
    .card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #3b82f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 AI SQL Analytics Assistant")
st.markdown("Ask analytical questions about your database tables in natural language.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dev_mode = st.toggle("Developer Mode", value=False, help="Show SQL and execution details")
    
    # Selected focus tables option
    st.subheader("🎯 Focus Tables")
    try:
        metadata = fetch_database_metadata()
        available_tables = [meta["table_name"] for meta in metadata]
    except Exception:
        available_tables = []
        
    if available_tables:
        selected_tables = st.multiselect(
            "Limit queries to:",
            options=available_tables,
            default=available_tables,
            help="Select which tables are included in schema context"
        )
    else:
        st.warning("⚠️ No active tables. Upload data in the Upload Page.")
        selected_tables = []

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display rephrased query if Developer Mode is on
        if dev_mode and "rephrased" in msg and msg["rephrased"]:
            st.info(f"🔄 **Contextualized Query:** *{msg['rephrased']}*")

        # Display SQL if Developer Mode is on
        if dev_mode and "sql" in msg and msg["sql"]:
            st.code(msg["sql"], language="sql")
            
        # Display data results table if present
        if "rows" in msg and msg["rows"]:
            df = pd.DataFrame(msg["rows"])
            st.dataframe(df, use_container_width=True)

# User Chat Input
user_input = st.chat_input("Ask a question about the database...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing tables and generating insights..."):
            focus_list = selected_tables if selected_tables else None
            res = process_user_query(user_input, focus_tables=focus_list, chat_history=st.session_state.messages[:-1])
            
            if not res["success"]:
                raw_error = res.get("error", "Query processing failed.")
                
                # Friendly error messages based on error type
                if "429" in raw_error or "rate_limit_exceeded" in raw_error.lower():
                    err_msg = (
                        "⚠️ **API Quota Exhausted / Rate Limit Exceeded**\n\n"
                        "Your Groq API rate limits have been hit.\n\n"
                        "**Quick fix options:**\n"
                        "1. Wait a minute and try again (llama-3.1-8b-instant has generous free-tier limits, but you can hit requests-per-minute limits).\n"
                        "2. Get a new API key from [console.groq.com](https://console.groq.com) and update `GROQ_API_KEY` in your `.env` file.\n"
                    )
                elif "401" in raw_error or "invalid_api_key" in raw_error.lower() or "authentication" in raw_error.lower():
                    err_msg = (
                        "🔑 **Invalid API Key**\n\n"
                        "Your `GROQ_API_KEY` is invalid or expired.\n"
                        "Please go to [console.groq.com](https://console.groq.com), "
                        "generate a new API key, and update it in your `.env` file."
                    )
                else:
                    err_msg = f"❌ **Error:** {raw_error}"
                    
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

            else:
                intent = res.get("intent", "SQL_QUERY")
                nl_response = res.get("nl_response", "")
                rephrased = res.get("rephrased_query", "")
                
                # Check for SQL result details
                sql_query = res.get("generated_sql", None)
                query_res = res.get("query_result") or {}
                rows = query_res.get("rows", [])
                
                # Render results in UI
                st.success(nl_response)
                
                if dev_mode and rephrased and rephrased != user_input:
                    st.info(f"🔄 **Contextualized Query:** *{rephrased}*")
                    
                if dev_mode and sql_query:
                    st.code(sql_query, language="sql")
                    
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    
                # Append assistant response to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": nl_response,
                    "sql": sql_query,
                    "rows": rows,
                    "rephrased": rephrased if rephrased != user_input else None
                })
