"""
llm/langchain_agent.py
──────────────────────
Optional autonomous LangChain SQL Agent fallback wrapper.
Only triggered when traditional execution fails, or if selected,
to dynamically search, self-correct, and execute complex query paths.
"""
import os
from typing import Dict, Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)


def run_autonomous_sql_agent(user_query: str) -> Dict[str, Any]:
    """
    Executes a LangChain SQL agent to autonomously query the SQL Server database.
    It can self-correct queries, discover columns, and execute multi-step database reasoning.
    """
    logger.info(f"Triggering autonomous LangChain SQL Agent for query: '{user_query}'")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "GROQ_API_KEY not found in environment variables."
        }

    try:
        # Force pre-import langchain_classic to avoid Windows C-extension / pyodbc DLL collisions
        from langchain_classic.agents import (
            create_openai_functions_agent,
            create_openai_tools_agent,
            create_react_agent,
            create_tool_calling_agent,
        )
        from langchain_classic.agents.agent import RunnableAgent, RunnableMultiActionAgent, AgentExecutor

        # Import LangChain components locally to prevent slow imports during standard runs
        from langchain_community.utilities import SQLDatabase
        from langchain_community.agent_toolkits import create_sql_agent
        from langchain_groq import ChatGroq
        from database.sql_server import get_engine

        # 1. Connect LangChain SQLDatabase to our active engine
        engine = get_engine()
        db = SQLDatabase(engine)

        # 2. Initialize the LLM (with dynamic fallbacks if keys are available)
        groq_model = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-120b")
        primary_llm = ChatGroq(
            model_name=groq_model,
            groq_api_key=api_key,
            temperature=0.0
        )

        fallbacks = []

        # OpenAI Agent Fallback
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                openai_agent_model = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o")
                fallbacks.append(ChatOpenAI(
                    model=openai_agent_model,
                    openai_api_key=openai_key,
                    temperature=0.0
                ))
                logger.info("OpenAI agent LLM fallback configured.")
            except ImportError:
                logger.warning("langchain-openai is not installed; skipping agent OpenAI fallback.")

        # Gemini Agent Fallback
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_agent_model = os.getenv("GEMINI_AGENT_MODEL", "gemini-1.5-pro")
                fallbacks.append(ChatGoogleGenerativeAI(
                    model=gemini_agent_model,
                    google_api_key=gemini_key,
                    temperature=0.0
                ))
                logger.info("Gemini agent LLM fallback configured.")
            except ImportError:
                logger.warning("langchain-google-genai is not installed; skipping agent Gemini fallback.")

        if fallbacks:
            llm = primary_llm.with_fallbacks(fallbacks)
        else:
            llm = primary_llm

        # 3. Create the SQL Agent
        # We use a tool-calling agent that binds database tools directly to the LLM
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=True,
            handle_parsing_errors=True
        )

        # 4. Execute the query
        logger.info("Executing agent run...")
        response = agent_executor.invoke({"input": user_query})
        output_text = response.get("output", "")

        return {
            "success": True,
            "nl_response": output_text,
            "generated_sql": "Executed autonomously by LangChain Agent"
        }

    except Exception as e:
        logger.exception("LangChain autonomous agent execution failed.")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("\n--- Testing Autonomous LangChain SQL Agent ---")
    res = run_autonomous_sql_agent("Show top 5 employees by salary")
    print("Result:", res)
