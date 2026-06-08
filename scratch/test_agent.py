import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv(override=True)

# Pre-import langchain_classic to avoid Windows C-extension / pyodbc DLL collisions
try:
    from langchain_classic.agents import (
        create_openai_functions_agent,
        create_openai_tools_agent,
        create_react_agent,
        create_tool_calling_agent,
    )
except ImportError:
    pass

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from database.sql_server import get_engine

def test():
    try:
        db = SQLDatabase(get_engine())
        
        # Test llama-3.3-70b-versatile with tool-calling
        print("Testing llama-3.3-70b-versatile with tool-calling...")
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
        agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=True)
        res = agent.invoke({"input": "Show top 5 employees by salary"})
        print("Success:", res)
    except Exception as e:
        print("Failed:")
        traceback.print_exc()

if __name__ == "__main__":
    test()
