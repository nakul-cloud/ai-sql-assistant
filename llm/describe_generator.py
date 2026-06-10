"""
llm/describe_generator.py
─────────────────────────
Handles 'explain this dataset' / 'what is this table about' intent.
Uses schema metadata only — no SQL execution needed.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_client import get_llm

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst explaining a specific business dataset to a non-technical user.
Your job is to describe ONLY the dataset provided in the schema below.

STRICT RULES:
- Describe ONLY the table shown in the schema. Do NOT mention or reference any other tables,
  datasets, or data sources — even if you know they exist.
- Do NOT invent columns, metrics, or capabilities not present in the schema.
- Do NOT run or suggest SQL queries. This is a plain English description only.
- Do NOT say "I queried the database" or imply any data was fetched.
- Translate column names into plain business language — no technical names.
  (e.g. "ai_investment_usd" → "AI spending", "ai_maturity_score" → "AI maturity level")
- Keep it to 4–6 sentences.
- End with 2–3 example questions the user could ask about this specific dataset.
- If the user seems confused or new, use a welcoming, simple tone.
"""),
    ("human", """User question: {user_query}

Schema for this dataset:
{schema_summary}

Plain English description:""")
])

def build_describe_chain():
    llm = get_llm(temperature=0.4)
    return DESCRIBE_PROMPT | llm | StrOutputParser()

_describe_chain = build_describe_chain()

def generate_dataset_description(user_query: str, schema_summary: str, stream: bool = False):
    logger.info("Generating dataset description.")
    if stream:
        return _describe_chain.stream({
            "user_query": user_query,
            "schema_summary": schema_summary
        })
    result = _describe_chain.invoke({
        "user_query": user_query,
        "schema_summary": schema_summary
    })
    return result.strip()


OVERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst summarizing all the available tables in a database to a business user.
Your job is to provide a brief high-level overview of the entire database.
For each table in the schema metadata provided below:
- Provide a brief 1-sentence summary of what it tracks.
- Mention its approximate row count.
Keep the entire response under 10 sentences. Do not write any SQL queries.
"""),
    ("human", """User question: {user_query}

Database Schema Summaries:
{schema_summary}

Database Overview:""")
])

def generate_database_overview(user_query: str, schema_summary: str, stream: bool = False):
    logger.info("Generating database overview.")
    llm = get_llm(temperature=0.4)
    chain = OVERVIEW_PROMPT | llm | StrOutputParser()
    if stream:
        return chain.stream({
            "user_query": user_query,
            "schema_summary": schema_summary
        })
    return chain.invoke({
        "user_query": user_query,
        "schema_summary": schema_summary
    }).strip()


if __name__ == "__main__":
    mock_schema = """
    Table: corporate_ai_adoption
    Columns: company_id, industry, ai_adoption_level, ai_investment_usd,
             cost_savings_usd, revenue_impact_usd, deployment_count,
             employee_ai_training_hours, ai_maturity_score, year
    """
    print(generate_dataset_description(
        "can you explain what this dataset is about? i'm new",
        mock_schema
    ))
