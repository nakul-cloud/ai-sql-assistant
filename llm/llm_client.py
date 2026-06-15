"""
llm/llm_client.py
─────────────────
Groq LLM client for the Enterprise AI SQL Analytics Assistant.
Uses llama-3.1-8b-instant via the Groq API and LangChain.

Config (from .env):
  GROQ_API_KEY  — your Groq API key (get one free at console.groq.com)
  GROQ_MODEL    — model to use (default: llama-3.1-8b-instant)
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)
logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_groq_client = None


def _get_groq_client():
    """Lazy singleton Groq client."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env — get a free key at console.groq.com")
        _groq_client = Groq(api_key=api_key)
        logger.info(f"Groq client initialized. Model: {GROQ_MODEL}")
    return _groq_client


@lru_cache(maxsize=2)
def get_llm(temperature: float = 0.0):
    """
    Returns a ChatGroq instance with dynamic fallback chains (Groq -> OpenAI -> Gemini)
    configured based on available environment variables.
    """
    # 1. Primary: Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set in .env")
    groq_model = os.getenv("GROQ_MODEL", GROQ_MODEL)
    
    primary_llm = ChatGroq(
        model_name=groq_model,
        groq_api_key=groq_key,
        temperature=temperature
    )

    fallbacks = []

    # 2. Fallback 1: OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            fallbacks.append(ChatOpenAI(
                model=openai_model,
                openai_api_key=openai_key,
                temperature=temperature
            ))
            logger.info("OpenAI LLM fallback configured.")
        except ImportError:
            logger.warning("langchain-openai is not installed; OpenAI fallback skipped.")

    # 3. Fallback 2: Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            fallbacks.append(ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=gemini_key,
                temperature=temperature
            ))
            logger.info("Gemini LLM fallback configured.")
        except ImportError:
            logger.warning("langchain-google-genai is not installed; Gemini fallback skipped.")

    if fallbacks:
        return primary_llm.with_fallbacks(fallbacks)
    return primary_llm


def generate_text(prompt: str) -> str:
    """
    Generate text using the main LLM chain (with fallbacks enabled).
    """
    llm = get_llm(temperature=0.0)
    logger.debug(f"LLM request: prompt_len={len(prompt)}")
    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":
    # Test both client types
    print("Testing Groq generate_text...")
    try:
        print(generate_text("Say hello in one word."))
    except Exception as e:
        print(f"Groq error: {e}")
        
    print("\nTesting LangChain ChatGroq get_llm...")
    try:
        llm = get_llm()
        response = llm.invoke("Say hello in one word.")
        print(response.content.strip())
    except Exception as e:
        print(f"LLM error: {e}")
