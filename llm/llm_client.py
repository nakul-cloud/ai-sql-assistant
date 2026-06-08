"""
llm/llm_client.py
─────────────────
Groq LLM client for the Enterprise AI SQL Analytics Assistant.
Uses llama-3.1-8b-instant via the Groq API.

Config (from .env):
  GROQ_API_KEY  — your Groq API key (get one free at console.groq.com)
  GROQ_MODEL    — model to use (default: llama-3.1-8b-instant)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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


def generate_text(prompt: str) -> str:
    """
    Generate text using Groq (llama-3.1-8b-instant).

    Args:
        prompt: The full prompt string.

    Returns:
        The generated text as a plain string.

    Raises:
        groq.AuthenticationError (401) — bad API key
        groq.RateLimitError (429)      — quota exhausted
        Any other Groq API error       — connection/server issues
    """
    client = _get_groq_client()
    model = os.getenv("GROQ_MODEL", GROQ_MODEL)
    logger.debug(f"Groq request: model={model}, prompt_len={len(prompt)}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()
