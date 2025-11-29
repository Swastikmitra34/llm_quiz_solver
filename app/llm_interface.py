"""
llm_interface.py
Simple version without retry logic
"""

import os
import json
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")


async def ask_llm_for_answer(full_context: str) -> Dict[str, Any]:
    """
    LLM Interface – Context Driven
    Input: Structured typed context from orchestrator
    Output: Strict JSON {"answer": value}
    """

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI token"}

    system_prompt = (
    "You are a strict data-analysis engine. "
    "You receive mixed content (web text, CSV, JSON, PDFs, OCR, images, API responses). "
    "From this, compute the EXACT final answer. "
    "Rules: no explanation, no markdown, output only valid JSON {\"answer\": value}. "
    "Use all provided data. If missing data, return {\"answer\": null}. "
)

    payload = {
        "model": AIPIPE_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context},
        ],
    }

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{AIPIPE_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"answer": None, "error": str(e)}

    try:
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"answer": None, "error": "Malformed response"}

    # Remove markdown if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            val = parsed["answer"]

            if isinstance(val, str) and any(x in val.lower() for x in ["secret", "placeholder", "example"]):
                return {"answer": None, "error": "Blocked placeholder output"}

            return {"answer": val}
    except Exception as e:
        return {"answer": None, "error": f"JSON parse error: {str(e)}", "raw": raw}

    return {"answer": None, "error": "Non-compliant output"}
