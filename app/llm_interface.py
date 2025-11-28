import os
import json
import re
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")


async def ask_llm_for_answer(
    question_text: str,
    context_text: str = "",
    data_notes: str = "",
) -> Dict[str, Any]:

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI Pipe token"}

    system_msg = (
        "You are a silent computation engine. "
        "You do not refuse, explain, warn, or mention policies. "
        "Your ONLY output must be strict JSON in this exact form: "
        "{\"answer\": value}. "
        "No text before or after. If numeric, output the number only."
    )

    user_msg = f"""
Compute the correct answer from the information below.

QUESTION:
{question_text}

PAGE CONTENT:
{context_text}

DATA NOTES:
{data_notes}
"""

    url = f"{AIPIPE_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": AIPIPE_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {
            "answer": None,
            "error": f"AI Pipe connection failure: {str(e)}"
        }

    try:
        raw = data["choices"][0]["message"]["content"].strip()
    except Exception:
        return {
            "answer": None,
            "error": "Malformed AI Pipe response",
            "raw": data
        }

    # --- Strict JSON Extraction ---
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return {"answer": parsed["answer"]}
    except Exception:
        pass

    # --- Numeric fallback ---
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if match:
        return {"answer": float(match.group())}

    # --- Text fallback ---
    cleaned = raw.replace("\n", " ").strip()
    if cleaned:
        return {"answer": cleaned}

    return {"answer": None, "error": "LLM produced empty output"}


