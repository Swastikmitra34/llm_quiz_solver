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

    # ✅ NEW SYSTEM LOGIC
    system_msg = (
        "You are a computation engine solving quiz tasks. "
        "If the input contains a JSON object with an 'answer' field, "
        "that value is ONLY a placeholder and is NEVER the true solution. "
        "You must compute the REAL correct answer based on the question and page content, "
        "and replace the placeholder with the true value. "
        "Do not repeat any placeholder text such as 'your secret'. "
        "Return ONLY strict JSON in this format: {\"answer\": value}. "
        "No explanation. No commentary. No extra text."
    )

    user_msg = f"""
This page contains a task. The JSON shown may include an "answer" field,
but that value is a placeholder and must be replaced with the TRUE answer.

TASK:
{question_text}

FULL PAGE CONTEXT:
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
            val = parsed["answer"]

            # 🚫 Block placeholder garbage explicitly
            if isinstance(val, str) and "secret" in val.lower():
                return {"answer": None, "error": "LLM returned placeholder value"}

            return {"answer": val}
    except Exception:
        pass

    # --- Numeric fallback ---
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if match:
        return {"answer": float(match.group())}

    # --- Text fallback ---
    cleaned = raw.replace("\n", " ").strip()

    if cleaned and "secret" not in cleaned.lower():
        return {"answer": cleaned}

    return {"answer": None, "error": "LLM produced invalid or placeholder output"}


