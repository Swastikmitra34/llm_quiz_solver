import os
import json
from textwrap import dedent
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")  # from aipipe.org
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://api.aipipe.org/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")


async def ask_llm_for_answer(
    question_text: str,
    context_text: str = "",
    data_notes: str = "",
) -> Dict[str, Any]:

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing OPENAI_API_KEY / AI Pipe token"}

    system_msg = """
You are an automated quiz-solving engine.
You MUST return a final answer in strict JSON format ONLY.

Rules:
- Return ONLY JSON.
- The JSON must contain exactly one key: "answer".
- No explanations. No Markdown. No text outside JSON.

Examples:
{"answer": 12345}
{"answer": "True"}
{"answer": "Paris"}
""".strip()


    user_msg = f"""
Question:
{question_text}

Context:
{context_text}

Data notes:
{data_notes}

Respond ONLY with valid JSON containing the key "answer".
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
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"answer": None, "error": f"LLM request failed: {e}"}

    try:
        raw_content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return {"answer": None, "error": f"Unexpected LLM response format: {e}", "raw": data}

    # Robust JSON extraction
    try:
        parsed = json.loads(raw_content)
    except Exception:
        import re
        num_match = re.search(r"-?\d+(\.\d+)?", raw_content)
        if num_match:
            return {"answer": float(num_match.group())}
        return {"answer": raw_content.strip()}

    if isinstance(parsed, dict) and "answer" in parsed:
        return parsed

    return {"answer": parsed}
