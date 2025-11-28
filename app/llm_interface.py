import os
import json
import re
from typing import Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")
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
You are a high-precision problem-solving AI.

You MUST return the final answer in strict JSON.
Format:
{"answer": "<final answer>"}

Rules:
- Only JSON. No commentary.
- No markdown.
- No reasoning text.
- If numeric, output only the number.
- If textual, output only the final word/phrase.
""".strip()

    user_msg = f"""
Solve the following quiz question.

QUESTION:
{question_text}

PAGE CONTEXT:
{context_text}

DATA NOTES:
{data_notes}

Return ONLY JSON with the key "answer".
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
        raw = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return {"answer": None, "error": f"Malformed LLM response", "raw": data}

    # 1. Proper JSON parsing
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return {"answer": parsed["answer"]}
    except:
        pass

    # 2. Fallback numeric extraction
    numeric = re.search(r"-?\d+(\.\d+)?", raw)
    if numeric:
        return {"answer": float(numeric.group())}

    # 3. Final fallback as cleaned text
    cleaned = raw.replace("\n", "").strip()
    if cleaned:
        return {"answer": cleaned}

    # 4. Absolute failsafe
    return {"answer": "0"}

