
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
    """
    Sends a quiz-solving request to an LLM and enforces strict JSON output.
    """

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing OPENAI_API_KEY / AI Pipe token"}

    system_msg = (
        "You are a high-precision problem-solving AI.\n\n"
        "You MUST return the final answer in strict JSON format:\n"
        "{\"answer\": \"<final answer>\"}\n\n"
        "Rules:\n"
        "- Only JSON. No commentary.\n"
        "- No markdown.\n"
        "- No reasoning text.\n"
        "- If numeric, output only the number.\n"
        "- If textual, output only the final word/phrase."
    )

    user_msg = (
        f"Solve the following quiz question.\n\n"
        f"QUESTION:\n{question_text}\n\n"
        f"PAGE CONTEXT:\n{context_text}\n\n"
        f"DATA NOTES:\n{data_notes}\n\n"
        "Return ONLY JSON with the key \"answer\"."
    )

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

    # --- API CALL ---
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return {"answer": None, "error": f"LLM request failed: {e}"}

    # --- RAW RESPONSE EXTRACTION ---
    try:
        raw_text = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"answer": None, "error": "Malformed LLM response", "raw": result}

    # --- 1. Try strict JSON ---
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return {"answer": parsed["answer"]}
    except Exception:
        pass

    # --- 2. Try numeric fallback ---
    number_match = re.search(r"-?\d+(\.\d+)?", raw_text)
    if number_match:
        try:
            return {"answer": float(number_match.group())}
        except:
            pass

    # --- 3. Clean fallback text ---
    cleaned = raw_text.replace("\n", "").strip()
    if cleaned:
        return {"answer": cleaned}

    # --- 4. Hard fallback ---
    return {"answer": "0"}

