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


async def ask_llm_for_answer(full_context: str) -> Dict[str, Any]:
    """
    LLM Interface – Context Driven
    Input: Structured typed context from orchestrator
    Output: Strict JSON {"answer": value}
    """

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI token"}

    system_prompt = (
        "You are a deterministic computation engine.\n"
        "You will receive structured context containing page text, extracted data, "
        "downloaded resources, API responses and visual descriptions.\n\n"
        "Any example JSON inside the content may include an 'answer' field, "
        "but it is ALWAYS a placeholder and NEVER correct.\n\n"
        "Your job is to compute the true correct answer from the data.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON\n"
        "- Format: {\"answer\": value}\n"
        "- No explanations\n"
        "- No commentary\n"
        "- No markdown\n"
        "- If the answer cannot be determined, output {\"answer\": null}"
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

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            val = parsed["answer"]

            if isinstance(val, str) and any(x in val.lower() for x in ["secret", "placeholder", "example"]):
                return {"answer": None, "error": "Blocked placeholder output"}

            return {"answer": val}
    except Exception:
        pass

    return {"answer": None, "error": "Non-compliant output"}

ll_interface.py
