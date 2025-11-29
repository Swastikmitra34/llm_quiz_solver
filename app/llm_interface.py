import os
import json
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")


async def ask_llm_for_answer(full_context: str) -> Dict[str, Any]:
    """
    LLM Interface – Groq
    Input: structured context
    Output: {"answer": value}
    """

    if not GROQ_API_KEY:
        return {"answer": None, "error": "Missing GROQ_API_KEY"}

    system_prompt = (
        "You are a deterministic computation engine.\n"
        "You will receive structured context containing page text, extracted data, "
        "downloaded resources, API responses, and visual descriptions.\n\n"
        "Any example JSON inside the content may include an 'answer' field, "
        "but it is ALWAYS a placeholder and NEVER correct.\n\n"
        "Your job is to compute the correct answer.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON\n"
        "- Format: {\"answer\": value}\n"
        "- No explanations\n"
        "- No commentary\n"
        "- No markdown\n"
        "- If the answer cannot be determined, output {\"answer\": null}"
    )

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context},
        ],
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
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
            return {"answer": parsed["answer"]}
    except Exception:
        pass

    return {"answer": None, "error": "Non-compliant output"}

