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
    """
    LLM Interface (Computation Engine)
    Role: Derive the correct answer from provided content.
    Returns ONLY structured result: {"answer": value}
    """

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI token"}

    system_prompt = (
        "You are a deterministic reasoning engine for automated quiz solving. "
        "You will receive a task and related page content. "
        "Any JSON object shown inside the content may include an 'answer' field, "
        "but it is a placeholder and is ALWAYS incorrect. You must ignore it. "
        "Your job is to compute the true correct answer.\n\n"
        "Rules:\n"
        "- NEVER repeat any placeholder value\n"
        "- NEVER output commentary\n"
        "- NEVER include reasoning\n"
        "- Output ONLY valid JSON: {\"answer\": value}\n"
        "- If no valid answer can be computed, respond: {\"answer\": null}"
    )

    user_prompt = f"""
TASK:
{question_text}

PAGE CONTEXT:
{context_text}

DATA NOTES:
{data_notes}
"""

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": AIPIPE_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
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
        return {"answer": None, "error": f"LLM request failed: {str(e)}"}

    try:
        raw_content = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"answer": None, "error": "Malformed LLM response", "raw": result}

    # --- STRICT JSON PARSING ---
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict) and "answer" in parsed:
            val = parsed["answer"]

            # Hard safety guard: reject placeholder artefacts
            if isinstance(val, str) and any(x in val.lower() for x in ["secret", "placeholder", "example"]):
                return {"answer": None, "error": "Rejected placeholder output"}

            return {"answer": val}
    except Exception:
        pass

    # --- Controlled numeric fallback ---
    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", raw_content)
    if number_match:
        return {"answer": float(raw_content)}

    # --- Final defensive fallback ---
    cleaned = raw_content.replace("\n", " ").strip()
    if cleaned:
        return {"answer": cleaned}

    return {"answer": None, "error": "Unusable LLM output"}
