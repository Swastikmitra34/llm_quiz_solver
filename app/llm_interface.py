import os
import json
import re
import asyncio
import time
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")

# Global throttle: minimum seconds between LLM calls
MIN_CALL_INTERVAL = 5
_last_llm_call_time = 0


def _should_bypass_llm(question_text: str) -> bool:
    """Detect procedural / numeric / data tasks (no LLM needed)."""
    keywords = [
        "sum", "total", "average", "mean", "count",
        "table", "column", "dataset", "file",
        "download", "scrape", "csv", "excel", "page 2"
    ]
    text = question_text.lower()
    return any(k in text for k in keywords)


async def _global_throttle():
    global _last_llm_call_time
    now = time.time()
    delta = now - _last_llm_call_time

    if delta < MIN_CALL_INTERVAL:
        await asyncio.sleep(MIN_CALL_INTERVAL - delta)

    _last_llm_call_time = time.time()


async def ask_llm_for_answer(
    question_text: str,
    context_text: str = "",
    data_notes: str = "",
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    Hardened LLM Interface:
    - global throttle
    - automatic retry on 429
    - bypass LLM for procedural tasks
    - strict JSON output
    """

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI token"}

    # --------------------------
    # 1. OPTIONAL: BYPASS LLM
    # --------------------------
    if _should_bypass_llm(question_text):
        return {
            "answer": None,
            "error": "Procedural task detected. LLM bypassed."
        }

    # --------------------------
    # 2. PROMPTS (minimal + strict)
    # --------------------------
    system_prompt = (
        "You are a strict computation engine.\n"
        "Return ONLY JSON: {\"answer\": <value>}.\n"
        "No explanations, no placeholders, no email/secret/url.\n"
        "If unknown, return {\"answer\": null}."
    )

    user_prompt = f"QUESTION:\n{question_text}\n\nCONTEXT:\n{context_text[:1500]}"

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

    # --------------------------
    # 3. RETRY LOGIC WITH BACKOFF
    # --------------------------
    base_delay = 3

    for attempt in range(max_retries):
        try:
            await _global_throttle()

            response = requests.post(
                f"{AIPIPE_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            # Rate limit
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    return {"answer": None, "error": "Rate limit exceeded after retries"}

                delay = base_delay * (2 ** attempt)  # 3 → 6 → 12 → 24
                await asyncio.sleep(delay)
                continue

            # Other HTTP errors
            response.raise_for_status()
            result = response.json()
            break

        except Exception as e:
            if attempt == max_retries - 1:
                return {"answer": None, "error": f"LLM failure: {str(e)}"}

    # --------------------------
    # 4. PARSE RESPONSE
    # --------------------------
    try:
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"answer": None, "error": "Malformed LLM response"}

    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()

    # Parse JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return {"answer": parsed["answer"]}
        return {"answer": parsed}
    except Exception:
        pass

    # Pure number
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return {"answer": float(raw) if "." in raw else int(raw)}

    # Boolean
    if raw.lower() in ["true", "false"]:
        return {"answer": raw.lower() == "true"}

    # String fallback
    if len(raw) < 800:
        return {"answer": raw}

    return {"answer": None, "error": "Unusable LLM output"}

