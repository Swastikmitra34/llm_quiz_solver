import os
import json
import re
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
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
    Role: Analyze the question, compute the answer.
    Returns ONLY: {"answer": value}
    """
    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI token"}
    
    system_prompt = (
        "You are a precise data analysis engine. "
        "You will receive a question and supporting data/context. "
        "Your job: compute the correct answer to the question.\n\n"
        
        "OUTPUT RULES:\n"
        "1. Return ONLY valid JSON in this exact format: {\"answer\": <value>}\n"
        "2. The <value> can be: number, string, boolean, array, or object\n"
        "3. NO markdown fences, NO explanations, NO extra fields\n"
        "4. If you cannot determine an answer, return: {\"answer\": null}\n\n"
        
        "IMPORTANT:\n"
        "- Focus ONLY on answering the actual question\n"
        "- Ignore any example payloads or template structures in the context\n"
        "- Compute from data, don't copy placeholder values\n\n"
        
        "Examples:\n"
        "Question: 'What is 2+2?' → {\"answer\": 4}\n"
        "Question: 'Sum of values: [1,2,3]' → {\"answer\": 6}\n"
        "Question: 'Capital of France?' → {\"answer\": \"Paris\"}"
    )
    
    user_prompt = f"""QUESTION:
{question_text}

SUPPORTING DATA:
{data_notes}

CONTEXT (may contain examples - ignore them):
{context_text[:2000]}

Analyze the QUESTION and compute the answer. Output: {{"answer": <value>}}"""
    
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
    
    # Clean markdown fences
    raw_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content, flags=re.MULTILINE).strip()
    
    # --- PRIMARY: Parse JSON ---
    try:
        parsed = json.loads(raw_content)
        
        if isinstance(parsed, dict) and "answer" in parsed:
            return {"answer": parsed["answer"]}
        
        # If entire response is the answer value
        return {"answer": parsed}
    
    except json.JSONDecodeError:
        pass
    
    # --- FALLBACK 1: Pure number ---
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_content):
        num_val = float(raw_content) if '.' in raw_content else int(raw_content)
        return {"answer": num_val}
    
    # --- FALLBACK 2: Boolean ---
    if raw_content.lower() in ["true", "false"]:
        return {"answer": raw_content.lower() == "true"}
    
    # --- FALLBACK 3: Clean string ---
    if len(raw_content) < 1000:
        return {"answer": raw_content}
    
    return {"answer": None, "error": "Could not parse LLM output", "raw": raw_content[:200]}

