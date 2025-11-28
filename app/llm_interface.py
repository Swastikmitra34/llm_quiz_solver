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
        "You are a precise data analysis engine that solves quiz questions.\n\n"
        
        "CRITICAL INSTRUCTION:\n"
        "The page content will show you an EXAMPLE submission format like:\n"
        "{\n"
        '  "email": "your email",\n'
        '  "secret": "your secret",\n'
        '  "url": "...",\n'
        '  "answer": "anything you want"\n'
        "}\n\n"
        
        "This is just showing you the API format. The values are PLACEHOLDERS.\n"
        "DO NOT copy this structure. DO NOT use these placeholder values.\n\n"
        
        "YOUR JOB:\n"
        "1. Find the actual QUESTION in the page (e.g., 'What is 2+2?', 'Sum the values', etc.)\n"
        "2. Compute the REAL answer to that question\n"
        "3. Return ONLY: {\"answer\": <your_computed_value>}\n\n"
        
        "OUTPUT FORMAT:\n"
        "- ONLY return: {\"answer\": <value>}\n"
        "- <value> can be: number, string, boolean, array, or object\n"
        "- NO extra fields like email, secret, url\n"
        "- NO explanations or reasoning\n"
        "- NO markdown formatting\n\n"
        
        "EXAMPLES OF CORRECT OUTPUT:\n"
        '- Question: "What is 2+2?" → {"answer": 4}\n'
        '- Question: "Sum [1,2,3]" → {"answer": 6}\n'
        '- Question: "Capital of France?" → {"answer": "Paris"}\n'
        '- Question: "Is 5 > 3?" → {"answer": true}\n\n'
        
        "WRONG OUTPUT (DO NOT DO THIS):\n"
        '- {"email": "...", "secret": "...", "url": "...", "answer": 123}\n'
        '- {"answer": "anything you want"}\n'
        '- {"answer": "placeholder"}\n'
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
