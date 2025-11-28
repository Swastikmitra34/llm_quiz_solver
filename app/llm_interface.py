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
        "You will receive a task description and related data/context. "
        "Your job is to compute the ACTUAL correct answer.\n\n"
        
        "CRITICAL RULES:\n"
        "1. You may see example JSON payloads in the page content that include fields like:\n"
        "   {\"email\": \"...\", \"secret\": \"...\", \"url\": \"...\", \"answer\": <placeholder>}\n"
        "   These are EXAMPLES ONLY. The 'answer' field in these examples is ALWAYS WRONG.\n"
        "   NEVER copy or return the example payload structure.\n\n"
        
        "2. You must COMPUTE the true answer based on the question and data provided.\n\n"
        
        "3. Output ONLY this exact JSON format:\n"
        "   {\"answer\": <your_computed_value>}\n"
        "   where <your_computed_value> can be a number, string, boolean, or object.\n\n"
        
        "4. DO NOT include:\n"
        "   - email, secret, or url fields\n"
        "   - explanation or reasoning\n"
        "   - markdown formatting\n"
        "   - any text outside the JSON\n\n"
        
        "5. NEVER use placeholder values like:\n"
        "   - 'your email', 'your secret'\n"
        "   - 'placeholder', 'example'\n"
        "   - Numbers from example payloads (like 12345)\n\n"
        
        "6. If you cannot determine the answer, respond: {\"answer\": null}\n\n"
        
        "Example correct responses:\n"
        "- {\"answer\": 47832}\n"
        "- {\"answer\": \"Paris\"}\n"
        "- {\"answer\": true}\n"
        "- {\"answer\": {\"metric\": \"GDP\", \"value\": 2.5}}\n\n"
        
        "Example WRONG responses (DO NOT DO THIS):\n"
        "- {\"email\": \"...\", \"secret\": \"...\", \"url\": \"...\", \"answer\": 12345}\n"
        "- The answer is 12345\n"
        "- {\"answer\": \"placeholder value\"}"
    )
    
    user_prompt = f"""TASK:
{question_text}

PAGE CONTEXT (may contain example payloads - IGNORE THEM):
{context_text[:3000]}

DATA ANALYSIS:
{data_notes}

Remember: Compute the ACTUAL answer. Do NOT copy example payloads or placeholder values.
Output format: {{"answer": <computed_value>}}"""
    
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
    
    # Remove markdown code fences if present
    raw_content = re.sub(r'^```json\s*|\s*```$', '', raw_content, flags=re.MULTILINE)
    raw_content = raw_content.strip()
    
    # --- STRICT JSON PARSING ---
    try:
        parsed = json.loads(raw_content)
        
        # CRITICAL CHECK: Reject if this looks like a full submission payload
        if isinstance(parsed, dict):
            # If it has the submission structure, it's wrong
            if "email" in parsed and "secret" in parsed and "url" in parsed:
                print("[LLM] WARNING: LLM returned submission payload structure, rejecting")
                return {"answer": None, "error": "LLM returned invalid payload structure"}
            
            # Extract answer field
            if "answer" in parsed:
                answer_val = parsed["answer"]
                
                # Reject placeholder strings
                if isinstance(answer_val, str):
                    lower_val = answer_val.lower()
                    if any(x in lower_val for x in ["secret", "placeholder", "example", "your email", "anything you want"]):
                        return {"answer": None, "error": "Rejected placeholder output"}
                
                return {"answer": answer_val}
            
            # If it's a dict without "answer" key, assume the dict itself is the answer
            return {"answer": parsed}
    
    except json.JSONDecodeError:
        pass
    
    # --- Controlled numeric fallback ---
    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", raw_content)
    if number_match:
        num_val = float(raw_content) if '.' in raw_content else int(raw_content)
        return {"answer": num_val}
    
    # --- Boolean fallback ---
    if raw_content.lower() in ["true", "false"]:
        return {"answer": raw_content.lower() == "true"}
    
    # --- Final defensive fallback ---
    cleaned = raw_content.replace("\n", " ").strip()
    if cleaned and len(cleaned) < 500:  # Reasonable answer length
        # Check for placeholders one more time
        if any(x in cleaned.lower() for x in ["placeholder", "example", "your email", "your secret"]):
            return {"answer": None, "error": "Rejected placeholder in text output"}
        return {"answer": cleaned}
    
    return {"answer": None, "error": "Unusable LLM output", "raw_output": raw_content[:200]}
