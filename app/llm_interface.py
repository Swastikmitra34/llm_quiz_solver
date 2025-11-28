import os
import json
import re
import asyncio
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
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    LLM Interface (Computation Engine) with automatic retry on rate limits.
    Role: Analyze the question, compute the answer.
    Returns ONLY: {"answer": value}
    
    Args:
        question_text: The quiz question to answer
        context_text: HTML or additional context
        data_notes: Data from files or scraped content
        max_retries: Maximum number of retry attempts for rate limits
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
        "2. If the question asks you to scrape a URL or get a 'secret code', look for that data in the SUPPORTING DATA section\n"
        "3. Compute the REAL answer to that question\n"
        "4. Return ONLY: {\"answer\": <your_computed_value>}\n\n"
        
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
        '- Question: "Is 5 > 3?" → {"answer": true}\n'
        '- Question: "Get the secret code from URL X" → {"answer": "ABC123"} (if ABC123 was found in supporting data)\n\n'
        
        "WRONG OUTPUT (DO NOT DO THIS):\n"
        '- {"email": "...", "secret": "...", "url": "...", "answer": 123}\n'
        '- {"answer": "anything you want"}\n'
        '- {"answer": "placeholder"}\n'
        '- {"answer": "your secret"}\n'
        '- {"answer": "the correct answer"}\n'
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
    
    # Retry loop with exponential backoff
    base_delay = 3  # Start with 3 seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{AIPIPE_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Check for rate limit before raising for status
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # Extract retry-after header if available
                    retry_after = response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        delay = int(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt)  # 3s, 6s, 12s, 24s
                    
                    print(f"[LLM] Rate limit (429). Waiting {delay}s before retry {attempt + 2}/{max_retries}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return {
                        "answer": None, 
                        "error": f"Rate limit exceeded after {max_retries} attempts"
                    }
            
            # For other HTTP errors, raise immediately
            response.raise_for_status()
            result = response.json()
            
            # Success - break out of retry loop
            break
            
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1 and "429" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] HTTP 429 error. Waiting {delay}s before retry {attempt + 2}/{max_retries}")
                await asyncio.sleep(delay)
                continue
            return {"answer": None, "error": f"LLM request failed: {str(e)}"}
        
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] Timeout. Waiting {delay}s before retry {attempt + 2}/{max_retries}")
                await asyncio.sleep(delay)
                continue
            return {"answer": None, "error": f"LLM request timeout: {str(e)}"}
        
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] Connection error. Waiting {delay}s before retry {attempt + 2}/{max_retries}")
                await asyncio.sleep(delay)
                continue
            return {"answer": None, "error": f"LLM connection failed: {str(e)}"}
        
        except Exception as e:
            # For unknown errors, check if it's rate-limit related
            error_str = str(e).lower()
            if ("429" in error_str or "rate limit" in error_str or "too many" in error_str) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] Rate limit detected in error. Waiting {delay}s before retry {attempt + 2}/{max_retries}")
                await asyncio.sleep(delay)
                continue
            return {"answer": None, "error": f"LLM request failed: {str(e)}"}
    
    # Parse the response
    try:
        raw_content = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        return {"answer": None, "error": f"Malformed LLM response: {str(e)}", "raw": result}
    
    # Clean markdown fences
    raw_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content, flags=re.MULTILINE).strip()
    
    print(f"[LLM] Raw response: {raw_content[:200]}")
    
    # --- PRIMARY: Parse JSON ---
    try:
        parsed = json.loads(raw_content)
        
        if isinstance(parsed, dict) and "answer" in parsed:
            answer_value = parsed["answer"]
            
            # Validate that answer is not a placeholder
            if isinstance(answer_value, str):
                lower_answer = answer_value.lower()
                placeholder_terms = [
                    "placeholder", "anything", "your", "example", 
                    "student", "correct answer", "the answer", "secret code you scraped"
                ]
                if any(term in lower_answer for term in placeholder_terms):
                    return {
                        "answer": None, 
                        "error": f"LLM returned placeholder answer: {answer_value}"
                    }
            
            return {"answer": answer_value}
        
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
    
    # --- FALLBACK 3: Clean string (but check for placeholders) ---
    if len(raw_content) < 1000:
        lower_content = raw_content.lower()
        placeholder_terms = [
            "placeholder", "anything", "your", "example", 
            "student", "correct answer", "the answer"
        ]
        if any(term in lower_content for term in placeholder_terms):
            return {
                "answer": None, 
                "error": f"LLM returned placeholder text: {raw_content[:100]}"
            }
        return {"answer": raw_content}
    
    return {"answer": None, "error": "Could not parse LLM output", "raw": raw_content[:200]}
