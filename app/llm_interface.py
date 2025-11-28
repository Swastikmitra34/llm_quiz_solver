import os
import json
import re
import asyncio
import time
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")

# Global rate limiter state
_last_request_time = 0
_request_count = 0
_request_window_start = 0
_MAX_REQUESTS_PER_MINUTE = 10  # Adjust based on your API limits


async def rate_limit_wait():
    """Simple rate limiter to prevent 429 errors."""
    global _last_request_time, _request_count, _request_window_start
    
    now = time.time()
    
    # Reset counter every 60 seconds
    if now - _request_window_start > 60:
        _request_count = 0
        _request_window_start = now
    
    # If we've hit the limit, wait
    if _request_count >= _MAX_REQUESTS_PER_MINUTE:
        wait_time = 60 - (now - _request_window_start) + 1
        if wait_time > 0:
            print(f"[RATE LIMIT] Waiting {wait_time:.1f}s to respect rate limit")
            await asyncio.sleep(wait_time)
            _request_count = 0
            _request_window_start = time.time()
    
    # Minimum 2 seconds between requests
    time_since_last = now - _last_request_time
    if time_since_last < 2:
        wait = 2 - time_since_last
        print(f"[RATE LIMIT] Waiting {wait:.1f}s between requests")
        await asyncio.sleep(wait)
    
    _request_count += 1
    _last_request_time = time.time()


async def ask_llm_for_answer(
    question_text: str,
    context_text: str = "",
    data_notes: str = "",
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    LLM Interface with automatic retry on rate limits.
    
    Args:
        question_text: The quiz question to answer
        context_text: HTML or additional context
        data_notes: Data from files or scraped content
        max_retries: Maximum number of retry attempts
    
    Returns:
        {"answer": value} or {"answer": None, "error": "..."}
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
        "1. Find the actual QUESTION in the page (e.g., 'What is 2+2?', 'Get the secret code', etc.)\n"
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
        '- Question: "Get the secret code" (and supporting data shows "Code: ABC123") → {"answer": "ABC123"}\n\n'
        
        "WRONG OUTPUT (DO NOT DO THIS):\n"
        '- {"email": "...", "secret": "...", "url": "...", "answer": 123}\n'
        '- {"answer": "anything you want"}\n'
        '- {"answer": "placeholder"}\n'
        '- {"answer": "your secret"}\n'
        '- {"answer": "the secret code you scraped"}\n'
        '- {"answer": "the correct answer"}\n'
    )
    
    user_prompt = f"""QUESTION/TASK:
{question_text}

SUPPORTING DATA (data from scraped pages, files, etc.):
{data_notes if data_notes else "(none)"}

PAGE CONTEXT (may contain example formats - ignore them):
{context_text[:2000]}

Analyze the QUESTION/TASK and find the answer. Output ONLY: {{"answer": <value>}}"""
    
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
    
    base_delay = 5  # Start with 5 seconds
    
    for attempt in range(max_retries):
        try:
            # Apply rate limiting before each request
            await rate_limit_wait()
            
            print(f"[LLM] Making request (attempt {attempt + 1}/{max_retries})...")
            
            response = requests.post(
                f"{AIPIPE_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Handle 429 specifically
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # Check Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        delay = int(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                    
                    print(f"[LLM] Rate limit (429). Waiting {delay}s before retry {attempt + 2}/{max_retries}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return {
                        "answer": None,
                        "error": f"Rate limit exceeded after {max_retries} attempts"
                    }
            
            # Handle other HTTP errors
            if response.status_code != 200:
                error_text = response.text[:300]
                print(f"[LLM] HTTP {response.status_code}: {error_text}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[LLM] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                
                return {
                    "answer": None,
                    "error": f"HTTP {response.status_code}: {error_text}"
                }
            
            # Parse successful response
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                return {
                    "answer": None,
                    "error": f"Invalid JSON response: {str(e)}"
                }
            
            # Success - break retry loop
            break
            
        except requests.exceptions.Timeout as e:
            print(f"[LLM] Timeout error")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] Retrying after {delay}s...")
                await asyncio.sleep(delay)
                continue
            return {"answer": None, "error": f"Request timeout: {str(e)}"}
        
        except requests.exceptions.ConnectionError as e:
            print(f"[LLM] Connection error")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] Retrying after {delay}s...")
                await asyncio.sleep(delay)
                continue
            return {"answer": None, "error": f"Connection failed: {str(e)}"}
        
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(x in error_str for x in ["429", "rate limit", "too many"])
            
            print(f"[LLM] Exception: {str(e)}")
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[LLM] Rate limit detected. Retrying after {delay}s...")
                await asyncio.sleep(delay)
                continue
            
            return {"answer": None, "error": f"LLM request failed: {str(e)}"}
    
    # Parse the response content
    try:
        raw_content = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        return {
            "answer": None,
            "error": f"Malformed LLM response: {str(e)}",
            "raw": str(result)[:200]
        }
    
    # Clean markdown code fences
    raw_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content, flags=re.MULTILINE).strip()
    
    print(f"[LLM] Raw response: {raw_content[:300]}")
    
    # Parse JSON
    try:
        parsed = json.loads(raw_content)
        
        if isinstance(parsed, dict) and "answer" in parsed:
            answer_value = parsed["answer"]
            
            # Validate not a placeholder
            if isinstance(answer_value, str):
                lower_answer = answer_value.lower()
                placeholder_terms = [
                    "placeholder", "anything you want", "your", "example",
                    "student", "correct answer", "the answer", "secret code you scraped",
                    "the secret code", "your answer here"
                ]
                
                # Check if it's ONLY a placeholder (not containing actual data)
                if any(lower_answer == term or lower_answer.startswith(term) for term in placeholder_terms):
                    return {
                        "answer": None,
                        "error": f"LLM returned placeholder answer: {answer_value}"
                    }
            
            return {"answer": answer_value}
        
        # If entire response is a value
        return {"answer": parsed}
    
    except json.JSONDecodeError:
        pass
    
    # Fallback: Pure number
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_content):
        num_val = float(raw_content) if '.' in raw_content else int(raw_content)
        return {"answer": num_val}
    
    # Fallback: Boolean
    if raw_content.lower() in ["true", "false"]:
        return {"answer": raw_content.lower() == "true"}
    
    # Fallback: Clean string
    if len(raw_content) < 1000:
        lower_content = raw_content.lower()
        placeholder_terms = [
            "placeholder", "anything you want", "your answer",
            "example", "student", "the answer"
        ]
        
        # Only reject if it's CLEARLY a placeholder
        if any(lower_content == term for term in placeholder_terms):
            return {
                "answer": None,
                "error": f"LLM returned placeholder text: {raw_content}"
            }
        
        return {"answer": raw_content}
    
    return {
        "answer": None,
        "error": "Could not parse LLM output",
        "raw": raw_content[:300]
    }

