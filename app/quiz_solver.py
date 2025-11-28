import time
import re
import base64
import json
from typing import Dict, Any, Optional, Union, List
import requests
from bs4 import BeautifulSoup

from .browser import fetch_page_html_and_text
from .llm_interface import ask_llm_for_answer
from .utils import (
    find_submit_url_from_text,
    find_download_links_from_html,
    normalize_url,
    download_and_load_data,
    extract_column_sum_from_question,
    classify_question_type,
)


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    time_left_seconds: float,
) -> Dict[str, Any]:
    """
    Solves a single quiz question by:
    1. Fetching and rendering the page (JavaScript-rendered HTML)
    2. Understanding what the question asks for
    3. Following links, downloading files, calling APIs as needed
    4. Processing data (clean, analyze, visualize)
    5. Submitting the answer
    """

    # Fetch the quiz page (with JS rendering via headless browser)
    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")
    question_text = text.strip()

    answer_value = None
    llm_info = {}

    # ======================================================
    # STEP 1: EXTRACT ALL AVAILABLE RESOURCES
    # ======================================================
    
    # Extract all URLs (links to scrape, APIs, data files)
    all_urls = set()
    
    # From text content
    all_urls.update(re.findall(r"https?://[^\s\"'<>]+", text))
    
    # From HTML links
    for link in soup.find_all('a', href=True):
        href = link['href']
        try:
            # Handle unicode escapes like \u0026
            href = href.encode('utf-8').decode('unicode_escape')
        except:
            pass
        
        if href.startswith('http'):
            all_urls.add(href)
        elif href.startswith('/'):
            base_url = '/'.join(quiz_url.split('/')[:3])
            all_urls.add(base_url + href)
        elif not href.startswith(('#', 'javascript:', 'mailto:')):
            base_path = '/'.join(quiz_url.split('/')[:-1])
            all_urls.add(f"{base_path}/{href}")
    
    # From HTML attributes (src, data-url, etc.)
    for tag in soup.find_all(attrs={'src': True}):
        src = tag.get('src')
        if src and src.startswith('http'):
            all_urls.add(src)
    
    # Extract submit URL (never hardcode!)
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)
    
    # Separate URLs by type
    current_page_url = quiz_url
    other_urls = [url for url in all_urls if url != current_page_url and url != submit_url]
    
    # Extract downloadable file links
    download_links = find_download_links_from_html(html)

    # ======================================================
    # STEP 2: UNDERSTAND THE QUESTION TYPE
    # ======================================================
    
    # Use LLM to understand what the question is asking for
    question_analysis = await ask_llm_for_answer(
        question_text=f"""Analyze this question and respond with a JSON object:
Question: {question_text}

Respond with:
{{
  "task_type": "scraping" | "api" | "data_analysis" | "visualization" | "text_processing" | "combination",
  "needs_external_url": true/false,
  "needs_file_download": true/false,
  "expected_answer_type": "string" | "number" | "boolean" | "json" | "base64_file",
  "what_to_extract": "brief description of what to extract/compute",
  "key_terms": ["list", "of", "keywords", "to", "look", "for"]
}}""",
        context_text=text,
        data_notes="",
    )
    
    try:
        task_info = json.loads(question_analysis.get("answer", "{}"))
    except:
        task_info = {
            "task_type": "unknown",
            "needs_external_url": bool(other_urls),
            "needs_file_download": bool(download_links),
            "expected_answer_type": "string",
        }

    # ======================================================
    # STEP 3: SCRAPE EXTERNAL PAGES IF NEEDED
    # ======================================================
    
    scraped_data = []
    
    if task_info.get("needs_external_url") and other_urls:
        for url in other_urls:
            try:
                sub_html, sub_text = await fetch_page_html_and_text(url)
                scraped_data.append({
                    "url": url,
                    "html": sub_html,
                    "text": sub_text,
                })
                
                # If question asks to scrape something specific, look for it
                if task_info.get("key_terms"):
                    for term in task_info["key_terms"]:
                        # Look for "term: value" or "term = value" patterns
                        pattern = rf"{re.escape(term)}[^\w]*[:\-=]?\s*([^\s<>\"']+)"
                        match = re.search(pattern, sub_text, re.IGNORECASE)
                        if match:
                            potential_answer = match.group(1).strip()
                            if answer_value is None:
                                answer_value = potential_answer
                                llm_info = {
                                    "mode": "scrape_pattern_match",
                                    "source": url,
                                    "term": term,
                                }
                                break
                
                if answer_value:
                    break
                    
            except Exception as e:
                continue

    # ======================================================
    # STEP 4: DOWNLOAD AND PROCESS FILES
    # ======================================================
    
    dataframes = []
    file_data = []
    
    if task_info.get("needs_file_download") or download_links:
        for link in download_links:
            try:
                full_link = normalize_url(quiz_url, link)
                meta, df = download_and_load_data(full_link)
                
                if df is not None:
                    dataframes.append({
                        "url": full_link,
                        "df": df,
                        "meta": meta,
                    })
                    file_data.append(meta)
            except Exception as e:
                continue

    # ======================================================
    # STEP 5: HANDLE API CALLS
    # ======================================================
    
    api_data = []
    
    if task_info.get("task_type") == "api" or "api" in question_text.lower():
        for url in other_urls:
            # Check if URL looks like an API endpoint
            if any(indicator in url.lower() for indicator in ['/api/', '.json', '/v1/', '/v2/']):
                try:
                    # Extract headers if mentioned in the question
                    headers = {}
                    header_patterns = [
                        r"header[s]?[:\s]+(\{[^}]+\})",
                        r"Authorization[:\s]+([^\s<>\"']+)",
                        r"API[- ]?key[:\s]+([^\s<>\"']+)",
                    ]
                    
                    for pattern in header_patterns:
                        match = re.search(pattern, question_text, re.IGNORECASE)
                        if match:
                            try:
                                headers = json.loads(match.group(1))
                            except:
                                # Single header value
                                headers = {"Authorization": match.group(1)}
                    
                    # Try API call
                    response = requests.get(url, headers=headers, timeout=30)
                    api_data.append({
                        "url": url,
                        "status": response.status_code,
                        "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                    })
                except:
                    continue

    # ======================================================
    # STEP 6: INTELLIGENT ANSWER EXTRACTION
    # ======================================================
    
    if answer_value is None:
        # Build comprehensive context for LLM
        context_parts = [
            f"QUESTION: {question_text}",
            f"\nMAIN PAGE CONTENT:\n{text}",
        ]
        
        if scraped_data:
            context_parts.append("\n\nSCRAPED PAGES:")
            for item in scraped_data:
                context_parts.append(f"\nURL: {item['url']}\nContent: {item['text'][:1000]}")
        
        if dataframes:
            context_parts.append("\n\nDATA FILES:")
            for item in dataframes:
                df = item["df"]
                context_parts.append(f"\nFile: {item['url']}")
                context_parts.append(f"Columns: {list(df.columns)}")
                context_parts.append(f"Shape: {df.shape}")
                context_parts.append(f"First few rows:\n{df.head().to_string()}")
        
        if api_data:
            context_parts.append("\n\nAPI RESPONSES:")
            for item in api_data:
                context_parts.append(f"\nAPI: {item['url']}")
                context_parts.append(f"Data: {json.dumps(item['data'])[:500]}")
        
        full_context = "\n".join(context_parts)
        
        # Ask LLM to solve based on all available information
        llm_result = await ask_llm_for_answer(
            question_text=f"""Based on all the information provided, answer this question.
            
Question: {question_text}

Expected answer type: {task_info.get('expected_answer_type', 'unknown')}

Provide ONLY the answer value, nothing else. If it's a number, return just the number. If it's a string, return just the string. If it's JSON, return valid JSON. If you need to create a visualization, return it as a base64 data URI.""",
            context_text=full_context,
            data_notes="",
        )
        
        if llm_result.get("answer") not in [None, "", "unknown"]:
            answer_value = llm_result["answer"]
            llm_info = {
                "mode": "llm_comprehensive",
                "task_type": task_info.get("task_type"),
                "sources_used": {
                    "scraped_pages": len(scraped_data),
                    "data_files": len(dataframes),
                    "api_calls": len(api_data),
                }
            }

    # ======================================================
    # STEP 7: TYPE CONVERSION AND VALIDATION
    # ======================================================
    
    if answer_value is not None:
        # Convert to expected type
        expected_type = task_info.get("expected_answer_type", "string")
        
        if expected_type == "number" and isinstance(answer_value, str):
            try:
                answer_value = float(answer_value) if "." in answer_value else int(answer_value)
            except:
                pass
        
        elif expected_type == "boolean" and isinstance(answer_value, str):
            answer_value = answer_value.lower() in ["true", "yes", "1"]
        
        elif expected_type == "json" and isinstance(answer_value, str):
            try:
                answer_value = json.loads(answer_value)
            except:
                pass

    # ======================================================
    # STEP 8: FINAL FALLBACK
    # ======================================================
    
    if answer_value is None:
        # Last resort: extract first number or prominent text
        num_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if num_match:
            answer_value = float(num_match.group()) if "." in num_match.group() else int(num_match.group())
        else:
            answer_value = "unknown"

    # ======================================================
    # STEP 9: SUBMIT ANSWER
    # ======================================================
    
    if not submit_url:
        return {
            "correct": False,
            "error": "No submit URL found in the question",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }
    
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer_value,
    }
    
    # Validate payload size (must be < 1MB)
    payload_json = json.dumps(payload)
    payload_size = len(payload_json.encode('utf-8'))
    
    if payload_size > 1024 * 1024:
        return {
            "correct": False,
            "error": f"Payload too large: {payload_size} bytes (max 1MB)",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }
    
    try:
        response = requests.post(submit_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        return {
            "correct": bool(result.get("correct")),
            "url": result.get("url"),  # Next quiz URL if provided
            "reason": result.get("reason"),
            "used_answer": answer_value,
            "llm_info": llm_info,
        }
        
    except Exception as e:
        return {
            "correct": False,
            "error": f"Submission failed: {str(e)}",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = 170.0,
) -> Dict[str, Any]:
    """
    Main quiz solver that handles the full quiz sequence:
    - Solves multiple questions in sequence
    - Handles retries within 3-minute window
    - Follows next URLs until quiz is complete
    """
    
    current_url = start_url
    history = []
    attempts_on_current = 0
    max_retries_per_question = 2
    
    while True:
        elapsed = time.time() - start_time
        
        # Check timeout (3 minutes = 180s, use 170s for safety)
        if elapsed > timeout_seconds:
            return {
                "status": "timeout",
                "history": history,
                "elapsed_seconds": elapsed,
                "message": "Quiz timed out after 3 minutes",
            }
        
        time_left = timeout_seconds - elapsed
        
        # Solve the current quiz
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            time_left_seconds=time_left,
        )
        
        attempts_on_current += 1
        
        history.append({
            "url": current_url,
            "attempt": attempts_on_current,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "llm_info": result.get("llm_info"),
            "elapsed_seconds": elapsed,
        })
        
        # Handle the result
        if result.get("correct"):
            # Answer was correct!
            if result.get("url"):
                # There's a next question
                current_url = result["url"]
                attempts_on_current = 0  # Reset attempts for new question
                continue
            else:
                # Quiz is complete!
                return {
                    "status": "completed",
                    "history": history,
                    "elapsed_seconds": time.time() - start_time,
                    "message": "Quiz completed successfully!",
                }
        
        else:
            # Answer was wrong
            next_url = result.get("url")
            
            if next_url:
                # We got a next URL even though answer was wrong
                # Decision: move to next question or retry current?
                
                if attempts_on_current < max_retries_per_question and time_left > 30:
                    # We have time and retries left, try again
                    continue
                else:
                    # Skip to next question
                    current_url = next_url
                    attempts_on_current = 0
                    continue
            
            else:
                # No next URL provided
                if attempts_on_current < max_retries_per_question and time_left > 30:
                    # Retry the same question
                    continue
                else:
                    # Give up on this question
                    return {
                        "status": "failed",
                        "history": history,
                        "elapsed_seconds": time.time() - start_time,
                        "message": f"Failed after {attempts_on_current} attempts",
                        "last_reason": result.get("reason"),
                    }
