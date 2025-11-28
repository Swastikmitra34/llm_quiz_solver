import time
import re
import base64
import json
from typing import Dict, Any, Optional, List
import requests
from bs4 import BeautifulSoup

from .browser import fetch_page_html_and_text
from .llm_interface import ask_llm_for_answer
from .utils import (
    find_download_links_from_html,
    normalize_url,
    download_and_load_data,
)


def decode_base64_in_html(html: str) -> str:
    """
    Decode base64 strings found in the HTML.
    Looks for atob() calls and decodes them.
    """
    decoded_parts = []
    
    # Find all atob() calls with base64 content
    atob_pattern = r'atob\([\'"`]([A-Za-z0-9+/=]+)[\'"`]\)'
    matches = re.findall(atob_pattern, html)
    
    for base64_str in matches:
        try:
            decoded = base64.b64decode(base64_str).decode('utf-8')
            decoded_parts.append(decoded)
        except:
            continue
    
    return '\n'.join(decoded_parts)


def find_submit_url_from_content(text: str, html: str = "") -> Optional[str]:
    """
    Enhanced submit URL finder with multiple extraction strategies.
    Extracts the submit URL from question text or HTML content.
    """
    # FIRST: Try to decode any base64 content in the HTML
    if html:
        decoded_content = decode_base64_in_html(html)
        if decoded_content:
            # Search in the decoded content first
            text = text + "\n" + decoded_content
    
    # Strategy 1: Look for explicit "submit" or "post" mentions with URLs
    patterns = [
        r'[Pp]ost\s+(?:your\s+answer\s+)?to\s+(https?://[^\s<>"\']+)',
        r'[Ss]ubmit\s+(?:your\s+answer\s+)?to\s+(https?://[^\s<>"\']+)',
        r'[Ss]end\s+(?:your\s+answer\s+)?to\s+(https?://[^\s<>"\']+)',
        r'POST\s+to\s+(https?://[^\s<>"\']+)',
        r'endpoint[:\s]+(https?://[^\s<>"\']+/submit[^\s<>"\']*)',
        r'submit[^\w]+(https?://[^\s<>"\']+)',
        r'answer\s+to\s+(https?://[^\s<>"\']+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            url = match.group(1).rstrip('.,;:!?')
            return url
    
    # Strategy 2: Look in HTML for forms or data-submit attributes
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check form actions
        for form in soup.find_all('form'):
            action = form.get('action')
            if action and 'submit' in action.lower():
                if action.startswith('http'):
                    return action
        
        # Check for data-submit or similar attributes
        for tag in soup.find_all(attrs={'data-submit': True}):
            return tag['data-submit']
        
        for tag in soup.find_all(attrs={'data-submit-url': True}):
            return tag['data-submit-url']
    
    # Strategy 3: Look for any URL containing "submit"
    all_urls = re.findall(r'https?://[^\s<>"\']+', text)
    for url in all_urls:
        if 'submit' in url.lower():
            return url.rstrip('.,;:!?')
    
    # Strategy 4: Extract from code blocks or pre tags in HTML
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check code blocks
        for code in soup.find_all(['code', 'pre']):
            code_text = code.get_text()
            for pattern in patterns:
                match = re.search(pattern, code_text, re.IGNORECASE)
                if match:
                    return match.group(1).rstrip('.,;:!?')
            
            # Also check for URLs in JSON payloads inside code blocks
            json_urls = re.findall(r'https?://[^\s<>"\']+', code_text)
            for url in json_urls:
                if 'submit' in url.lower():
                    return url.rstrip('.,;:!?')
    
    return None


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

    # ======================================================
    # STEP 1: FETCH THE QUIZ PAGE (WITH JS RENDERING)
    # ======================================================
    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")
    
    # Decode any base64 content in the page
    decoded_content = decode_base64_in_html(html)
    if decoded_content:
        text = text + "\n\n" + decoded_content
    
    question_text = text.strip()

    answer_value = None
    llm_info = {}

    # ======================================================
    # STEP 2: EXTRACT ALL AVAILABLE RESOURCES
    # ======================================================
    
    # Extract all URLs (links to scrape, APIs, data files)
    all_urls = set()
    
    # From text content (including decoded content)
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
    
    # Extract submit URL (CRITICAL: never hardcode!)
    submit_url = find_submit_url_from_content(text, html)
    
    # If still no submit URL, try to infer from the quiz URL
    if not submit_url:
        # Common pattern: quiz domain + /submit
        parsed_quiz_url = quiz_url.split('?')[0]  # Remove query params
        base_domain = '/'.join(parsed_quiz_url.split('/')[:3])
        potential_submit_url = f"{base_domain}/submit"
        
        # Verify this URL exists in the page content
        if potential_submit_url in text or '/submit' in text:
            submit_url = potential_submit_url
    
    # Separate URLs by type
    current_page_url = quiz_url
    other_urls = [url for url in all_urls if url != current_page_url and url != submit_url]
    
    # Extract downloadable file links
    download_links = find_download_links_from_html(html)

    # ======================================================
    # STEP 3: UNDERSTAND THE QUESTION TYPE
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
    # STEP 4: SCRAPE EXTERNAL PAGES IF NEEDED
    # ======================================================
    
    scraped_data = []
    
    if (task_info.get("needs_external_url") or task_info.get("task_type") == "scraping") and other_urls:
        for url in other_urls:
            try:
                sub_html, sub_text = await fetch_page_html_and_text(url)
                
                # Also decode any base64 in the scraped page
                decoded_sub = decode_base64_in_html(sub_html)
                if decoded_sub:
                    sub_text = sub_text + "\n\n" + decoded_sub
                
                scraped_data.append({
                    "url": url,
                    "html": sub_html,
                    "text": sub_text,
                })
                
                # If question asks to scrape something specific, look for it
                if task_info.get("key_terms"):
                    for term in task_info["key_terms"]:
                        # Look for "term: value" or "term = value" patterns
                        # More flexible pattern matching
                        patterns = [
                            rf"{re.escape(term)}[^\w]*[:\-=]?\s*([^\s<>\"']+)",
                            rf"{re.escape(term)}[^\w]*is[^\w]*([^\s<>\"']+)",
                            rf"<[^>]*{re.escape(term)}[^>]*>([^<]+)</",
                            rf"{re.escape(term)}[:\s]+([A-Za-z0-9_-]+)",
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, sub_text, re.IGNORECASE)
                            if match:
                                potential_answer = match.group(1).strip()
                                if answer_value is None and potential_answer:
                                    answer_value = potential_answer
                                    llm_info = {
                                        "mode": "scrape_pattern_match",
                                        "source": url,
                                        "term": term,
                                    }
                                    break
                        
                        if answer_value:
                            break
                
                if answer_value:
                    break
                    
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue

    # ======================================================
    # STEP 5: DOWNLOAD AND PROCESS FILES
    # ======================================================
    
    dataframes = []
    file_data = []
    
    if task_info.get("needs_file_download") or download_links or task_info.get("task_type") == "data_analysis":
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
                print(f"Error downloading {link}: {e}")
                continue

    # ======================================================
    # STEP 6: HANDLE API CALLS
    # ======================================================
    
    api_data = []
    
    if task_info.get("task_type") == "api" or "api" in question_text.lower():
        for url in other_urls:
            # Check if URL looks like an API endpoint
            if any(indicator in url.lower() for indicator in ['/api/', '.json', '/v1/', '/v2/', 'api.']):
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
                    
                    # Parse response based on content type
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        data = response.json()
                    else:
                        data = response.text
                    
                    api_data.append({
                        "url": url,
                        "status": response.status_code,
                        "data": data,
                    })
                except Exception as e:
                    print(f"Error calling API {url}: {e}")
                    continue

    # ======================================================
    # STEP 7: INTELLIGENT ANSWER EXTRACTION
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
                context_parts.append(f"\nURL: {item['url']}\nContent: {item['text'][:2000]}")
        
        if dataframes:
            context_parts.append("\n\nDATA FILES:")
            for item in dataframes:
                df = item["df"]
                context_parts.append(f"\nFile: {item['url']}")
                context_parts.append(f"Columns: {list(df.columns)}")
                context_parts.append(f"Shape: {df.shape}")
                context_parts.append(f"First few rows:\n{df.head(10).to_string()}")
                
                # Add basic statistics if numeric columns exist
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    context_parts.append(f"Summary statistics:\n{df[numeric_cols].describe().to_string()}")
        
        if api_data:
            context_parts.append("\n\nAPI RESPONSES:")
            for item in api_data:
                context_parts.append(f"\nAPI: {item['url']}")
                context_parts.append(f"Status: {item['status']}")
                data_str = json.dumps(item['data'], indent=2) if isinstance(item['data'], dict) else str(item['data'])
                context_parts.append(f"Data: {data_str[:1000]}")
        
        full_context = "\n".join(context_parts)
        
        # Ask LLM to solve based on all available information
        llm_result = await ask_llm_for_answer(
            question_text=f"""Based on all the information provided, answer this question EXACTLY as requested.

Question: {question_text}

Expected answer type: {task_info.get('expected_answer_type', 'unknown')}

IMPORTANT INSTRUCTIONS:
- Provide ONLY the answer value, nothing else.
- If it's a number, return just the number (e.g., 12345).
- If it's a string, return just the string (e.g., secretcode123).
- If it's JSON, return valid JSON only.
- If you need to create a visualization, return it as a base64 data URI.
- Do NOT include any explanation, preamble, or additional text.
- Do NOT include the question in your response.
- Just the answer value.""",
            context_text=full_context,
            data_notes="",
        )
        
        raw_answer = llm_result.get("answer", "")
        
        if raw_answer and raw_answer not in [None, "", "unknown", "Unknown"]:
            # Clean up the answer
            answer_value = raw_answer.strip()
            
            # Remove common LLM artifacts
            answer_value = re.sub(r'^(Answer:|Response:|Result:)\s*', '', answer_value, flags=re.IGNORECASE)
            answer_value = answer_value.strip()
            
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
    # STEP 8: TYPE CONVERSION AND VALIDATION
    # ======================================================
    
    if answer_value is not None:
        # Convert to expected type
        expected_type = task_info.get("expected_answer_type", "string")
        
        if expected_type == "number" and isinstance(answer_value, str):
            try:
                # Remove any commas or whitespace
                clean_num = answer_value.replace(",", "").replace(" ", "")
                answer_value = float(clean_num) if "." in clean_num else int(clean_num)
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
    # STEP 9: FINAL FALLBACK
    # ======================================================
    
    if answer_value is None or answer_value == "":
        # Last resort: extract first number or prominent text
        num_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if num_match:
            answer_value = float(num_match.group()) if "." in num_match.group() else int(num_match.group())
        else:
            # Try to find any prominent text that might be the answer
            # Look for text in headings or emphasized elements
            for tag in soup.find_all(['h1', 'h2', 'h3', 'strong', 'b']):
                tag_text = tag.get_text().strip()
                if len(tag_text) > 0 and len(tag_text) < 100:
                    answer_value = tag_text
                    break
            
            if not answer_value:
                answer_value = "unknown"

    # ======================================================
    # STEP 10: SUBMIT ANSWER
    # ======================================================
    
    if not submit_url:
        return {
            "correct": False,
            "error": "No submit URL found in the question",
            "used_answer": answer_value,
            "llm_info": llm_info,
            "debug_info": {
                "question_preview": question_text[:500],
                "html_preview": html[:500] if html else None,
                "decoded_content_preview": decoded_content[:500] if decoded_content else None,
            }
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
        
    except requests.exceptions.HTTPError as e:
        return {
            "correct": False,
            "error": f"HTTP Error: {e.response.status_code} - {e.response.text[:200]}",
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
    
    Returns:
        Dictionary with status, history, and results
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
        print(f"\n{'='*60}")
        print(f"Attempting quiz at: {current_url}")
        print(f"Attempt #{attempts_on_current + 1}, Time left: {time_left:.1f}s")
        print(f"{'='*60}\n")
        
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
        
        # Log result
        if result.get("correct"):
            print(f"✓ CORRECT! Answer: {result.get('used_answer')}")
        else:
            print(f"✗ INCORRECT. Answer: {result.get('used_answer')}")
            if result.get("reason"):
                print(f"  Reason: {result.get('reason')}")
            if result.get("error"):
                print(f"  Error: {result.get('error')}")
        
        # Handle the result
        if result.get("correct"):
            # Answer was correct!
            if result.get("url"):
                # There's a next question
                print(f"→ Moving to next question: {result.get('url')}")
                current_url = result["url"]
                attempts_on_current = 0  # Reset attempts for new question
                continue
            else:
                # Quiz is complete!
                print("\n🎉 QUIZ COMPLETED SUCCESSFULLY!")
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
                    print(f"→ Retrying current question (attempt {attempts_on_current + 1}/{max_retries_per_question})")
                    continue
                else:
                    # Skip to next question
                    print(f"→ Skipping to next question: {next_url}")
                    current_url = next_url
                    attempts_on_current = 0
                    continue
            
            else:
                # No next URL provided
                if attempts_on_current < max_retries_per_question and time_left > 30:
                    # Retry the same question
                    print(f"→ Retrying current question (attempt {attempts_on_current + 1}/{max_retries_per_question})")
                    continue
                else:
                    # Give up on this question
                    print(f"\n✗ Failed after {attempts_on_current} attempts")
                    return {
                        "status": "failed",
                        "history": history,
                        "elapsed_seconds": time.time() - start_time,
                        "message": f"Failed after {attempts_on_current} attempts",
                        "last_reason": result.get("reason"),
                    }
