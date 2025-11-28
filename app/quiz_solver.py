import time
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from .browser import fetch_page_html_and_text
from .llm_interface import ask_llm_for_answer
from .utils import (
    find_submit_url_from_text,
    find_download_links_from_html,
    normalize_url,
    download_and_load_data,
    extract_column_sum_from_question,
)


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: int = 170,
) -> Dict[str, Any]:
    """
    Main orchestration function for solving quiz chain.
    """
    
    visited_urls: List[str] = []
    results: List[Dict[str, Any]] = []
    current_url = start_url
    max_attempts_per_url = 3
    
    print(f"[SOLVER] Starting quiz chain from: {start_url}")
    
    while current_url:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"[SOLVER] Timeout reached ({elapsed:.1f}s)")
            return {
                "status": "timeout",
                "visited_urls": visited_urls,
                "results": results,
                "elapsed_seconds": elapsed,
            }
        
        if current_url in visited_urls:
            print(f"[SOLVER] Already visited {current_url}, breaking loop")
            break
        
        visited_urls.append(current_url)
        print(f"\n[SOLVER] Processing URL: {current_url}")
        print(f"[SOLVER] Time elapsed: {elapsed:.1f}s / {timeout_seconds}s")
        
        quiz_result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            start_time=start_time,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts_per_url,
        )
        
        results.append(quiz_result)
        next_url = quiz_result.get("next_url")
        
        if quiz_result.get("status") == "correct":
            print(f"[SOLVER] ✓ Correct answer for {current_url}")
            if next_url:
                current_url = next_url
                print(f"[SOLVER] → Moving to next URL: {next_url}")
            else:
                print("[SOLVER] ✓ Quiz chain complete (no next URL)")
                break
        
        elif quiz_result.get("status") == "incorrect":
            print(f"[SOLVER] ✗ Incorrect answer for {current_url}")
            if next_url:
                print(f"[SOLVER] → Skipping to next URL: {next_url}")
                current_url = next_url
            else:
                print("[SOLVER] No next URL provided, stopping")
                break
        
        else:
            print(f"[SOLVER] Error processing {current_url}: {quiz_result.get('error')}")
            if next_url:
                print(f"[SOLVER] → Attempting next URL despite error: {next_url}")
                current_url = next_url
            else:
                break
    
    final_elapsed = time.time() - start_time
    return {
        "status": "completed",
        "visited_urls": visited_urls,
        "results": results,
        "elapsed_seconds": final_elapsed,
    }


def extract_urls_from_text(text: str, base_url: str) -> List[str]:
    """Extract all URLs from text, including relative URLs."""
    urls = []
    
    # Match full URLs
    full_url_pattern = r'https?://[^\s<>"\'()]+(?:[^\s<>"\'().,;!?])'
    urls.extend(re.findall(full_url_pattern, text))
    
    # Match relative URLs with context
    relative_url_pattern = r'(?:href=|src=|visit|scrape|download|fetch|get|from)[\s"\']*(\/[^\s<>"\'()]+)'
    relative_matches = re.findall(relative_url_pattern, text, re.IGNORECASE)
    
    for rel_url in relative_matches:
        abs_url = urljoin(base_url, rel_url.strip('"\''))
        urls.append(abs_url)
    
    # Standalone relative URLs
    standalone_relative = r'(?:^|\s)(\/[a-zA-Z0-9_\-\/\?&=%.]+)'
    standalone_matches = re.findall(standalone_relative, text, re.MULTILINE)
    
    for rel_url in standalone_matches:
        abs_url = urljoin(base_url, rel_url.strip())
        urls.append(abs_url)
    
    # Deduplicate
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def find_submit_url_enhanced(html: str, visible_text: str, base_url: str) -> Optional[str]:
    """Enhanced submit URL detection supporting relative and absolute URLs."""
    
    # Pattern 1: Full URLs with /submit
    pattern1 = r'https?://[^\s<>"\']+/submit[^\s<>"\']*'
    
    for text in [html, visible_text]:
        match = re.search(pattern1, text)
        if match:
            url = match.group(0).rstrip('.,;!?)')
            print(f"[SUBMIT URL] Found full URL: {url}")
            return url
    
    # Pattern 2: Relative URLs with POST/submit keywords
    pattern2 = r'(?:POST|post|submit|send).*?(?:to|at|endpoint|url)[:\s]+([/][^\s<>"\']+)'
    
    for text in [visible_text, html]:
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            rel_url = match.group(1).rstrip('.,;!?)')
            abs_url = urljoin(base_url, rel_url)
            print(f"[SUBMIT URL] Found relative URL: {rel_url} -> {abs_url}")
            return abs_url
    
    # Pattern 3: Any /submit path
    pattern3 = r'([/]submit[^\s<>"\']*)'
    
    for text in [visible_text, html]:
        match = re.search(pattern3, text)
        if match:
            rel_url = match.group(1).rstrip('.,;!?)')
            abs_url = urljoin(base_url, rel_url)
            print(f"[SUBMIT URL] Found /submit path: {rel_url} -> {abs_url}")
            return abs_url
    
    # Pattern 4: Check HTML forms
    soup = BeautifulSoup(html, 'html.parser')
    forms = soup.find_all('form')
    for form in forms:
        action = form.get('action')
        if action:
            abs_url = urljoin(base_url, action)
            print(f"[SUBMIT URL] Found form action: {action} -> {abs_url}")
            return abs_url
    
    print("[SUBMIT URL] No submit URL found")
    return None


async def scrape_additional_urls(urls: List[str], quiz_url: str) -> str:
    """Scrape additional URLs mentioned in quiz instructions."""
    context = ""
    
    parsed_quiz = urlparse(quiz_url)
    additional_urls = [u for u in urls if urlparse(u).path != parsed_quiz.path]
    
    if not additional_urls:
        return context
    
    print(f"[SCRAPE] Found {len(additional_urls)} additional URLs to scrape")
    
    for url in additional_urls[:5]:
        try:
            print(f"[SCRAPE] Fetching: {url}")
            html, visible_text = await fetch_page_html_and_text(url)
            
            if visible_text:
                context += f"\n\n=== CONTENT FROM {url} ===\n"
                context += visible_text[:3000]
                context += "\n=== END CONTENT ===\n"
                print(f"[SCRAPE] Successfully scraped {len(visible_text)} chars from {url}")
            
        except Exception as e:
            print(f"[SCRAPE] Failed to fetch {url}: {e}")
            context += f"\n\n=== FAILED TO FETCH {url}: {str(e)} ===\n"
    
    return context


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    start_time: float,
    timeout_seconds: int,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Solve a single quiz page with comprehensive error handling."""
    
    result = {
        "url": quiz_url,
        "status": "error",
        "answer": None,
        "next_url": None,
        "attempts": 0,
        "reason": None,
    }
    
    try:
        # Step 1: Fetch page content
        print(f"[QUIZ] Fetching page: {quiz_url}")
        
        try:
            html, visible_text = await fetch_page_html_and_text(quiz_url)
        except Exception as e:
            result["error"] = f"Failed to fetch page: {str(e)}"
            return result
        
        if not visible_text:
            result["error"] = "No visible text extracted from page"
            return result
        
        print(f"[QUIZ] Extracted {len(visible_text)} chars of visible text")
        print(f"[QUIZ] Visible text preview:\n{visible_text[:800]}\n...")
        
        # Step 2: Find submit URL
        submit_url = find_submit_url_enhanced(html, visible_text, quiz_url)
        
        if not submit_url:
            submit_url = find_submit_url_from_text(html)
            if not submit_url:
                submit_url = find_submit_url_from_text(visible_text)
        
        if not submit_url:
            result["error"] = "Could not find submit URL in page"
            print(f"[QUIZ] ❌ Submit URL not found")
            print(f"[QUIZ] Full visible text:\n{visible_text}")
            return result
        
        print(f"[QUIZ] ✓ Submit URL: {submit_url}")
        
        # Step 3: Extract all URLs from page
        all_urls = extract_urls_from_text(visible_text, quiz_url)
        print(f"[QUIZ] Found {len(all_urls)} URLs in page: {all_urls}")
        
        # Step 4: Scrape additional URLs (for multi-step tasks)
        scraped_context = ""
        if all_urls:
            scraped_context = await scrape_additional_urls(all_urls, quiz_url)
        
        # Step 5: Process data sources
        data_context = ""
        data_links = find_download_links_from_html(html)
        
        # Add data URLs from text
        text_urls = [u for u in all_urls if any(ext in u.lower() for ext in 
                     ['.csv', '.json', '.xlsx', '.xls', '.pdf', '.txt', '.xml'])]
        data_links.extend(text_urls)
        
        if data_links:
            print(f"[QUIZ] Found {len(data_links)} data link(s)")
            for link in data_links[:3]:
                full_url = normalize_url(quiz_url, link)
                print(f"[QUIZ] Downloading: {full_url}")
                
                try:
                    meta, df = download_and_load_data(full_url)
                    
                    col_name = extract_column_sum_from_question(visible_text)
                    if col_name and df is not None and col_name in df.columns:
                        computed_sum = df[col_name].sum()
                        meta += f"\n\n**COMPUTED: sum of '{col_name}' column = {computed_sum}**"
                        print(f"[QUIZ] Computed sum of '{col_name}': {computed_sum}")
                    
                    data_context += f"\n\n--- Data from {full_url} ---\n{meta}\n"
                    
                except Exception as e:
                    print(f"[QUIZ] Failed to process {full_url}: {e}")
                    data_context += f"\n\n--- Failed to load {full_url}: {str(e)} ---\n"
        
        # Step 6: Combine all context
        full_context = data_context
        if scraped_context:
            full_context += "\n\n" + scraped_context
        
        # Step 7: Attempt submission with retries
        for attempt in range(1, max_attempts + 1):
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                result["status"] = "timeout"
                result["reason"] = "Exceeded time limit"
                return result
            
            result["attempts"] = attempt
            print(f"\n[QUIZ] Attempt {attempt}/{max_attempts}")
            
            # Add progressive delay between attempts to avoid rate limits
            if attempt > 1:
                delay = 5 * attempt  # 10s, 15s, 20s...
                print(f"[QUIZ] Waiting {delay}s before retry to avoid rate limits...")
                await asyncio.sleep(delay)
            
            # Get answer from LLM
            try:
                llm_response = await ask_llm_for_answer(
                    question_text=visible_text,
                    context_text=html[:5000],
                    data_notes=full_context,
                )
            except Exception as e:
                print(f"[QUIZ] LLM invocation error: {e}")
                result["error"] = f"LLM invocation failed: {str(e)}"
                
                # Check if rate limit
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < max_attempts:
                        print(f"[QUIZ] Rate limit detected, will retry after delay")
                        continue
                
                if attempt < max_attempts:
                    await asyncio.sleep(3)
                    continue
                else:
                    return result
            
            if "error" in llm_response:
                error_msg = str(llm_response['error'])
                print(f"[QUIZ] LLM error: {error_msg}")
                
                # Check for rate limit in error
                if "429" in error_msg or "rate limit" in error_msg.lower() or "too many" in error_msg.lower():
                    result["error"] = "Rate limit exceeded after retries"
                    if attempt < max_attempts:
                        print(f"[QUIZ] Rate limit detected, will retry with longer delay")
                        continue
                    else:
                        return result
                
                result["error"] = error_msg
                if attempt < max_attempts:
                    await asyncio.sleep(3)
                    continue
                else:
                    return result
            
            answer = llm_response.get("answer")
            
            # Check for nested payload structure
            if isinstance(answer, dict) and all(k in answer for k in ["email", "secret", "url", "answer"]):
                print(f"[QUIZ] ⚠️  LLM returned example payload structure")
                nested_answer = answer.get("answer")
                
                if isinstance(nested_answer, str):
                    lower = nested_answer.lower()
                    if any(x in lower for x in ["placeholder", "anything", "your", "example", "student"]):
                        print(f"[QUIZ] Nested answer '{nested_answer}' is a placeholder")
                        result["error"] = "LLM returned placeholder answer"
                        if attempt < max_attempts:
                            continue
                        else:
                            return result
                
                answer = nested_answer
                print(f"[QUIZ] Extracted nested answer: {answer}")
            
            # Check for placeholder strings
            if isinstance(answer, str):
                lower = answer.lower()
                if any(x in lower for x in ["placeholder", "anything you want", "your answer", 
                                            "example", "student", "the secret code you scraped"]):
                    print(f"[QUIZ] Answer '{answer}' is a placeholder")
                    result["error"] = "LLM returned placeholder answer"
                    if attempt < max_attempts:
                        continue
                    else:
                        return result
            
            if answer is None:
                print("[QUIZ] LLM returned null answer")
                result["error"] = "LLM could not determine answer"
                if attempt < max_attempts:
                    await asyncio.sleep(3)
                    continue
                else:
                    return result
            
            print(f"[QUIZ] LLM answer: {answer} (type: {type(answer).__name__})")
            result["answer"] = answer
            
            # Submit answer
            try:
                submission_result = submit_answer(
                    submit_url=submit_url,
                    email=email,
                    secret=secret,
                    quiz_url=quiz_url,
                    answer=answer,
                )
            except Exception as e:
                print(f"[QUIZ] Submission exception: {e}")
                submission_result = {"success": False, "error": str(e)}
            
            if submission_result.get("success"):
                is_correct = submission_result.get("correct", False)
                next_url = submission_result.get("next_url")
                reason = submission_result.get("reason")
                
                result["status"] = "correct" if is_correct else "incorrect"
                result["next_url"] = next_url
                result["reason"] = reason
                
                if is_correct:
                    print(f"[QUIZ] ✓ Answer accepted!")
                    return result
                else:
                    print(f"[QUIZ] ✗ Wrong answer: {reason}")
                    if next_url:
                        print(f"[QUIZ] Server provided next URL despite wrong answer")
                        return result
                    if attempt < max_attempts:
                        print(f"[QUIZ] Retrying... ({attempt + 1}/{max_attempts})")
                        continue
                    else:
                        return result
            else:
                error = submission_result.get("error", "Unknown submission error")
                print(f"[QUIZ] Submission failed: {error}")
                result["error"] = error
                
                if attempt < max_attempts:
                    await asyncio.sleep(3)
                    continue
                else:
                    return result
        
        return result
        
    except Exception as e:
        print(f"[QUIZ] Exception in solve_single_quiz: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        return result


def submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
) -> Dict[str, Any]:
    """Submit answer to the quiz endpoint."""
    
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }
    
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"Failed to serialize payload: {str(e)}"}
    
    if len(payload_json) > 1_000_000:
        return {"success": False, "error": f"Payload too large: {len(payload_json)} bytes"}
    
    try:
        print(f"[SUBMIT] POST to {submit_url}")
        print(f"[SUBMIT] Payload: {payload_json[:300]}")
        
        response = requests.post(
            submit_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        print(f"[SUBMIT] Response status: {response.status_code}")
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
        
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON response: {str(e)}"
            }
        
        print(f"[SUBMIT] Response: {json.dumps(response_data, indent=2)}")
        
        return {
            "success": True,
            "correct": response_data.get("correct", False),
            "next_url": response_data.get("url"),
            "reason": response_data.get("reason"),
        }
        
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout (30s)"}
    
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}
