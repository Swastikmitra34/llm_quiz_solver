import time
import re
import json
from typing import Dict, Any, List, Optional
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
    
    Args:
        email: Student email
        secret: Student secret
        start_url: Initial quiz URL
        start_time: Unix timestamp when POST was received
        timeout_seconds: Maximum time allowed (default 170s, 10s buffer)
    
    Returns:
        Dictionary with completion status and results
    """
    
    visited_urls: List[str] = []
    results: List[Dict[str, Any]] = []
    current_url = start_url
    max_attempts_per_url = 2
    
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
        
        # Prevent infinite loops
        if current_url in visited_urls:
            print(f"[SOLVER] Already visited {current_url}, breaking loop")
            break
        
        visited_urls.append(current_url)
        print(f"\n[SOLVER] Processing URL: {current_url}")
        print(f"[SOLVER] Time elapsed: {elapsed:.1f}s / {timeout_seconds}s")
        
        # Attempt to solve current quiz page
        quiz_result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            start_time=start_time,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts_per_url,
        )
        
        results.append(quiz_result)
        
        # Extract next URL from result
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
            # If server provides next URL even on failure, we can skip ahead
            if next_url:
                print(f"[SOLVER] → Skipping to next URL: {next_url}")
                current_url = next_url
            else:
                print("[SOLVER] No next URL provided, stopping")
                break
        
        else:
            # Error or unknown status
            print(f"[SOLVER] Error processing {current_url}: {quiz_result.get('error')}")
            break
    
    final_elapsed = time.time() - start_time
    return {
        "status": "completed",
        "visited_urls": visited_urls,
        "results": results,
        "elapsed_seconds": final_elapsed,
    }


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    start_time: float,
    timeout_seconds: int,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """
    Solve a single quiz page with retry logic.
    
    Returns:
        {
            "url": quiz_url,
            "status": "correct" | "incorrect" | "error",
            "answer": submitted_answer,
            "next_url": next_quiz_url or None,
            "attempts": attempt_count,
            "reason": failure_reason if any,
        }
    """
    
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
        html, visible_text = await fetch_page_html_and_text(quiz_url)
        
        if not visible_text:
            result["error"] = "No visible text extracted from page"
            return result
        
        print(f"[QUIZ] Extracted {len(visible_text)} chars of visible text")
        
        # Step 2: Find submit URL
        submit_url = find_submit_url_from_text(html)
        if not submit_url:
            submit_url = find_submit_url_from_text(visible_text)
        
        if not submit_url:
            result["error"] = "Could not find submit URL in page"
            return result
        
        print(f"[QUIZ] Submit URL: {submit_url}")
        
        # Step 3: Detect and process data sources
        data_context = ""
        data_links = find_download_links_from_html(html)
        
        if data_links:
            print(f"[QUIZ] Found {len(data_links)} data link(s)")
            for link in data_links[:3]:  # Process max 3 data files
                full_url = normalize_url(quiz_url, link)
                print(f"[QUIZ] Downloading: {full_url}")
                
                try:
                    meta, df = download_and_load_data(full_url)
                    
                    # Simple heuristic: if question asks for sum of a column, compute it
                    col_name = extract_column_sum_from_question(visible_text)
                    if col_name and col_name in df.columns:
                        computed_sum = df[col_name].sum()
                        meta += f"\n\n**COMPUTED: sum of '{col_name}' column = {computed_sum}**"
                        print(f"[QUIZ] Computed sum of '{col_name}': {computed_sum}")
                    
                    data_context += f"\n\n--- Data from {full_url} ---\n{meta}\n"
                    
                except Exception as e:
                    print(f"[QUIZ] Failed to process {full_url}: {e}")
                    data_context += f"\n\n--- Failed to load {full_url}: {e} ---\n"
        
        # Step 4: Attempt submission (with retries)
        for attempt in range(1, max_attempts + 1):
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                result["status"] = "timeout"
                result["reason"] = "Exceeded time limit"
                return result
            
            result["attempts"] = attempt
            print(f"\n[QUIZ] Attempt {attempt}/{max_attempts}")
            
            # Get answer from LLM
            llm_response = await ask_llm_for_answer(
                question_text=visible_text,
                context_text=html[:5000],  # Limit HTML context
                data_notes=data_context,
            )
            
            if "error" in llm_response:
                print(f"[QUIZ] LLM error: {llm_response['error']}")
                result["error"] = llm_response["error"]
                continue
            
            answer = llm_response.get("answer")
            if answer is None:
                print("[QUIZ] LLM returned null answer")
                result["error"] = "LLM could not determine answer"
                continue
            
            print(f"[QUIZ] LLM answer: {answer} (type: {type(answer).__name__})")
            result["answer"] = answer
            
            # Submit answer
            submission_result = submit_answer(
                submit_url=submit_url,
                email=email,
                secret=secret,
                quiz_url=quiz_url,
                answer=answer,
            )
            
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
                    # If we got a next URL even on wrong answer, we can proceed
                    if next_url:
                        print(f"[QUIZ] Server provided next URL despite wrong answer")
                        return result
                    # Otherwise retry if attempts remain
                    if attempt < max_attempts:
                        print(f"[QUIZ] Retrying... ({attempt + 1}/{max_attempts})")
                        continue
                    else:
                        return result
            else:
                # Submission failed (network, server error, etc.)
                error = submission_result.get("error", "Unknown submission error")
                print(f"[QUIZ] Submission failed: {error}")
                result["error"] = error
                
                if attempt < max_attempts:
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    return result
        
        return result
        
    except Exception as e:
        print(f"[QUIZ] Exception in solve_single_quiz: {e}")
        result["error"] = str(e)
        return result


def submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
) -> Dict[str, Any]:
    """
    Submit answer to the quiz endpoint.
    
    Returns:
        {
            "success": bool,
            "correct": bool (if success),
            "next_url": str or None (if success),
            "reason": str or None (if success),
            "error": str (if not success),
        }
    """
    
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }
    
    # Ensure payload is under 1MB
    payload_json = json.dumps(payload)
    if len(payload_json) > 1_000_000:
        return {
            "success": False,
            "error": f"Payload too large: {len(payload_json)} bytes (max 1MB)"
        }
    
    try:
        print(f"[SUBMIT] POST to {submit_url}")
        print(f"[SUBMIT] Payload size: {len(payload_json)} bytes")
        
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
        
        response_data = response.json()
        print(f"[SUBMIT] Response: {json.dumps(response_data, indent=2)}")
        
        return {
            "success": True,
            "correct": response_data.get("correct", False),
            "next_url": response_data.get("url"),
            "reason": response_data.get("reason"),
        }
        
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout (30s)"}
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON response: {str(e)}"}
    
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}
