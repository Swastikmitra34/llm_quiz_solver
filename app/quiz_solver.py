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

    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")
    question_text = text.strip()

    answer_value = None
    llm_info = {}

    # ======================================================
    # 1. AUTO FOLLOW EMBEDDED TASK URLS (JS rendered)
    # ======================================================

    # Find all URLs in the text and HTML
    embedded_urls = set()
    
    # Extract from text
    embedded_urls.update(re.findall(r"https?://[^\s\"'>]+", text))
    
    # Extract from HTML href attributes
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            embedded_urls.add(href)
        elif href.startswith('/'):
            # Convert relative URLs to absolute
            base_url = '/'.join(quiz_url.split('/')[:3])
            embedded_urls.add(base_url + href)

    # Also check for URLs in the HTML that might be in other attributes
    embedded_urls.update(re.findall(r"https?://[^\s\"'>]+", html))

    for url in embedded_urls:
        # Skip if it's the exact same URL as current quiz_url
        if url == quiz_url:
            continue
            
        try:
            sub_html, sub_text = await fetch_page_html_and_text(url)
            sub_text = sub_text.strip()

            # Look for secret/token/code patterns - be more flexible
            patterns = [
                r"(secret|code|token|answer)[^\w]*[:\-]?\s*([A-Za-z0-9_-]{3,})",
                r"secret[^\w]*code[^\w]*[:\-]?\s*([A-Za-z0-9_-]{3,})",
                r"code[^\w]*[:\-]?\s*([A-Za-z0-9_-]{3,})",
                r"token[^\w]*[:\-]?\s*([A-Za-z0-9_-]{3,})",
                r"answer[^\w]*[:\-]?\s*([A-Za-z0-9_-]{3,})",
            ]
            
            for pattern in patterns:
                token_match = re.search(pattern, sub_text, re.IGNORECASE)
                if token_match:
                    # Get the last group which should be the actual value
                    answer_value = token_match.group(token_match.lastindex)
                    llm_info = {"mode": "scrape_auto", "source": url}
                    break
            
            if answer_value:
                break
                
        except Exception as e:
            # Log error if needed but continue trying other URLs
            continue

    # ======================================================
    # 2. API ENDPOINT HANDLING (with headers if provided)
    # ======================================================

    if answer_value is None:
        # Check if there's an API endpoint mentioned
        api_patterns = [
            r"API[^\n]*?(https?://[^\s\"'<>]+)",
            r"endpoint[^\n]*?(https?://[^\s\"'<>]+)",
            r"GET[^\n]*?(https?://[^\s\"'<>]+)",
            r"POST[^\n]*?(https?://[^\s\"'<>]+)",
        ]
        
        for pattern in api_patterns:
            api_match = re.search(pattern, text, re.IGNORECASE)
            if api_match:
                api_url = api_match.group(1)
                
                # Look for headers in the question
                headers = {}
                header_match = re.search(r"header[s]?[:\s]+(.*?)(?:\n|$)", text, re.IGNORECASE)
                if header_match:
                    try:
                        headers = json.loads(header_match.group(1))
                    except:
                        pass
                
                try:
                    api_response = requests.get(api_url, headers=headers, timeout=30)
                    api_data = api_response.json()
                    
                    # Try to extract answer from API response
                    if isinstance(api_data, dict):
                        for key in ['answer', 'result', 'value', 'data']:
                            if key in api_data:
                                answer_value = api_data[key]
                                llm_info = {"mode": "api_fetch", "source": api_url}
                                break
                except:
                    pass
                
                if answer_value:
                    break

    # ======================================================
    # 3. DATA FILE HANDLING (CSV, XLS, JSON, PDF etc)
    # ======================================================

    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context_parts = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            dataframes.append({"df": df, "link": full_link})
            data_context_parts.append(meta)
        except:
            pass

    data_context_text = "\n\n".join(data_context_parts)

    # Handle numeric questions with data files
    if answer_value is None and dataframes:
        question_type = classify_question_type(question_text)
        
        if question_type == "numeric":
            col = extract_column_sum_from_question(question_text)
            if col:
                for item in dataframes:
                    df = item["df"]
                    match_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                    if match_col:
                        answer_value = float(df[match_col].sum())
                        llm_info = {"mode": "numeric_auto", "column": match_col}
                        break
        
        # Handle filtering/sorting/aggregation questions
        elif question_type in ["filter", "aggregate", "sort"]:
            # Let LLM handle complex data operations
            pass

    # ======================================================
    # 4. VISUALIZATION HANDLING (charts, images)
    # ======================================================

    if answer_value is None:
        # Check if question asks for visualization
        viz_keywords = ["chart", "graph", "plot", "visualize", "visualization", "image"]
        if any(keyword in question_text.lower() for keyword in viz_keywords):
            # This needs LLM to generate the visualization
            llm_info["requires_visualization"] = True

    # ======================================================
    # 5. LLM FALLBACK FOR COMPLEX TASKS
    # ======================================================

    if answer_value is None:
        llm_result = await ask_llm_for_answer(
            question_text=question_text,
            context_text=text,
            data_notes=data_context_text,
        )

        if llm_result.get("answer") not in [None, "", "unknown"]:
            answer_value = llm_result["answer"]
            
            # Check if LLM returned a file/image (base64)
            if isinstance(answer_value, str) and answer_value.startswith("data:"):
                llm_info = {"mode": "llm_visualization", "llm_raw": llm_result}
            else:
                llm_info = {"mode": "llm_fallback", "llm_raw": llm_result}

    # ======================================================
    # 6. FINAL DEFENSIVE GUARD
    # ======================================================

    if answer_value is None:
        # Try to extract any number from the text
        num = re.search(r"-?\d+(\.\d+)?", text)
        answer_value = float(num.group()) if num else "unknown"

    # ======================================================
    # 7. PREPARE ANSWER (handle different types)
    # ======================================================

    # Convert answer to appropriate type
    if isinstance(answer_value, str):
        # Check if it's a boolean string
        if answer_value.lower() in ["true", "false"]:
            answer_value = answer_value.lower() == "true"
        # Check if it's a number string
        elif re.match(r"^-?\d+(\.\d+)?$", answer_value):
            answer_value = float(answer_value) if "." in answer_value else int(answer_value)
        # Check if it's JSON
        elif answer_value.startswith("{") or answer_value.startswith("["):
            try:
                answer_value = json.loads(answer_value)
            except:
                pass

    # ======================================================
    # 8. SUBMIT ANSWER
    # ======================================================

    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    if not submit_url:
        return {
            "correct": False,
            "error": "Submit URL not detected",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer_value,
    }

    # Check payload size (must be under 1MB)
    payload_size = len(json.dumps(payload).encode('utf-8'))
    if payload_size > 1024 * 1024:  # 1MB
        return {
            "correct": False,
            "error": f"Payload too large: {payload_size} bytes (max 1MB)",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    try:
        resp = requests.post(submit_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "correct": False,
            "error": f"Submission failed: {str(e)}",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer_value,
        "llm_info": llm_info,
    }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = 170.0,  # 3 minutes = 180 seconds, use 170 for safety
) -> Dict[str, Any]:

    current_url = start_url
    history = []
    last_result = None

    while True:
        elapsed = time.time() - start_time
        
        # Check if we're running out of time
        if elapsed > timeout_seconds:
            return {
                "status": "timeout",
                "history": history,
                "elapsed_seconds": elapsed,
            }

        time_left = timeout_seconds - elapsed

        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            time_left_seconds=time_left,
        )

        history.append({
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "llm_info": result.get("llm_info"),
            "elapsed_seconds": elapsed,
        })

        last_result = result

        # If there's a new URL, continue to next quiz
        if result.get("url"):
            current_url = result["url"]
            
            # If answer was wrong but we got a new URL, we can choose to:
            # 1. Re-submit to current URL (if we think we can fix it)
            # 2. Skip to next URL
            # For now, we'll always move to the next URL
            continue

        # No new URL means either:
        # 1. Quiz is complete (correct answer, no new URL)
        # 2. Answer was wrong and no new URL provided (should retry if time permits)
        
        if result.get("correct"):
            # Quiz completed successfully
            return {
                "status": "finished_correct",
                "history": history,
                "elapsed_seconds": time.time() - start_time,
            }
        else:
            # Answer was wrong and no new URL
            # We could retry here if we have time and a better answer
            # For now, we'll end the quiz
            return {
                "status": "finished_incorrect",
                "history": history,
                "reason": result.get("reason"),
                "elapsed_seconds": time.time() - start_time,
            }
