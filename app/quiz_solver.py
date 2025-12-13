"""
SPEED-OPTIMIZED quiz_solver.py

"""

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
    extract_api_headers_from_text,
    extract_text_from_pdf,
    process_image,
    normalize_dataframe_to_json,
    parse_github_api_response,
    call_api,
    extract_api_urls_from_text,
)

MAX_GLOBAL_SECONDS = 170
MAX_QUIZ_SECONDS = 7  # Strict per-quiz limit
SKIP_HEAVY_PROCESSING_AFTER = 100  # After 100s, skip heavy operations


def sanitize_question_text(text: str) -> str:
    """Remove submission instructions and JSON examples from question text"""
    text = re.sub(r"Post your answer[\s\S]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\{[^}]{20,}\}", "", text)
    return text.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    """Extract the main question from HTML - FAST"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Quick extraction - first h1/h2 or first 200 chars
    for tag in ['h1', 'h2', 'h3']:
        elem = soup.find(tag)
        if elem:
            return sanitize_question_text(elem.get_text()[:200])
    
    return sanitize_question_text(fallback_text[:200])


async def gather_page_resources_fast(quiz_url: str, html: str, text: str, 
                                     email: str = "", skip_heavy: bool = False) -> Dict[str, Any]:
    """
    SPEED-OPTIMIZED resource gathering
    Skip heavy operations when time is running out
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Submit URL (critical)
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)
    
    # API headers (quick)
    api_headers = extract_api_headers_from_text(text)
    
    # Download data files (essential, but limit to first 2)
    download_links = find_download_links_from_html(html)[:2]  # LIMIT
    dataframes = []
    data_context = []
    pdf_texts = []
    
    for link in download_links:
        try:
            full_url = normalize_url(quiz_url, link)
            
            if full_url.lower().endswith('.pdf'):
                if not skip_heavy:  # Skip PDFs if time is tight
                    pdf_text = extract_text_from_pdf(full_url, api_headers)
                    pdf_texts.append(f"PDF: {full_url}\n{pdf_text[:1000]}")  # Limit PDF text
                continue
            
            meta, df = download_and_load_data(full_url, api_headers)
            
            # Only normalize if small dataset
            normalized = None
            if len(df) <= 100:  # Skip normalization for large datasets
                normalized = normalize_dataframe_to_json(df)
            
            dataframes.append({
                "url": full_url, 
                "df": df,
                "normalized_json": normalized
            })
            data_context.append(meta)
            
        except Exception as e:
            data_context.append(f"Failed to load {link}: {str(e)}")
            continue
    
    # Images (limit to first 2, skip if time is tight)
    images = []
    for img in soup.find_all('img', src=True)[:2]:  # LIMIT
        img_url = normalize_url(quiz_url, img['src'])
        if img_url.startswith('http'):
            images.append(img_url)
    
    image_data = []
    if not skip_heavy and images:  # Skip images if time is tight
        for img_url in images[:2]:
            try:
                img_info = process_image(img_url, api_headers)
                if 'error' not in img_info:
                    image_data.append(img_info)
            except:
                continue
    
    # API calls (essential, limit to first 3)
    api_endpoints = extract_api_urls_from_text(text)[:3]  # LIMIT
    api_responses = []
    
    for api in api_endpoints:
        try:
            result = call_api(api['url'], api['method'], api_headers)
            
            if result.get('success'):
                response_data = result.get('data') or result.get('text', '')
                
                # GitHub API special handling
                if 'api.github.com' in api['url'] and isinstance(response_data, dict):
                    if 'tree' in response_data:
                        prefix_match = re.search(r'prefix[:\s=]+["\']?([^"\'>\s]+)["\']?', text, re.IGNORECASE)
                        ext_match = re.search(r'extension[:\s=]+["\']?([^"\'>\s]+)["\']?', text, re.IGNORECASE)
                        
                        parsed = parse_github_api_response(
                            response_data,
                            filter_prefix=prefix_match.group(1) if prefix_match else "",
                            filter_extension=ext_match.group(1) if ext_match else ""
                        )
                        
                        if email and parsed.get('total_files') is not None:
                            parsed['calculated_count'] = parsed['total_files'] + (len(email) % 2)
                        
                        response_data = parsed
                
                api_responses.append({
                    'url': api['url'],
                    'method': api['method'],
                    'response': response_data
                })
        except:
            continue
    
    # Other URLs (quick scan, limited)
    all_urls = set(re.findall(r"https?://[^\s\"'<>]+", text)[:5])  # LIMIT
    
    return {
        "submit_url": submit_url,
        "dataframes": dataframes,
        "data_context_text": "\n\n".join(data_context),
        "pdf_texts": pdf_texts,
        "image_data": image_data,
        "api_responses": api_responses,
        "api_headers": api_headers,
        "other_urls": list(all_urls)[:5],  # LIMIT
    }


def build_llm_context_fast(question_text: str, page_text: str, 
                           resources: Dict[str, Any], email: str = "") -> str:
    """
    SPEED-OPTIMIZED context building
    Prioritize essential data, minimize noise
    """
    parts = [
        "=== QUESTION ===",
        question_text[:500],  # LIMIT
        "\n=== PAGE CONTENT ===",
        sanitize_question_text(page_text)[:1000],  # LIMIT
    ]
    
    # User info (quick)
    if email:
        parts.append(f"\n=== USER INFO ===")
        parts.append(f"Email: {email}, Length: {len(email)}, Mod 2: {len(email) % 2}")
    
    # Data files (essential)
    if resources["dataframes"]:
        parts.append("\n=== DATA FILES ===")
        for item in resources["dataframes"][:2]:  # LIMIT to 2 files
            df = item["df"]
            parts.append(f"\nFile: {item['url']}")
            parts.append(f"Shape: {df.shape}, Columns: {list(df.columns)}")
            parts.append(f"First 5 rows:\n{df.head(5).to_string()}")  # LIMIT to 5 rows
            
            if item.get("normalized_json"):
                parts.append(f"Normalized JSON (first 3):")
                parts.append(json.dumps(item["normalized_json"][:3], indent=2))  # LIMIT
    
    # Images (if present)
    if resources.get("image_data"):
        parts.append("\n=== IMAGES ===")
        for img in resources["image_data"][:2]:  # LIMIT
            if img.get('dominant_color'):
                parts.append(f"Image: {img['url']}, Color: {img['dominant_color']}")
            if img.get('ocr_text'):
                parts.append(f"OCR: {img['ocr_text'][:200]}")  # LIMIT
    
    # API responses (essential)
    if resources.get("api_responses"):
        parts.append("\n=== API RESPONSES ===")
        for api in resources["api_responses"][:3]:  # LIMIT
            parts.append(f"{api['method']} {api['url']}")
            response = api['response']
            if isinstance(response, dict):
                if 'calculated_count' in response:
                    parts.append(f"Calculated: {response['calculated_count']}")
                else:
                    parts.append(f"{json.dumps(response, indent=2)[:500]}")  # LIMIT
            else:
                parts.append(f"{str(response)[:500]}")  # LIMIT
    
    context = "\n".join(parts)
    
    # Hard limit on context size
    if len(context) > 8000:  # Reduced from 15000
        context = context[:8000] + "\n...[truncated for speed]"
    
    return context


def normalize_answer_type(val):
    """Convert answer to appropriate type - FAST"""
    if val is None:
        return None
    
    if isinstance(val, (bool, int, float, list, dict)):
        return val
        
    if isinstance(val, str):
        s = val.strip()
        
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s) if "." in s else int(s)
        
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except:
                pass
        
        return s
    
    return val


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    remaining: float,
    global_elapsed: float,
    cached_submit_url: Optional[str] = None,
) -> Dict[str, Any]:
    """SPEED-OPTIMIZED single quiz solver"""
    
    quiz_start = time.time()
    
    # Skip heavy processing if time is running out
    skip_heavy = global_elapsed > SKIP_HEAVY_PROCESSING_AFTER
    
    # Fetch page (with timeout)
    try:
        html, text = await fetch_page_html_and_text(quiz_url)
    except Exception as e:
        return {"correct": False, "error": f"Failed to fetch: {str(e)}"}

    question = extract_visible_question(html, text)
    
    # Gather resources (fast mode)
    resources = await gather_page_resources_fast(quiz_url, html, text, email, skip_heavy)
    
    submit_url = resources["submit_url"] or cached_submit_url
    
    if not submit_url:
        return {"correct": False, "error": "Submit URL not found"}

    # Build context (fast mode)
    context = build_llm_context_fast(question, text, resources, email)
    
    # Get answer from LLM
    llm_result = await ask_llm_for_answer(context)

    if "error" in llm_result and llm_result.get("answer") is None:
        return {
            "correct": False,
            "error": f"LLM error: {llm_result['error']}",
            "submit_url": submit_url,
        }
    
    answer = normalize_answer_type(llm_result.get("answer"))

    # Submit answer
    payload = {"email": email, "secret": secret, "url": quiz_url, "answer": answer}

    try:
        response = requests.post(submit_url, json=payload, timeout=10)  # Reduced timeout
        response.raise_for_status()
        data = response.json()
        
    except Exception as e:
        return {"correct": False, "error": f"Submission failed: {str(e)}", "submit_url": submit_url}

    quiz_time = time.time() - quiz_start
    
    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer,
        "response_data": data,
        "submit_url": submit_url,
        "quiz_time": quiz_time,
    }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = MAX_GLOBAL_SECONDS,
) -> Dict[str, Any]:
    """SPEED-OPTIMIZED main quiz solver loop"""

    current_url = start_url
    history = []
    quiz_count = 0
    cached_submit_url = None

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 5:  # Tighter buffer
            return {
                "status": "timeout",
                "history": history,
                "message": f"Timeout after {elapsed:.1f}s, solved {quiz_count}/24 quizzes"
            }

        quiz_count += 1
        
        # Quick progress indicator
        if quiz_count % 5 == 0:
            print(f"\n⚡ Progress: {quiz_count}/24 quizzes, {elapsed:.1f}s elapsed")
        
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            remaining=remaining,
            global_elapsed=elapsed,
            cached_submit_url=cached_submit_url,
        )

        if result.get("submit_url"):
            cached_submit_url = result["submit_url"]

        history.append({
            "quiz_number": quiz_count,
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "elapsed": elapsed,
            "quiz_time": result.get("quiz_time", 0),
        })

        next_url = result.get("url")

        if next_url:
            current_url = next_url
            continue

        # Completed or failed
        if result.get("correct"):
            return {
                "status": "completed",
                "history": history,
                "message": f"✅ Solved all {quiz_count} quizzes in {elapsed:.1f}s!"
            }
        else:
            return {
                "status": "failed",
                "history": history,
                "message": f"❌ Failed on quiz {quiz_count}/24: {result.get('error') or result.get('reason')}"
            }
