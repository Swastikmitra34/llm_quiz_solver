"""
quiz_solver.py
Enhanced version supporting all question types with submit URL persistence
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
    call_api,
    extract_api_urls_from_text,
    create_visualization,
)

MAX_GLOBAL_SECONDS = 170


def sanitize_question_text(text: str) -> str:
    """Remove submission instructions and JSON examples from question text"""
    text = re.sub(r"Post your answer[\s\S]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\{[^}]{20,}\}", "", text)
    return text.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    """Extract the main question from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'div']):
        t = elem.get_text().strip()
        if not t or len(t) < 10:
            continue
        if any(indicator in t.lower() for indicator in ['question', 'q.', 'what', 'how', 'calculate', 'find', 'download']):
            candidates.append(t)

    raw = "\n".join(candidates[:5]) if candidates else fallback_text
    return sanitize_question_text(raw)


async def gather_page_resources(quiz_url: str, html: str, text: str) -> Dict[str, Any]:
    """Enhanced resource gathering with all data types"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Submit URL
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)
    
    # API headers if specified
    api_headers = extract_api_headers_from_text(text)
    
    # Download data files
    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context = []
    pdf_texts = []
    
    for link in download_links:
        try:
            full_url = normalize_url(quiz_url, link)
            
            if full_url.lower().endswith('.pdf'):
                pdf_text = extract_text_from_pdf(full_url, api_headers)
                pdf_texts.append(f"PDF: {full_url}\n{pdf_text[:3000]}")
                continue
            
            meta, df = download_and_load_data(full_url, api_headers)
            dataframes.append({"url": full_url, "df": df})
            data_context.append(meta)
            
        except Exception as e:
            data_context.append(f"Failed to load {link}: {str(e)}")
            continue
    
    # Find images
    images = []
    for img in soup.find_all('img', src=True):
        img_url = normalize_url(quiz_url, img['src'])
        if img_url.startswith('http'):
            images.append(img_url)
    
    # Process images
    image_data = []
    for img_url in images[:3]:
        try:
            img_info = process_image(img_url, api_headers)
            if 'error' not in img_info:
                image_data.append(img_info)
        except:
            continue
    
    # Find API endpoints
    api_endpoints = extract_api_urls_from_text(text)
    api_responses = []
    
    for api in api_endpoints[:3]:
        try:
            result = call_api(api['url'], api['method'], api_headers)
            if result.get('success'):
                api_responses.append({
                    'url': api['url'],
                    'method': api['method'],
                    'response': result.get('data') or result.get('text', '')[:1000]
                })
        except:
            continue
    
    # Collect all URLs
    all_urls = set(re.findall(r"https?://[^\s\"'<>]+", text))
    for a in soup.find_all("a", href=True):
        all_urls.add(normalize_url(quiz_url, a["href"]))
    
    other_urls = {u for u in all_urls if u not in {quiz_url, submit_url}}
    
    return {
        "submit_url": submit_url,
        "dataframes": dataframes,
        "data_context_text": "\n\n".join(data_context),
        "pdf_texts": pdf_texts,
        "image_data": image_data,
        "api_responses": api_responses,
        "api_headers": api_headers,
        "other_urls": list(other_urls)[:10],
    }


def build_llm_context(question_text: str, page_text: str, resources: Dict[str, Any]) -> str:
    """Build comprehensive context including all resource types"""
    parts = [
        "=== QUESTION ===",
        question_text,
        "\n=== PAGE CONTENT ===",
        sanitize_question_text(page_text)[:2000],
    ]
    
    if resources["dataframes"]:
        parts.append("\n=== DATA FILES ===")
        for item in resources["dataframes"]:
            df = item["df"]
            parts.extend([
                f"\nFile: {item['url']}",
                f"Shape: {df.shape} (rows × columns)",
                f"Columns: {list(df.columns)}",
                f"\nFirst 10 rows:\n{df.head(10).to_string()}",
            ])
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                parts.append(f"\nNumeric column statistics:\n{df[numeric_cols].describe().to_string()}")
    
    if resources.get("pdf_texts"):
        parts.append("\n=== PDF CONTENT ===")
        parts.extend(resources["pdf_texts"])
    
    if resources.get("image_data"):
        parts.append("\n=== IMAGES ===")
        for img in resources["image_data"]:
            parts.append(f"\nImage: {img.get('url')}")
            parts.append(f"Size: {img.get('size')}, Format: {img.get('format')}")
            if img.get('ocr_text'):
                parts.append(f"OCR Text: {img['ocr_text'][:500]}")
    
    if resources.get("api_responses"):
        parts.append("\n=== API RESPONSES ===")
        for api in resources["api_responses"]:
            parts.append(f"\n{api['method']} {api['url']}")
            parts.append(f"Response: {json.dumps(api['response'], indent=2)[:1000]}")
    
    if resources.get("api_headers"):
        parts.append(f"\n=== API HEADERS ===")
        parts.append(json.dumps(resources["api_headers"], indent=2))
    
    if resources["other_urls"]:
        parts.append("\n=== OTHER URLS FOUND ===")
        parts.extend(resources["other_urls"][:10])
    
    context = "\n".join(parts)
    
    if len(context) > 15000:
        context = context[:15000] + "\n... [truncated]"
    
    return context


def normalize_answer_type(val):
    """Convert answer to appropriate type"""
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
        
        if s.startswith("{") or s.startswith("["):
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
    cached_submit_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Solve a single quiz question with submit URL persistence"""

    print(f"\n{'='*60}")
    print(f"Solving: {quiz_url}")
    print(f"Time remaining: {remaining:.1f}s")
    print(f"{'='*60}")
    
    try:
        html, text = await fetch_page_html_and_text(quiz_url)
        print(f"✓ Fetched page ({len(html)} chars HTML, {len(text)} chars text)")
    except Exception as e:
        return {"correct": False, "error": f"Failed to fetch page: {str(e)}"}

    question = extract_visible_question(html, text)
    print(f"✓ Extracted question: {question[:100]}...")
    
    print("Gathering resources...")
    resources = await gather_page_resources(quiz_url, html, text)
    
    # Use cached submit URL if current page doesn't have one
    submit_url = resources["submit_url"] or cached_submit_url
    
    print(f"  - Submit URL: {submit_url} {'(cached)' if not resources['submit_url'] and cached_submit_url else ''}")
    print(f"  - Data files: {len(resources['dataframes'])}")
    print(f"  - PDF files: {len(resources.get('pdf_texts', []))}")
    print(f"  - Images: {len(resources.get('image_data', []))}")
    print(f"  - API calls: {len(resources.get('api_responses', []))}")
    
    if not submit_url:
        return {"correct": False, "error": "Submit URL not found"}

    context = build_llm_context(question, text, resources)
    print(f"✓ Built context ({len(context)} chars)")
    
    with open("debug_context.txt", "w", encoding="utf-8") as f:
        f.write(context)
    print(f"✓ Context saved to debug_context.txt")

    print("Calling LLM...")
    llm_result = await ask_llm_for_answer(context)

    if "error" in llm_result and llm_result.get("answer") is None:
        print(f"✗ LLM error: {llm_result['error']}")
        print(f"✗ You can check debug_context.txt to see what data was collected")
        return {
            "correct": False,
            "error": f"LLM error: {llm_result['error']}",
            "llm_info": llm_result,
            "context_preview": context[:500] + "...",
            "submit_url": submit_url,
        }
    
    answer = normalize_answer_type(llm_result.get("answer"))
    print(f"✓ LLM answer: {answer}")

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    payload_size = len(json.dumps(payload).encode("utf-8"))
    if payload_size > 1024 * 1024:
        return {"correct": False, "error": f"Payload too large ({payload_size} bytes > 1MB)"}

    print(f"Submitting to: {submit_url}")
    try:
        response = requests.post(submit_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Response: correct={data.get('correct')}, next_url={data.get('url')}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Submission failed: {str(e)}")
        return {"correct": False, "error": f"Submission failed: {str(e)}", "submit_url": submit_url}
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {str(e)}")
        return {"correct": False, "error": f"Invalid JSON response: {str(e)}", "submit_url": submit_url}

    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer,
        "llm_info": llm_result,
        "response_data": data,
        "submit_url": submit_url,  # Return submit URL for caching
    }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = MAX_GLOBAL_SECONDS,
) -> Dict[str, Any]:
    """Main quiz solver loop with submit URL persistence"""

    current_url = start_url
    history = []
    quiz_count = 0
    cached_submit_url = None  # Track submit URL across quizzes

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 10:
            return {
                "status": "timeout",
                "history": history,
                "message": f"Timeout after {elapsed:.1f}s, solved {quiz_count} quizzes"
            }

        quiz_count += 1
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            remaining=remaining,
            cached_submit_url=cached_submit_url,  # Pass cached submit URL
        )

        # Update cached submit URL if we got a new one
        if result.get("submit_url"):
            cached_submit_url = result["submit_url"]

        history.append({
            "quiz_number": quiz_count,
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "elapsed": time.time() - start_time,
        })

        next_url = result.get("url")

        if next_url:
            current_url = next_url
            print(f"\n→ Moving to next quiz: {next_url}")
            continue

        if result.get("correct"):
            return {
                "status": "completed",
                "history": history,
                "message": f"Successfully solved all {quiz_count} quizzes in {elapsed:.1f}s"
            }
        else:
            return {
                "status": "failed",
                "history": history,
                "message": f"Failed on quiz {quiz_count}: {result.get('error') or result.get('reason')}"
            }
