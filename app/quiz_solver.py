

import time
import re
import json
from typing import Dict, Any, List

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
    # Remove "Post your answer..." sections
    text = re.sub(r"Post your answer[\s\S]*", "", text, flags=re.IGNORECASE)
    # Remove JSON code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline JSON examples
    text = re.sub(r"\{[^}]{20,}\}", "", text)
    return text.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    """Extract the main question from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    # Look for question elements
    for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'div']):
        t = elem.get_text().strip()
        if not t or len(t) < 10:
            continue
        # Question indicators
        if any(indicator in t.lower() for indicator in ['question', 'q.', 'what', 'how', 'calculate', 'find', 'download']):
            candidates.append(t)

    raw = "\n".join(candidates[:5]) if candidates else fallback_text
    return sanitize_question_text(raw)


async def gather_page_resources(quiz_url: str, html: str, text: str) -> Dict[str, Any]:
    """
    Enhanced resource gathering:
    - Submit URL
    - Downloadable files (CSV, Excel, PDF, JSON)
    - API endpoints with headers
    - Images
    - Other relevant URLs
    """
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
            
            # Handle PDFs separately
            if full_url.lower().endswith('.pdf'):
                pdf_text = extract_text_from_pdf(full_url, api_headers)
                pdf_texts.append(f"PDF: {full_url}\n{pdf_text[:3000]}")
                continue
            
            # Load data files
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
    
    # Process images (OCR, analysis)
    image_data = []
    for img_url in images[:3]:  # Limit to first 3 images
        try:
            img_info = process_image(img_url, api_headers)
            if 'error' not in img_info:
                image_data.append(img_info)
        except:
            continue
    
    # Find API endpoints
    api_endpoints = extract_api_urls_from_text(text)
    api_responses = []
    
    for api in api_endpoints[:3]:  # Limit API calls
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
        "other_urls": list(other_urls)[:10],  # Limit URLs
    }


def build_llm_context(question_text: str, page_text: str, resources: Dict[str, Any]) -> str:
    """
    Build comprehensive context including all resource types
    """
    parts = [
        "=== QUESTION ===",
        question_text,
        "\n=== PAGE CONTENT ===",
        sanitize_question_text(page_text)[:2000],  # Limit page text
    ]
    
    # Data files
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
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                parts.append(f"\nNumeric column statistics:\n{df[numeric_cols].describe().to_string()}")
    
    # PDF content
    if resources.get("pdf_texts"):
        parts.append("\n=== PDF CONTENT ===")
        parts.extend(resources["pdf_texts"])
    
    # Images with OCR
    if resources.get("image_data"):
        parts.append("\n=== IMAGES ===")
        for img in resources["image_data"]:
            parts.append(f"\nImage: {img.get('url')}")
            parts.append(f"Size: {img.get('size')}, Format: {img.get('format')}")
            if img.get('ocr_text'):
                parts.append(f"OCR Text: {img['ocr_text'][:500]}")
    
    # API responses
    if resources.get("api_responses"):
        parts.append("\n=== API RESPONSES ===")
        for api in resources["api_responses"]:
            parts.append(f"\n{api['method']} {api['url']}")
            parts.append(f"Response: {json.dumps(api['response'], indent=2)[:1000]}")
    
    # API headers if present
    if resources.get("api_headers"):
        parts.append(f"\n=== API HEADERS ===")
        parts.append(json.dumps(resources["api_headers"], indent=2))
    
    # Other URLs
    if resources["other_urls"]:
        parts.append("\n=== OTHER URLS FOUND ===")
        parts.extend(resources["other_urls"][:10])
    
    context = "\n".join(parts)
    
    # Ensure context isn't too large (limit to ~15k chars)
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
        
        # Boolean
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        
        # Number
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s) if "." in s else int(s)
        
        # JSON
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
) -> Dict[str, Any]:
    """Solve a single quiz question with enhanced resource handling"""

    print(f"\n{'='*60}")
    print(f"Solving: {quiz_url}")
    print(f"Time remaining: {remaining:.1f}s")
    print(f"{'='*60}")
    
    # Fetch page content (with JS support from browser module)
    try:
        html, text = await fetch_page_html_and_text(quiz_url)
        print(f"✓ Fetched page ({len(html)} chars HTML, {len(text)} chars text)")
    except Exception as e:
        return {"correct": False, "error": f"Failed to fetch page: {str(e)}"}

    # Extract question
    question = extract_visible_question(html, text)
    print(f"✓ Extracted question: {question[:100]}...")
    
    # Gather all resources
    print("Gathering resources...")
    resources = await gather_page_resources(quiz_url, html, text)
    
    print(f"  - Submit URL: {resources['submit_url']}")
    print(f"  - Data files: {len(resources['dataframes'])}")
    print(f"  - PDF files: {len(resources.get('pdf_texts', []))}")
    print(f"  - Images: {len(resources.get('image_data', []))}")
    print(f"  - API calls: {len(resources.get('api_responses', []))}")
    
    if not resources["submit_url"]:
        return {"correct": False, "error": "Submit URL not found"}

    # Build context for LLM
    context = build_llm_context(question, text, resources)
    print(f"✓ Built context ({len(context)} chars)")

    # Get answer from LLM
    print("Calling LLM...")
    llm_result = await ask_llm_for_answer(context, max_retries=3)

    if "error" in llm_result and llm_result.get("answer") is None:
        print(f"✗ LLM error: {llm_result['error']}")
        return {
            "correct": False,
            "error": f"LLM error: {llm_result['error']}",
            "llm_info": llm_result
        }
    
    answer = normalize_answer_type(llm_result.get("answer"))
    print(f"✓ LLM answer: {answer}")

    # Build submission payload
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    # Check payload size
    payload_size = len(json.dumps(payload).encode("utf-8"))
    if payload_size > 1024 * 1024:
        return {"correct": False, "error": f"Payload too large ({payload_size} bytes > 1MB)"}

    # Submit answer
    print(f"Submitting to: {resources['submit_url']}")
    try:
        response = requests.post(resources["submit_url"], json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Response: correct={data.get('correct')}, next_url={data.get('url')}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Submission failed: {str(e)}")
        return {"correct": False, "error": f"Submission failed: {str(e)}"}
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {str(e)}")
        return {"correct": False, "error": f"Invalid JSON response: {str(e)}"}

    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer,
        "llm_info": llm_result,
        "response_data": data,
    }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = MAX_GLOBAL_SECONDS,
) -> Dict[str, Any]:
    """Main quiz solver loop with enhanced capabilities"""

    current_url = start_url
    history = []
    quiz_count = 0

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 10:  # 10 second safety buffer
            return {
                "status": "timeout",
                "history": history,
                "message": f"Timeout after {elapsed:.1f}s, solved {quiz_count} quizzes"
            }

        # Solve current quiz
        quiz_count += 1
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            remaining=remaining,
        )

        # Record in history
        history.append({
            "quiz_number": quiz_count,
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "elapsed": time.time() - start_time,
        })

        # Get next URL
        next_url = result.get("url")

        # Continue if there's a next URL
        if next_url:
            current_url = next_url
            print(f"\n→ Moving to next quiz: {next_url}")
            continue

        # No next URL - quiz sequence is complete
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
