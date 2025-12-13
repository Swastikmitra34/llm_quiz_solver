"""
quiz_solver.py
Compatible with existing utils.py - no missing imports
Increased time limits for 24-question assessments
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

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_GLOBAL_SECONDS = 300  # 5 minutes for all quizzes
MAX_QUIZ_SECONDS = 15     # Warn if a quiz takes longer than this


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
        if any(indicator in t.lower() for indicator in ['question', 'q.', 'what', 'how', 'calculate', 'find', 'download', 'transcribe', 'color', 'count']):
            candidates.append(t)

    raw = "\n".join(candidates[:5]) if candidates else fallback_text
    return sanitize_question_text(raw)


async def gather_page_resources(quiz_url: str, html: str, text: str, email: str = "") -> Dict[str, Any]:
    """
    Gather page resources - compatible with existing utils.py
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Submit URL
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)
    
    # API headers if specified
    api_headers = extract_api_headers_from_text(text)
    
    # Download and process data files
    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context = []
    pdf_texts = []
    
    for link in download_links[:3]:  # Limit to 3 files
        try:
            full_url = normalize_url(quiz_url, link)
            
            if full_url.lower().endswith('.pdf'):
                pdf_text = extract_text_from_pdf(full_url, api_headers)
                pdf_texts.append(f"PDF: {full_url}\n{pdf_text[:2000]}")
                continue
            
            meta, df = download_and_load_data(full_url, api_headers)
            dataframes.append({"url": full_url, "df": df})
            data_context.append(meta)
            
        except Exception as e:
            data_context.append(f"Failed to load {link}: {str(e)}")
            continue
    
    # Find and process images
    images = []
    for img in soup.find_all('img', src=True)[:3]:  # Limit to 3 images
        img_url = normalize_url(quiz_url, img['src'])
        if img_url.startswith('http'):
            images.append(img_url)
    
    image_data = []
    for img_url in images:
        try:
            img_info = process_image(img_url, api_headers)
            if 'error' not in img_info:
                image_data.append(img_info)
        except:
            continue
    
    # Find and call API endpoints
    api_endpoints = extract_api_urls_from_text(text)[:5]  # Limit to 5
    api_responses = []
    
    for api in api_endpoints:
        try:
            result = call_api(api['url'], api['method'], api_headers)
            
            if result.get('success'):
                response_data = result.get('data') or result.get('text', '')
                
                # Special handling for GitHub API
                if 'api.github.com' in api['url'] and isinstance(response_data, dict):
                    if 'tree' in response_data:
                        # Parse GitHub tree
                        prefix_match = re.search(r'prefix[:\s=]+["\']?([^"\'>\s]+)["\']?', text, re.IGNORECASE)
                        ext_match = re.search(r'extension[:\s=]+["\']?([^"\'>\s]+)["\']?', text, re.IGNORECASE)
                        
                        prefix = prefix_match.group(1) if prefix_match else ""
                        extension = ext_match.group(1) if ext_match else ""
                        
                        # Count files matching criteria
                        tree_items = response_data.get('tree', [])
                        count = 0
                        files = []
                        
                        for item in tree_items:
                            path = item.get('path', '')
                            item_type = item.get('type', '')
                            
                            if item_type == 'blob':  # File
                                if (not prefix or path.startswith(prefix)) and \
                                   (not extension or path.endswith(extension)):
                                    count += 1
                                    files.append(path)
                        
                        # Apply email formula if email provided
                        calculated_count = count
                        if email:
                            calculated_count = count + (len(email) % 2)
                        
                        response_data = {
                            'total_files': count,
                            'calculated_count': calculated_count,
                            'files': files[:10],  # First 10 files
                            'filter_prefix': prefix,
                            'filter_extension': extension
                        }
                
                api_responses.append({
                    'url': api['url'],
                    'method': api['method'],
                    'response': response_data
                })
        except:
            continue
    
    # Collect all URLs
    all_urls = set(re.findall(r"https?://[^\s\"'<>]+", text)[:10])
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
        "other_urls": list(other_urls)[:8],
    }


def build_llm_context(question_text: str, page_text: str, resources: Dict[str, Any], email: str = "") -> str:
    """Build context for LLM"""
    parts = [
        "=== QUESTION ===",
        question_text[:800],
        "\n=== PAGE CONTENT ===",
        sanitize_question_text(page_text)[:1500],
    ]
    
    # User info
    if email:
        parts.append(f"\n=== USER INFO ===")
        parts.append(f"Email: {email}")
        parts.append(f"Email length: {len(email)}")
        parts.append(f"Email length mod 2: {len(email) % 2}")
    
    # Data files
    if resources["dataframes"]:
        parts.append("\n=== DATA FILES ===")
        for item in resources["dataframes"][:3]:
            df = item["df"]
            parts.append(f"\nFile: {item['url']}")
            parts.append(f"Shape: {df.shape} (rows × columns)")
            parts.append(f"Columns: {list(df.columns)}")
            parts.append(f"\nFirst 8 rows:\n{df.head(8).to_string()}")
            
            # Add statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                parts.append(f"\nNumeric statistics:\n{df[numeric_cols].describe().to_string()}")
    
    # PDF content
    if resources.get("pdf_texts"):
        parts.append("\n=== PDF CONTENT ===")
        parts.extend(resources["pdf_texts"])
    
    # Images
    if resources.get("image_data"):
        parts.append("\n=== IMAGES ===")
        for img in resources["image_data"][:3]:
            parts.append(f"\nImage: {img.get('url')}")
            parts.append(f"Size: {img.get('size')}, Format: {img.get('format')}")
            if img.get('ocr_text') and len(img['ocr_text'].strip()) > 0:
                parts.append(f"OCR Text:\n{img['ocr_text'][:300]}")
    
    # API responses
    if resources.get("api_responses"):
        parts.append("\n=== API RESPONSES ===")
        for api in resources["api_responses"][:5]:
            parts.append(f"\n{api['method']} {api['url']}")
            
            response = api['response']
            if isinstance(response, dict):
                if 'calculated_count' in response:
                    parts.append(f"Total files: {response['total_files']}")
                    parts.append(f"Calculated count (with email formula): {response['calculated_count']}")
                    if response.get('files'):
                        parts.append(f"Files found:")
                        for f in response['files'][:10]:
                            parts.append(f"  - {f}")
                else:
                    parts.append(f"Response:\n{json.dumps(response, indent=2)[:800]}")
            else:
                parts.append(f"Response: {str(response)[:800]}")
    
    # API headers
    if resources.get("api_headers"):
        parts.append(f"\n=== API HEADERS ===")
        parts.append(json.dumps(resources["api_headers"], indent=2))
    
    # Other URLs
    if resources["other_urls"]:
        parts.append("\n=== OTHER URLS FOUND ===")
        parts.extend(resources["other_urls"][:8])
    
    context = "\n".join(parts)
    
    # Limit context size
    if len(context) > 12000:
        context = context[:12000] + "\n... [truncated]"
    
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
    """Solve a single quiz question"""
    
    quiz_start_time = time.time()

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
    resources = await gather_page_resources(quiz_url, html, text, email)
    
    # Use cached submit URL if current page doesn't have one
    submit_url = resources["submit_url"] or cached_submit_url
    
    print(f"  - Submit URL: {submit_url} {'(cached)' if not resources['submit_url'] and cached_submit_url else ''}")
    print(f"  - Data files: {len(resources['dataframes'])}")
    print(f"  - PDF files: {len(resources.get('pdf_texts', []))}")
    print(f"  - Images: {len(resources.get('image_data', []))}")
    print(f"  - API calls: {len(resources.get('api_responses', []))}")
    
    if not submit_url:
        return {"correct": False, "error": "Submit URL not found"}

    context = build_llm_context(question, text, resources, email)
    print(f"✓ Built context ({len(context)} chars)")
    
    with open("debug_context.txt", "w", encoding="utf-8") as f:
        f.write(context)
    print(f"✓ Context saved to debug_context.txt")

    print("Calling LLM...")
    llm_result = await ask_llm_for_answer(context)

    if "error" in llm_result and llm_result.get("answer") is None:
        print(f"✗ LLM error: {llm_result['error']}")
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
        response = requests.post(submit_url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Response: correct={data.get('correct')}, next_url={data.get('url')}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Submission failed: {str(e)}")
        return {"correct": False, "error": f"Submission failed: {str(e)}", "submit_url": submit_url}
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {str(e)}")
        return {"correct": False, "error": f"Invalid JSON response: {str(e)}", "submit_url": submit_url}

    quiz_time = time.time() - quiz_start_time
    status = "✓" if data.get("correct") else "✗"
    print(f"{status} Quiz completed in {quiz_time:.2f}s")

    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer,
        "llm_info": llm_result,
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
    """Main quiz solver loop"""

    current_url = start_url
    history = []
    quiz_count = 0
    cached_submit_url = None

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 10:
            return {
                "status": "timeout",
                "history": history,
                "message": f"Timeout after {elapsed:.1f}s, solved {quiz_count}/24 quizzes"
            }

        quiz_count += 1
        
        # Progress indicator every 5 quizzes
        if quiz_count % 5 == 0:
            print(f"\n⚡ Progress: {quiz_count}/24 quizzes, {elapsed:.1f}s elapsed")
        
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            remaining=remaining,
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

        if result.get("correct"):
            return {
                "status": "completed",
                "history": history,
                "message": f"✅ Successfully solved all {quiz_count} quizzes in {elapsed:.1f}s!"
            }
        else:
            return {
                "status": "failed",
                "history": history,
                "message": f"❌ Failed on quiz {quiz_count}/24: {result.get('error') or result.get('reason')}"
            }
