"""
quiz_solver.py
Fixed version with improved error handling and retry logic
"""

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
)

MAX_GLOBAL_SECONDS = 170


def sanitize_question_text(text: str) -> str:
    """Remove submission instructions and JSON examples from question text"""
    text = re.sub(r"Post your answer[\s\S]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[\s\S]*?\}", "", text)
    return text.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    """Extract the main question from HTML, prioritizing question-related elements"""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for elem in soup.find_all(text=True):
        t = elem.strip()
        if not t:
            continue
        # Look for question indicators
        if t.lower().startswith("q") or "question" in t.lower():
            candidates.append(t)

    raw = "\n".join(candidates) if candidates else fallback_text
    return sanitize_question_text(raw)


def gather_page_resources(quiz_url: str, html: str, text: str) -> Dict[str, Any]:
    """Extract all resources from the quiz page: submit URL, data files, links"""
    soup = BeautifulSoup(html, "html.parser")

    # Find submit URL
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    # Find and download data files
    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context = []

    for link in download_links:
        try:
            full = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full)
            dataframes.append({"url": full, "df": df})
            data_context.append(meta)
        except Exception as e:
            data_context.append(f"Failed to load {link}: {str(e)}")
            continue

    # Collect all URLs from page
    all_urls = set(re.findall(r"https?://[^\s\"'<>]+", text))
    for a in soup.find_all("a", href=True):
        all_urls.add(normalize_url(quiz_url, a["href"]))

    other_urls = {u for u in all_urls if u not in {quiz_url, submit_url}}

    return {
        "submit_url": submit_url,
        "dataframes": dataframes,
        "data_context_text": "\n\n".join(data_context),
        "other_urls": list(other_urls),
    }


def build_llm_context(question_text: str, page_text: str, resources: Dict[str, Any]) -> str:
    """Build comprehensive context for LLM including question, page content, and data"""
    parts = [
        "QUESTION:",
        question_text,
        "\nPAGE TEXT:",
        sanitize_question_text(page_text),
    ]

    if resources["dataframes"]:
        parts.append("\nDATA FILE SUMMARIES:")
        for item in resources["dataframes"]:
            df = item["df"]
            parts.extend([
                f"\nFile: {item['url']}",
                f"Shape: {df.shape}",
                f"Columns: {list(df.columns)}",
                f"\nFirst few rows:\n{df.head().to_string()}",
            ])

    if resources["data_context_text"]:
        parts.append("\nDATA CONTEXT:")
        parts.append(resources["data_context_text"])

    if resources["other_urls"]:
        parts.append("\nOTHER URLS FOUND:")
        parts.extend(resources["other_urls"][:10])  # Limit to avoid overwhelming

    return "\n".join(parts)


def normalize_answer_type(val):
    """Convert answer to appropriate type (bool, int, float, json, or string)"""
    if val is None:
        return None
        
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
            except Exception:
                return val
        
        return s
    
    return val


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    remaining: float,
) -> Dict[str, Any]:
    """Solve a single quiz question"""

    # Fetch page content
    try:
        html, text = await fetch_page_html_and_text(quiz_url)
    except Exception as e:
        return {"correct": False, "error": f"Failed to fetch page: {str(e)}"}

    # Extract question and resources
    question = extract_visible_question(html, text)
    resources = gather_page_resources(quiz_url, html, text)

    if not resources["submit_url"]:
        return {"correct": False, "error": "Submit URL not found"}

    # Build context for LLM
    context = build_llm_context(question, text, resources)

    # Get answer from LLM
    llm_result = await ask_llm_for_answer(
        question_text=question,
        context_text=context,
        data_notes="",
    )

    if "error" in llm_result and llm_result.get("answer") is None:
        return {
            "correct": False,
            "error": f"LLM error: {llm_result['error']}",
            "llm_info": llm_result
        }

    answer = normalize_answer_type(llm_result.get("answer"))

    # Build submission payload
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    # Check payload size
    if len(json.dumps(payload).encode("utf-8")) > 1024 * 1024:
        return {"correct": False, "error": "Payload too large (>1MB)"}

    # Submit answer
    try:
        response = requests.post(resources["submit_url"], json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"correct": False, "error": f"Submission failed: {str(e)}"}
    except json.JSONDecodeError as e:
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
    """Main quiz solver loop - handles multiple questions in sequence"""

    current_url = start_url
    history = []

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 10:  # Leave 10 seconds buffer
            return {
                "status": "timeout",
                "history": history,
                "message": f"Timeout after {elapsed:.1f}s"
            }

        # Solve current quiz
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            remaining=remaining,
        )

        # Record in history
        history.append({
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "elapsed": time.time() - start_time,
        })

        # Get next URL
        next_url = result.get("url")

        # If there's a next URL, continue
        if next_url:
            current_url = next_url
            continue

        # No next URL - quiz is complete
        if result.get("correct"):
            return {
                "status": "completed",
                "history": history,
                "message": "All quizzes solved successfully"
            }
        else:
            return {
                "status": "failed",
                "history": history,
                "message": f"Failed on quiz: {result.get('error') or result.get('reason')}"
            }
