import time
import re
import json
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from .browser import fetch_page_html_and_text
from .llm_interface import ask_llm_for_answer
from .utils import (
    find_download_links_from_html,
    normalize_url,
    download_and_load_data,
    extract_column_sum_from_question,
)

# ===================== SUBMIT URL DETECTION =====================

URL_REGEX = re.compile(r"https?://[^\s\"'>\)\]]+", re.IGNORECASE)
SUBMIT_HINTS = ["submit", "post", "answer"]


def find_submit_url_general(html: str, visible_text: str, base_url: str) -> Optional[str]:
    candidates = []

    # Text URLs
    candidates.extend(URL_REGEX.findall(visible_text))

    # Anchor links
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            candidates.append(urljoin(base_url, href))

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None

    context = (html + visible_text).lower()

    def score(url):
        path = urlparse(url).path.lower()
        return sum(k in path for k in SUBMIT_HINTS) + sum(k in context for k in SUBMIT_HINTS)

    return sorted(candidates, key=score, reverse=True)[0]


# ===================== MAIN SOLVER =====================

async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: int = 170,
) -> Dict[str, Any]:

    visited_urls: List[str] = []
    results: List[Dict[str, Any]] = []
    current_url = start_url
    max_attempts_per_url = 2

    while current_url:
        if time.time() - start_time > timeout_seconds:
            return {
                "status": "timeout",
                "visited_urls": visited_urls,
                "results": results,
                "elapsed_seconds": time.time() - start_time,
            }

        if current_url in visited_urls:
            break

        visited_urls.append(current_url)

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

        if quiz_result.get("status") == "correct" and next_url:
            current_url = next_url
        else:
            break

    return {
        "status": "completed",
        "visited_urls": visited_urls,
        "results": results,
        "elapsed_seconds": time.time() - start_time,
    }


# ===================== SINGLE QUIZ LOGIC =====================

async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    start_time: float,
    timeout_seconds: int,
    max_attempts: int = 2,
) -> Dict[str, Any]:

    result = {
        "url": quiz_url,
        "status": "error",
        "answer": None,
        "next_url": None,
        "attempts": 0,
        "reason": None,
    }

    html, visible_text = await fetch_page_html_and_text(quiz_url)
    if not visible_text:
        result["error"] = "No visible text"
        return result

    submit_url = find_submit_url_general(html, visible_text, quiz_url)
    if not submit_url:
        result["error"] = "Submit URL not found"
        return result

    # ------------------ DATA + SCRAPE COLLECTION ------------------

    data_context = ""

    # Downloadable files
    for link in find_download_links_from_html(html):
        full_url = normalize_url(quiz_url, link)
        try:
            meta, df = download_and_load_data(full_url)
            col = extract_column_sum_from_question(visible_text)
            if col and col in df.columns:
                meta += f"\nAUTO SUM ({col}) = {df[col].sum()}"
            data_context += meta
        except Exception as e:
            data_context += f"\nData load failed: {e}"

    # Scraped linked resources
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        txt = (a.get_text() or "").lower()
        if any(k in txt for k in ["scrape", "data", "download"]):
            resource_url = normalize_url(quiz_url, a.get("href"))
            try:
                _, scraped_text = await fetch_page_html_and_text(resource_url)
                data_context += f"\nSCRAPED FROM {resource_url}:\n{scraped_text[:1500]}"
            except:
                pass

    # ------------------ LLM ANSWER LOOP ------------------

    for attempt in range(1, max_attempts + 1):
        if time.time() - start_time > timeout_seconds:
            result["status"] = "timeout"
            return result

        result["attempts"] = attempt

        llm_response = await ask_llm_for_answer(
            question_text=visible_text,
            context_text=html[:5000],
            data_notes=data_context,
        )

        if "error" in llm_response:
            result["error"] = llm_response["error"]
            continue

        answer = llm_response.get("answer")

        if isinstance(answer, dict) and "answer" in answer:
            answer = answer["answer"]

        if isinstance(answer, str):
            ans = answer.lower()
            if any(bad in ans for bad in ["placeholder", "anything you want", "example"]):
                continue

        if answer is None:
            continue

        result["answer"] = answer

        submission = submit_answer(
            submit_url,
            email,
            secret,
            quiz_url,
            answer,
        )

        if submission.get("success"):
            result["status"] = "correct" if submission.get("correct") else "incorrect"
            result["next_url"] = submission.get("next_url")
            result["reason"] = submission.get("reason")
            return result

        result["error"] = submission.get("error")

    return result


# ===================== SUBMIT ANSWER =====================

def submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
) -> Dict[str, Any]:

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    payload_json = json.dumps(payload)
    if len(payload_json) > 1_000_000:
        return {"success": False, "error": "Payload exceeds 1MB"}

    try:
        response = requests.post(
            submit_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code != 200:
            return {"success": False, "error": response.text}

        data = response.json()
        return {
            "success": True,
            "correct": data.get("correct", False),
            "next_url": data.get("url"),
            "reason": data.get("reason"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

