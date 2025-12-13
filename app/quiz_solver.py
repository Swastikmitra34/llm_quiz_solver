"""
quiz_solver.py
Compatible with existing utils.py
Hard limit: 3 minutes per question
Max questions: 24
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
MAX_QUIZ_SECONDS = 180   # 3 minutes per question
MAX_QUIZZES = 24


# ============================================================================
# HELPERS
# ============================================================================
def sanitize_question_text(text: str) -> str:
    text = re.sub(r"Post your answer[\s\S]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\{[^}]{20,}\}", "", text)
    return text.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'div']):
        t = elem.get_text().strip()
        if len(t) < 10:
            continue
        if any(k in t.lower() for k in [
            'question', 'what', 'how', 'calculate',
            'find', 'count', 'download', 'color'
        ]):
            candidates.append(t)

    raw = "\n".join(candidates[:5]) if candidates else fallback_text
    return sanitize_question_text(raw)


def normalize_answer_type(val):
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


# ============================================================================
# RESOURCE GATHERING
# ============================================================================
async def gather_page_resources(
    quiz_url: str,
    html: str,
    text: str,
    email: str = ""
) -> Dict[str, Any]:

    soup = BeautifulSoup(html, "html.parser")

    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)
    api_headers = extract_api_headers_from_text(text)

    dataframes = []
    data_context = []
    pdf_texts = []

    for link in find_download_links_from_html(html)[:3]:
        try:
            full_url = normalize_url(quiz_url, link)

            if full_url.lower().endswith(".pdf"):
                pdf = extract_text_from_pdf(full_url, api_headers)
                pdf_texts.append(pdf[:2000])
                continue

            meta, df = download_and_load_data(full_url, api_headers)
            dataframes.append({"url": full_url, "df": df})
            data_context.append(meta)

        except Exception as e:
            data_context.append(f"Failed {link}: {e}")

    image_data = []
    for img in soup.find_all("img", src=True)[:3]:
        try:
            img_url = normalize_url(quiz_url, img["src"])
            info = process_image(img_url, api_headers)
            if "error" not in info:
                image_data.append(info)
        except:
            pass

    api_responses = []
    for api in extract_api_urls_from_text(text)[:5]:
        try:
            res = call_api(api["url"], api["method"], api_headers)
            if res.get("success"):
                api_responses.append({
                    "url": api["url"],
                    "method": api["method"],
                    "response": res.get("data") or res.get("text")
                })
        except:
            pass

    return {
        "submit_url": submit_url,
        "dataframes": dataframes,
        "data_context_text": "\n".join(data_context),
        "pdf_texts": pdf_texts,
        "image_data": image_data,
        "api_responses": api_responses,
        "api_headers": api_headers,
    }


# ============================================================================
# CONTEXT BUILDING
# ============================================================================
def build_llm_context(
    question: str,
    page_text: str,
    resources: Dict[str, Any],
    email: str = ""
) -> str:

    parts = [
        "=== QUESTION ===",
        question[:800],
        "\n=== PAGE TEXT ===",
        sanitize_question_text(page_text)[:1500],
    ]

    if email:
        parts.append(f"\nEmail: {email}")
        parts.append(f"Email length mod 2: {len(email) % 2}")

    for item in resources["dataframes"]:
        df = item["df"]
        parts.append(f"\nFile: {item['url']}")
        parts.append(f"Shape: {df.shape}")
        parts.append(df.head(8).to_string())

    if resources["pdf_texts"]:
        parts.append("\n=== PDF ===")
        parts.extend(resources["pdf_texts"])

    if resources["image_data"]:
        parts.append("\n=== IMAGES ===")
        for img in resources["image_data"]:
            parts.append(str(img))

    if resources["api_responses"]:
        parts.append("\n=== API ===")
        for api in resources["api_responses"]:
            parts.append(json.dumps(api, indent=2)[:800])

    ctx = "\n".join(parts)
    return ctx[:12000]


# ============================================================================
# SINGLE QUIZ SOLVER (3 MIN HARD LIMIT)
# ============================================================================
async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    cached_submit_url: Optional[str] = None,
) -> Dict[str, Any]:

    quiz_start = time.time()

    def check_timeout():
        if time.time() - quiz_start > MAX_QUIZ_SECONDS:
            raise TimeoutError("Quiz exceeded 3 minutes")

    try:
        html, text = await fetch_page_html_and_text(quiz_url)
        check_timeout()

        question = extract_visible_question(html, text)
        resources = await gather_page_resources(quiz_url, html, text, email)
        check_timeout()

        submit_url = resources["submit_url"] or cached_submit_url
        if not submit_url:
            return {"correct": False, "error": "Submit URL not found"}

        context = build_llm_context(question, text, resources, email)
        llm = await ask_llm_for_answer(context)
        check_timeout()

        answer = normalize_answer_type(llm.get("answer"))

        payload = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": answer,
        }

        r = requests.post(submit_url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()

        return {
            "correct": bool(data.get("correct")),
            "url": data.get("url"),
            "reason": data.get("reason"),
            "used_answer": answer,
            "submit_url": submit_url,
            "quiz_time": time.time() - quiz_start,
        }

    except TimeoutError:
        return {
            "correct": False,
            "error": "Per-question timeout (180s)",
            "quiz_time": time.time() - quiz_start,
        }

    except Exception as e:
        return {
            "correct": False,
            "error": str(e),
            "quiz_time": time.time() - quiz_start,
        }


# ============================================================================
# MAIN LOOP
# ============================================================================
async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
) -> Dict[str, Any]:

    current_url = start_url
    cached_submit_url = None
    history = []

    for quiz_number in range(1, MAX_QUIZZES + 1):

        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            cached_submit_url=cached_submit_url,
        )

        if result.get("submit_url"):
            cached_submit_url = result["submit_url"]

        history.append({
            "quiz_number": quiz_number,
            "url": current_url,
            "correct": result.get("correct"),
            "error": result.get("error"),
            "quiz_time": result.get("quiz_time"),
        })

        if not result.get("correct"):
            return {
                "status": "failed",
                "history": history,
                "message": f"Failed at quiz {quiz_number}",
            }

        if not result.get("url"):
            return {
                "status": "completed",
                "history": history,
                "message": f"Solved {quiz_number} quizzes successfully",
            }

        current_url = result["url"]

    return {
        "status": "completed",
        "history": history,
        "message": "Solved all 24 quizzes",
    }
