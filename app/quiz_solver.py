"""
quiz_solver.py
Silent production implementation.
Fully aligned with the defined architecture and execution logic.
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
    text = re.sub(r"Post your answer[\s\S]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[\s\S]*?\}", "", text)
    return text.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for elem in soup.find_all(text=True):
        t = elem.strip()
        if not t:
            continue
        if t.lower().startswith("q") or "question" in t.lower():
            candidates.append(t)

    raw = "\n".join(candidates) if candidates else fallback_text
    return sanitize_question_text(raw)


def gather_page_resources(quiz_url: str, html: str, text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context = []

    for link in download_links:
        try:
            full = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full)
            dataframes.append({"url": full, "df": df})
            data_context.append(meta)
        except Exception:
            continue

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
                f"File: {item['url']}",
                f"Shape: {df.shape}",
                f"Columns: {list(df.columns)}",
                df.head().to_string(),
            ])

    if resources["other_urls"]:
        parts.append("\nOTHER URLS:")
        parts.extend(resources["other_urls"])

    return "\n".join(parts)


def normalize_answer_type(val):
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s) if "." in s else int(s)
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    remaining: float,
) -> Dict[str, Any]:

    html, text = await fetch_page_html_and_text(quiz_url)

    question = extract_visible_question(html, text)
    resources = gather_page_resources(quiz_url, html, text)

    if not resources["submit_url"]:
        return {"correct": False, "error": "Submit URL not found"}

    context = build_llm_context(question, text, resources)

    llm_result = await ask_llm_for_answer(
        question_text=question,
        context_text=context,
        data_notes="",
    )

    answer = normalize_answer_type(llm_result.get("answer"))

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    if len(json.dumps(payload).encode("utf-8")) > 1024 * 1024:
        return {"correct": False, "error": "Payload too large"}

    try:
        response = requests.post(resources["submit_url"], json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"correct": False, "error": str(e)}

    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer,
        "llm_info": llm_result,
    }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout: float = MAX_GLOBAL_SECONDS,
) -> Dict[str, Any]:

    current_url = start_url
    history = []

    while True:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed

        if remaining <= 0:
            return {"status": "timeout", "history": history}

        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            remaining=remaining,
        )

        history.append({
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
        })

        next_url = result.get("url")

        if next_url:
            current_url = next_url
            continue

        return {
            "status": "finished_correct" if result.get("correct") else "finished_incorrect",
            "history": history
        }

