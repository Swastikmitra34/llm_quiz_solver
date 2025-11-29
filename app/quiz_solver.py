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


# We keep global timeout management in solve_quiz, not here
MAX_GLOBAL_SECONDS = 170  # main.py already uses this


def sanitize_question_text(full_text: str) -> str:
    """
    Remove trailing instructions (like 'Post your answer ...')
    and any example JSON payloads that could confuse the LLM.
    """
    # Cut off from "Post your answer" onwards
    cleaned = re.sub(
        r"Post your answer[\s\S]*",
        "",
        full_text,
        flags=re.IGNORECASE,
    )

    # Remove obvious JSON blobs (example payloads)
    cleaned = re.sub(r"\{[\s\S]*?\}", "", cleaned)

    return cleaned.strip()


def extract_visible_question(html: str, fallback_text: str) -> str:
    """
    Try to extract the actual question text from the rendered HTML.
    Prefer lines that start with 'Q' or contain 'question'.
    """
    soup = BeautifulSoup(html, "html.parser")
    possible: List[str] = []

    for elem in soup.find_all(text=True):
        t = elem.strip()
        if not t:
            continue
        lower = t.lower()
        if lower.startswith("q") or "question" in lower:
            possible.append(t)

    if possible:
        raw = "\n".join(possible)
    else:
        raw = fallback_text.strip()

    return sanitize_question_text(raw)


def gather_page_resources(
    quiz_url: str,
    html: str,
    text: str,
) -> Dict[str, Any]:
    """
    Scrape the page for:
    - submit URL
    - downloadable data files (CSV/JSON/Excel/TXT)
    - other URLs (potential APIs / extra pages)
    - summaries of loaded dataframes
    """
    soup = BeautifulSoup(html, "html.parser")

    # Submit URL
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    # Downloadable files
    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context_parts = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            dataframes.append({"url": full_link, "df": df})
            data_context_parts.append(meta)
        except Exception:
            # If a file fails, skip it, don't crash solver
            continue

    data_context_text = "\n\n".join(data_context_parts)

    # Other URLs on page (potential APIs / extra pages)
    all_urls = set(re.findall(r"https?://[^\s\"'<>]+", text))
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            all_urls.add(href)
        elif href.startswith("/"):
            base = "/".join(quiz_url.split("/")[:3])
            all_urls.add(base + href)
        else:
            # Relative link
            base = "/".join(quiz_url.split("/")[:-1])
            all_urls.add(base + "/" + href)

    # Remove self-url and submit-url from extra URLs
    other_urls = {u for u in all_urls if u != quiz_url and u != submit_url}

    return {
        "submit_url": submit_url,
        "dataframes": dataframes,
        "data_context_text": data_context_text,
        "other_urls": list(other_urls),
    }


def build_llm_context(
    question_text: str,
    page_text: str,
    resources: Dict[str, Any],
) -> str:
    """
    Build a big text context for the LLM:
    - Question
    - Main page text (sanitized)
    - Dataframe previews
    - Extra URLs (for reference)
    - API responses (if we choose to call them later)
    """
    parts: List[str] = []
    parts.append("QUESTION:")
    parts.append(question_text)
    parts.append("\nPAGE TEXT (sanitized):")
    parts.append(sanitize_question_text(page_text))

    if resources["dataframes"]:
        parts.append("\n\nDATA FILE SUMMARIES:")
        for item in resources["dataframes"]:
            df = item["df"]
            url = item["url"]
            parts.append(f"\nFile: {url}")
            parts.append(f"Shape: {df.shape}")
            parts.append(f"Columns: {list(df.columns)}")
            parts.append("Head preview:")
            parts.append(df.head().to_string())

    if resources["other_urls"]:
        parts.append("\n\nOTHER URLS ON PAGE:")
        for u in resources["other_urls"]:
            parts.append(f"- {u}")

    return "\n".join(parts)


def normalize_answer_type(answer_value):
    """
    Try to coerce LLM output to an appropriate JSON type:
    - If 'true'/'false' → bool
    - If numeric string → int/float
    - If JSON-looking string → parse JSON
    Otherwise leave as-is.
    """
    if isinstance(answer_value, str):
        s = answer_value.strip()

        # Boolean
        if s.lower() in ["true", "false"]:
            return s.lower() == "true"

        # Number
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            if "." in s:
                try:
                    return float(s)
                except Exception:
                    pass
            else:
                try:
                    return int(s)
                except Exception:
                    pass

        # JSON (object or array)
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                # If JSON parsing fails, just return raw string
                return answer_value

    return answer_value


# ----------------- SINGLE QUIZ ATTEMPT ----------------- #

async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    time_left_seconds: float,
) -> Dict[str, Any]:
    """
    Solve a single quiz URL once:
    - Render page
    - Gather resources
    - Ask LLM for 'answer'
    - Submit JSON payload
    - Return correctness + next_url
    """

    # 1. Render page (JS executed)
    html, text = await fetch_page_html_and_text(quiz_url)

    # 2. Extract question
    question_text = extract_visible_question(html, text)

    # 3. Gather resources (files, URLs, submit_url)
    resources = gather_page_resources(quiz_url, html, text)
    submit_url = resources["submit_url"]

    if not submit_url:
        return {
            "correct": False,
            "error": "Submit URL not detected on page",
            "used_answer": None,
            "llm_info": None,
        }

    # 4. Build context for LLM
    llm_context = build_llm_context(question_text, text, resources)

    # 5. Call LLM – single source of truth for answer
    llm_result = await ask_llm_for_answer(
        question_text=question_text,
        context_text=llm_context,
        data_notes="",
    )

    raw_answer = llm_result.get("answer")
    used_answer = normalize_answer_type(raw_answer)

    # 6. Build submission payload
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": used_answer,
    }

    # 1 MB cap
    payload_bytes = len(json.dumps(payload).encode("utf-8"))
    if payload_bytes > 1024 * 1024:
        return {
            "correct": False,
            "error": f"Payload too large: {payload_bytes} bytes",
            "used_answer": used_answer,
            "llm_info": {"mode": "llm", "raw": llm_result},
        }

    # 7. Submit to quiz's submit URL
    try:
        resp = requests.post(submit_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "correct": False,
            "error": f"Submission failed: {e}",
            "used_answer": used_answer,
            "llm_info": {"mode": "llm", "raw": llm_result},
        }

    return {
        "correct": bool(data.get("correct", False)),
        "url": data.get("url"),   # next quiz URL if present
        "reason": data.get("reason"),
        "used_answer": used_answer,
        "llm_info": {"mode": "llm", "raw": llm_result},
    }


# ----------------- QUIZ CHAIN DRIVER ----------------- #

async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = MAX_GLOBAL_SECONDS,
) -> Dict[str, Any]:
    """
    Drives the full quiz chain:
    - Starts at start_url
    - Follows 'url' from response until none given
    - Stops on timeout or final question
    """

    current_url = start_url
    history = []

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 0:
            return {"status": "timeout", "history": history}

        # Single attempt per URL (you’re allowed to resubmit but not required)
        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            time_left_seconds=remaining,
        )

        history.append({
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "llm_info": result.get("llm_info"),
        })

        next_url = result.get("url")

        # If there is a next URL, move to that regardless of correct/incorrect.
        if next_url:
            current_url = next_url
            continue

        # No next URL: quiz chain ends here.
        if result.get("correct"):
            return {"status": "finished_correct", "history": history}
        else:
            return {"status": "finished_incorrect", "history": history}
