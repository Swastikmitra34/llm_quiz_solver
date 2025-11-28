import time
import re
import requests
from typing import Dict, Any
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

# Retry behaviour inside the 3-minute window
MAX_RETRIES_PER_QUESTION = 2
MIN_SECONDS_TO_RETRY = 20


# -------------------- Helper functions -------------------- #

def extract_answer_from_template(text: str, quiz_url: str) -> str | None:
    """
    Detects explicit JSON example payloads and extracts the "answer" field.
    We **only** trust this in demo-style URLs (containing 'demo'),
    so that real quiz pages are not accidentally treated as templates.
    """
    if "demo" not in quiz_url:
        return None

    match = re.search(r'"answer"\s*:\s*("?[^"\n]+")', text)
    if match:
        raw = match.group(1)
        return raw.strip('"')
    return None


def sanitize_question_text(full_text: str) -> str:
    """
    Remove trailing instructions (like 'Post your answer ...')
    and any example JSON blocks that confuse the LLM.
    """
    # Cut off anything after 'Post your answer'
    cleaned = re.sub(
        r"Post your answer[\s\S]*",
        "",
        full_text,
        flags=re.IGNORECASE,
    )

    # Remove JSON-like blocks (example payloads)
    cleaned = re.sub(r"\{[\s\S]*?\}", "", cleaned)

    # Collapse and strip
    return cleaned.strip()


# -------------------- Single-quiz solver -------------------- #

async def solve_single_quiz_attempt(
    email: str,
    secret: str,
    quiz_url: str,
) -> Dict[str, Any]:
    """
    Solve exactly ONE quiz URL once (no retries inside this function).
    """

    # 1. Fetch rendered page (JS executed)
    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")

    # 2. Find submit URL (must not be hardcoded)
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    # ---------------- TEMPLATE MODE (demo only) ---------------- #
    template_answer = extract_answer_from_template(text, quiz_url)
    if template_answer and submit_url:
        payload = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": template_answer,
        }

        try:
            resp = requests.post(submit_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {
                "correct": False,
                "error": f"Submission failed (template mode): {e}",
                "used_answer": template_answer,
                "llm_info": {"mode": "template_followed"},
            }

        return {
            "correct": bool(data.get("correct", False)),
            "next_url": data.get("url"),
            "reason": data.get("reason"),
            "used_answer": template_answer,
            "llm_info": {"mode": "template_followed"},
        }

    # ---------------- QUESTION EXTRACTION ---------------- #

    # Prefer lines that look like questions (start with Q... or contain 'question')
    possible = []
    for elem in soup.find_all(text=True):
        t = elem.strip()
        if t and (t.lower().startswith("q") or "question" in t.lower()):
            possible.append(t)

    raw_question_text = "\n".join(possible) if possible else text.strip()
    question_text = sanitize_question_text(raw_question_text)

    # ---------------- DATA FILE HANDLING ---------------- #

    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context_parts = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            data_context_parts.append(meta)
            dataframes.append({"url": full_link, "df": df})
        except Exception:
            # Ignore a bad file, keep going
            pass

    data_context_text = "\n\n".join(data_context_parts)

    # ---------------- QUESTION TYPE ---------------- #

    question_type = classify_question_type(question_text)
    answer_value = None
    llm_info: Dict[str, Any] = {}

    # ---------------- NUMERIC / DATA-FIRST LOGIC ---------------- #

    if question_type == "numeric" and dataframes:
        col_name = extract_column_sum_from_question(question_text)
        if col_name:
            for item in dataframes:
                df = item["df"]
                col_map = {c.lower(): c for c in df.columns}
                if col_name.lower() in col_map:
                    try:
                        s = df[col_map[col_name.lower()]].sum()
                        answer_value = float(s) if hasattr(s, "item") else s
                        llm_info = {
                            "mode": "numeric_auto",
                            "column": col_map[col_name.lower()],
                            "source_url": item["url"],
                        }
                        break
                    except Exception:
                        # Try next dataframe
                        continue

    # ---------------- SIMPLE SCRAPING LOGIC ---------------- #
    # Very lightweight: detect patterns like "secret code: XYZ123"
    if answer_value is None:
        m = re.search(
            r"(secret|code|token|answer)[^\w]*[:\-]\s*([A-Za-z0-9_-]{3,})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            answer_value = m.group(2)
            llm_info = {"mode": "regex_extract", "label": m.group(1).lower()}

    # ---------------- LLM FALLBACK (TEXT / GENERIC) ---------------- #

    # Only call the LLM if:
    # - numeric logic failed, OR
    # - classify_question_type said it's not clearly numeric
    if answer_value is None:
        llm_result = await ask_llm_for_answer(
            question_text=question_text,      # cleaned question only
            context_text=data_context_text,   # structured data previews (no noisy HTML)
            data_notes="",
        )

        candidate = llm_result.get("answer")

        # Basic sanity: ignore trivial garbage like "is", "a", etc.
        if isinstance(candidate, str) and len(candidate.strip()) < 2:
            answer_value = None
        else:
            answer_value = candidate

        llm_info = {"mode": "llm_reasoned", "llm_raw": llm_result}

    # ---------------- FINAL NUMERIC SAFETY NET ---------------- #

    if answer_value is None:
        detected = re.search(r"-?\d+(\.\d+)?", question_text)
        answer_value = float(detected.group()) if detected else "unknown"

    # ---------------- SUBMIT ANSWER ---------------- #

    if submit_url is None:
        return {
            "correct": False,
            "error": "Submit URL not found",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer_value,
    }

    try:
        resp = requests.post(submit_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "correct": False,
            "error": f"Submission failed: {e}",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    return {
        "correct": bool(data.get("correct", False)),
        "next_url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer_value,
        "llm_info": llm_info,
    }


# -------------------- Multi-quiz driver -------------------- #

async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = 170.0,
) -> Dict[str, Any]:
    """
    Main driver:
    - Follows chained quiz URLs.
    - Retries each question a limited number of times.
    - Respects total time budget (~3 minutes).
    """

    history = []
    current_url = start_url

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 0:
            return {"status": "timeout", "history": history}

        retry_count = 0

        # Retry loop for the same quiz URL
        while retry_count <= MAX_RETRIES_PER_QUESTION:
            question_start = time.time()

            result = await solve_single_quiz_attempt(
                email=email,
                secret=secret,
                quiz_url=current_url,
            )

            history.append({
                "url": current_url,
                "correct": result.get("correct"),
                "answer": result.get("used_answer"),
                "reason": result.get("reason"),
                "error": result.get("error"),
                "llm_info": result.get("llm_info"),
            })

            # If correct:
            if result.get("correct"):
                # If there is a next URL → move on
                if result.get("next_url"):
                    current_url = result["next_url"]
                    break  # break retry loop, go to next quiz
                # No next URL → quiz chain finished
                return {"status": "finished_correct", "history": history}

            # If incorrect:
            time_spent = time.time() - question_start
            remaining_after = remaining - time_spent

            # Not enough time left to retry safely
            if remaining_after < MIN_SECONDS_TO_RETRY:
                break

            retry_count += 1

        # After retries for this question:
        if result.get("next_url"):
            # Even if wrong, if server gives next URL, follow it
            current_url = result["next_url"]
            continue

        # No next URL and not correct → finish as incorrect
        return {"status": "finished_incorrect", "history": history}



