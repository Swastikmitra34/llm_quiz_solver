import time
import re
from typing import Dict, Any
import requests
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


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    time_left_seconds: float,
) -> Dict[str, Any]:

    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")
    question_text = text.strip()

    answer_value = None
    llm_info = {}

    # ======================================================
    # 1. AUTO FOLLOW EMBEDDED TASK URLS (JS rendered)
    # ======================================================

    embedded_urls = re.findall(r"https?://[^\s\"'>]+", text)

    for url in embedded_urls:
        if quiz_url in url:
            continue
        try:
            sub_html, sub_text = await fetch_page_html_and_text(url)
            sub_text = sub_text.strip()

            token_match = re.search(
                r"(secret|code|token|answer)[^\w]*[:\-]?\s*([A-Za-z0-9_-]{3,})",
                sub_text,
                re.IGNORECASE
            )

            if token_match:
                answer_value = token_match.group(2)
                llm_info = {"mode": "scrape_auto", "source": url}
                break
        except:
            continue

    # ======================================================
    # 2. DATA FILE HANDLING (CSV, XLS, JSON etc)
    # ======================================================

    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context_parts = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            dataframes.append({"df": df})
            data_context_parts.append(meta)
        except:
            pass

    data_context_text = "\n\n".join(data_context_parts)

    if answer_value is None and dataframes:
        question_type = classify_question_type(question_text)
        if question_type == "numeric":
            col = extract_column_sum_from_question(question_text)
            if col:
                for item in dataframes:
                    df = item["df"]
                    match_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                    if match_col:
                        answer_value = float(df[match_col].sum())
                        llm_info = {"mode": "numeric_auto"}
                        break

    # ======================================================
    # 3. STRICT LLM FALLBACK ONLY WHEN ALL ELSE FAILS
    # ======================================================

    if answer_value is None:
        llm_result = await ask_llm_for_answer(
            question_text=question_text,
            context_text=text,
            data_notes=data_context_text,
        )

        if llm_result.get("answer") not in [None, "", "unknown"]:
            answer_value = llm_result["answer"]
            llm_info = {"mode": "llm_fallback", "llm_raw": llm_result}

    # ======================================================
    # 4. FINAL DEFENSIVE GUARD
    # ======================================================

    if answer_value is None:
        num = re.search(r"-?\d+(\.\d+)?", text)
        answer_value = float(num.group()) if num else "unknown"

    # ======================================================
    # 5. SUBMIT ANSWER
    # ======================================================

    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    if not submit_url:
        return {
            "correct": False,
            "error": "Submit URL not detected",
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
            "error": f"Submission failed: {str(e)}",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    return {
        "correct": bool(data.get("correct")),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "used_answer": answer_value,
        "llm_info": llm_info,
    }


async def solve_quiz(
    email: str,
    secret: str,
    start_url: str,
    start_time: float,
    timeout_seconds: float = 170.0,
) -> Dict[str, Any]:

    current_url = start_url
    history = []

    while True:
        if time.time() - start_time > timeout_seconds:
            return {"status": "timeout", "history": history}

        result = await solve_single_quiz(
            email=email,
            secret=secret,
            quiz_url=current_url,
            time_left_seconds=timeout_seconds,
        )

        history.append({
            "url": current_url,
            "correct": result.get("correct"),
            "used_answer": result.get("used_answer"),
            "reason": result.get("reason"),
            "llm_info": result.get("llm_info"),
        })

        if result.get("url"):
            current_url = result["url"]
            continue

        return {
            "status": "finished_correct" if result.get("correct") else "finished_incorrect",
            "history": history,
        }

