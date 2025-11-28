import time
from typing import Dict, Any
import re
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
    # 1. HARD FIX: SCRAPE DEMO TASK HANDLER
    # ======================================================

    scrape_url_match = re.search(
        r"https?://[^\s\"']*demo-scrape-data[^\s\"']*", html
    )

    if scrape_url_match:
        scrape_url = scrape_url_match.group()
        try:
            scrape_resp = requests.get(scrape_url, timeout=20)
            scrape_resp.raise_for_status()
            scrape_html = scrape_resp.text

            secret_match = re.search(
                r"Secret\s*Code\s*[:\-]?\s*([A-Za-z0-9_-]+)",
                scrape_html,
                re.IGNORECASE
            )

            if secret_match:
                answer_value = secret_match.group(1)
                llm_info = {"mode": "scrape_auto"}

        except Exception as e:
            answer_value = None

    # ======================================================
    # 2. LOAD DATA FILES IF PRESENT
    # ======================================================

    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context_parts = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            data_context_parts.append(meta)
            dataframes.append({"url": full_link, "df": df})
        except:
            pass

    data_context_text = "\n\n".join(data_context_parts)

    # ======================================================
    # 3. NUMERIC DATA HANDLER
    # ======================================================

    if answer_value is None and dataframes:
        question_type = classify_question_type(question_text)
        if question_type == "numeric":
            col_name = extract_column_sum_from_question(question_text)
            if col_name:
                for item in dataframes:
                    df = item["df"]
                    col_map = {c.lower(): c for c in df.columns}
                    if col_name.lower() in col_map:
                        real_col = col_map[col_name.lower()]
                        s = df[real_col].sum()
                        answer_value = float(s)
                        llm_info = {"mode": "numeric_auto"}
                        break

    # ======================================================
    # 4. LLM FALLBACK (ONLY WHEN NECESSARY)
    # ======================================================

    if answer_value is None:
        llm_result = await ask_llm_for_answer(
            question_text=question_text,
            context_text=text,
            data_notes=data_context_text,
        )
        answer_value = llm_result.get("answer")
        llm_info = {"mode": "llm_fallback", "llm_raw": llm_result}

    # ======================================================
    # 5. ABSOLUTE SAFETY NET
    # ======================================================

    if answer_value in [None, "", "unknown"]:
        match = re.search(r"-?\d+(\.\d+)?", question_text)
        answer_value = float(match.group()) if match else "unknown"

    # ======================================================
    # 6. SUBMIT ANSWER
    # ======================================================

    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

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
            "error": f"Submission failed: {str(e)}",
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    return {
        "correct": bool(data.get("correct", False)),
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
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 0:
            return {"status": "timeout", "history": history}

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
            "llm_info": result.get("llm_info"),
        })

        if result.get("url"):
            current_url = result["url"]
            continue

        if result.get("correct"):
            return {"status": "finished_correct", "history": history}

        return {"status": "finished_incorrect", "history": history}
