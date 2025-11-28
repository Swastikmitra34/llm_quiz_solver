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

MAX_RETRIES_PER_QUESTION = 2
MIN_SECONDS_TO_RETRY = 20


def extract_answer_from_template(text: str):
    match = re.search(r'"answer"\s*:\s*("?[^"\n]+")', text)
    if match:
        raw = match.group(1)
        return raw.strip('"')
    return None


async def solve_single_quiz_attempt(
    email: str,
    secret: str,
    quiz_url: str,
) -> Dict[str, Any]:

    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")

    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    # ---------- TEMPLATE MODE ----------
    template_answer = extract_answer_from_template(text)
    if template_answer and submit_url:
        payload = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": template_answer,
        }
        resp = requests.post(submit_url, json=payload, timeout=30)
        data = resp.json()

        return {
            "correct": bool(data.get("correct", False)),
            "next_url": data.get("url"),
            "reason": data.get("reason"),
            "used_answer": template_answer,
            "llm_info": {"mode": "template_followed"},
        }

    # ---------- GENERIC SCRAPE LINK MODE ----------
    instruction_text = text.lower()

    if re.search(r"(scrape|get|extract|fetch).*page", instruction_text):
        links = [a.get("href") for a in soup.find_all("a") if a.get("href")]

        for link in links:
            try:
                full_link = normalize_url(quiz_url, link)
                scrape_html, scrape_text = await fetch_page_html_and_text(full_link)

                # Look for 3–8 digit code (common pattern)
                code_match = re.search(r"\b(\d{3,8})\b", scrape_text)

                if code_match and submit_url:
                    answer_value = code_match.group(1)

                    payload = {
                        "email": email,
                        "secret": secret,
                        "url": quiz_url,
                        "answer": answer_value,
                    }

                    resp = requests.post(submit_url, json=payload, timeout=30)
                    data = resp.json()

                    return {
                        "correct": bool(data.get("correct", False)),
                        "next_url": data.get("url"),
                        "reason": data.get("reason"),
                        "used_answer": answer_value,
                        "llm_info": {"mode": "generic_scrape_link"},
                    }
            except Exception:
                continue

    # ---------- QUESTION EXTRACTION ----------
    possible = []
    for elem in soup.find_all(text=True):
        t = elem.strip()
        if t and (t.lower().startswith("q") or "question" in t.lower()):
            possible.append(t)

    question_text = "\n".join(possible) if possible else text.strip()

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
    question_type = classify_question_type(question_text)

    answer_value = None
    llm_info = {}

    # ---------- NUMERIC CSV / DATA HANDLING ----------
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
                        llm_info = {"mode": "numeric_auto"}
                        break
                    except:
                        pass

    # ---------- LLM FALLBACK (only if necessary) ----------
    if answer_value is None:
        llm_result = await ask_llm_for_answer(
            question_text=question_text,
            context_text=text,
            data_notes=data_context_text,
        )

        candidate = llm_result.get("answer")

        # Reject garbage like "your secret"
        if isinstance(candidate, str) and "secret" in candidate.lower():
            answer_value = None
        else:
            answer_value = candidate

        llm_info = {"mode": "llm_reasoned", "llm_raw": llm_result}

    if answer_value is None:
        detected = re.search(r"-?\d+(\.\d+)?", question_text)
        answer_value = float(detected.group()) if detected else "unknown"

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

    resp = requests.post(submit_url, json=payload, timeout=30)
    data = resp.json()

    return {
        "correct": bool(data.get("correct", False)),
        "next_url": data.get("url"),
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

    history = []
    current_url = start_url

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 0:
            return {"status": "timeout", "history": history}

        retry_count = 0

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
                "llm_info": result.get("llm_info"),
            })

            if result.get("correct"):
                if result.get("next_url"):
                    current_url = result["next_url"]
                    break
                return {"status": "finished_correct", "history": history}

            if (remaining - (time.time() - question_start)) < MIN_SECONDS_TO_RETRY:
                break

            retry_count += 1

        if result.get("next_url"):
            current_url = result["next_url"]
            continue

        return {"status": "finished_incorrect", "history": history}

