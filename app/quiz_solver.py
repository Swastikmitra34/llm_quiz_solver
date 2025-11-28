import time
from typing import Dict, Any, List
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

    # ---------- Extract question ----------
    possible = []
    for elem in soup.find_all(text=True):
        t = elem.strip()
        if t and (t.lower().startswith("q") or "question" in t.lower()):
            possible.append(t)

    question_text = "\n".join(possible) if possible else text.strip()

    print("\n================ QUIZ DEBUG ================")
    print("QUIZ URL:", quiz_url)
    print("EXTRACTED QUESTION:\n", question_text)

    # ---------- Load downloadable data ----------
    download_links = find_download_links_from_html(html)
    dataframes = []
    data_context_parts = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            data_context_parts.append(meta)
            dataframes.append({"url": full_link, "df": df})
        except Exception as e:
            data_context_parts.append(f"Failed to load {link}: {str(e)}")

    data_context_text = "\n\n".join(data_context_parts)

    print("\n---- DATA CONTEXT ----")
    print(data_context_text if data_context_text else "No external data detected.")

    # ---------- Determine question type ----------
    question_type = classify_question_type(question_text)
    print("\nDetected question type:", question_type)

    answer_value = None
    llm_info = {}

    # ---------- Strategy Selector ----------
    if question_type == "numeric" and dataframes:
        col_name = extract_column_sum_from_question(question_text)
        if col_name:
            for item in dataframes:
                df = item["df"]
                col_map = {c.lower(): c for c in df.columns}
                if col_name.lower() in col_map:
                    try:
                        real_col = col_map[col_name.lower()]
                        s = df[real_col].sum()
                        answer_value = float(s) if hasattr(s, "item") else s
                        llm_info = {"mode": "numeric_auto"}
                        print("Numeric auto-solved answer:", answer_value)
                        break
                    except Exception as e:
                        print("Numeric auto-failed:", e)

    # ---------- LLM Fallback ----------
    if answer_value is None:
        print("\n---- PROMPT SENT TO LLM ----")
        print("QUESTION:\n", question_text)
        print("DATA NOTES:\n", data_context_text)
        
        llm_result = await ask_llm_for_answer(
            question_text=question_text,
            context_text=text,
            data_notes=data_context_text,
        )

        print("\n---- RAW LLM RESPONSE ----")
        print(llm_result)

        answer_value = llm_result.get("answer")
        llm_info = {"mode": "llm_reasoned", "llm_raw": llm_result}

    # ---------- Absolute Safety Net ----------
    if answer_value is None or str(answer_value).strip().lower() in ["unknown", "none", ""]:
        detected = re.search(r"-?\d+(\.\d+)?", question_text)
        if detected:
            answer_value = float(detected.group())
            print("Fallback numeric extraction used:", answer_value)
        else:
            answer_value = 0
            print("Hard fallback used: 0")

    print("\n>>> FINAL ANSWER TO SUBMIT:", answer_value)

    # ---------- Find submit URL ----------
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    if submit_url is None:
        return {
            "correct": False,
            "error": "Submit URL not found",
            "url": None,
            "submit_url": None,
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
            "url": None,
            "submit_url": submit_url,
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    return {
        "correct": bool(data.get("correct", False)),
        "url": data.get("url"),
        "reason": data.get("reason"),
        "raw_response": data,
        "used_answer": answer_value,
        "submit_url": submit_url,
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

        if not result.get("correct"):
            if result.get("url"):
                current_url = result["url"]
                continue
            return {"status": "finished_incorrect", "history": history}

        if result.get("url"):
            current_url = result["url"]
            continue

        return {"status": "finished_correct", "history": history}


