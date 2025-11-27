import time
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
    extract_column_sum_from_question,
)


async def solve_single_quiz(
    email: str,
    secret: str,
    quiz_url: str,
    time_left_seconds: float,
) -> Dict[str, Any]:

    html, text = await fetch_page_html_and_text(quiz_url)
    soup = BeautifulSoup(html, "html.parser")

    # 1. Extract question text
    possible: List[str] = []
    for elem in soup.find_all(text=True):
        t = elem.strip()
        if t and (t.startswith("Q") or "Question" in t):
            possible.append(t)

    question_text = "\n".join(possible) if possible else text

    # 2. Load downloadable data
    download_links = find_download_links_from_html(html)
    dataframes: List[Dict[str, Any]] = []
    data_context_parts: List[str] = []

    for link in download_links:
        try:
            full_link = normalize_url(quiz_url, link)
            meta, df = download_and_load_data(full_link)
            data_context_parts.append(meta)
            dataframes.append({"url": full_link, "df": df})
        except Exception as e:
            data_context_parts.append(f"Failed to load data from {link}: {e}")

    data_context_text = "\n\n".join(data_context_parts)

    # 3. HARD LOGIC FIRST : solve sum questions without LLM
    numeric_answer = None

    if dataframes and "sum" in question_text.lower():
        col_name = extract_column_sum_from_question(question_text)
        if col_name:
            for item in dataframes:
                df = item["df"]
                df_cols_lower = {c.lower(): c for c in df.columns}
                if col_name.lower() in df_cols_lower:
                    real_col = df_cols_lower[col_name.lower()]
                    try:
                        s = df[real_col].sum()
                        numeric_answer = float(s) if hasattr(s, "item") else s
                        break
                    except Exception:
                        pass

    # 4. Choose answer source
    if numeric_answer is not None:
        answer_value = numeric_answer
        llm_info = {"used_llm": False}

    else:
        llm_result = await ask_llm_for_answer(
            question_text=question_text,
            context_text=text,
            data_notes=data_context_text,
        )

        answer_value = llm_result.get("answer")

        # SAFETY FALLBACK: never respond with None or "unknown"
        if answer_value is None:
            import re
            combined = question_text + "\n" + text + "\n" + data_context_text
            m = re.search(r"-?\d+(\.\d+)?", combined)
            if m:
                answer_value = float(m.group())
            else:
                answer_value = 0

        llm_info = {"used_llm": True, "llm_raw": llm_result}

    # 5. Find submit URL
    submit_url = find_submit_url_from_text(text) or find_submit_url_from_text(html)

    if submit_url is None:
        return {
            "correct": False,
            "error": "Could not find submit URL",
            "url": None,
            "submit_url": None,
            "used_answer": answer_value,
            "llm_info": llm_info,
        }

    # 6. Submit answer
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
            "error": f"Submit failed: {e}",
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
    history: List[Dict[str, Any]] = []

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
