import re
import io
from typing import List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup


def find_submit_url_from_text(text: str) -> Optional[str]:
    urls = re.findall(r"https?://[^\s\"'>]+", text)
    for u in urls:
        if "submit" in u.lower():
            return u.strip()
    return None


def find_download_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(ext in href.lower() for ext in [
            ".csv", ".json", ".xlsx", ".xls", ".txt"
        ]):
            links.append(href)

    return links


def normalize_url(base_url: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href

    from urllib.parse import urljoin
    return urljoin(base_url, href)


def download_and_load_data(url: str) -> Tuple[str, pd.DataFrame]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    data = resp.content

    if "csv" in content_type or url.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    elif "json" in content_type or url.lower().endswith(".json"):
        df = pd.read_json(io.BytesIO(data))
    elif "excel" in content_type or url.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception as e:
            raise ValueError(f"Unsupported data format at {url}: {e}")

    preview = df.head(10).to_markdown(index=False)
    meta = (
        f"URL: {url}\n"
        f"Shape: {df.shape}\n"
        f"Columns: {list(df.columns)}\n"
        f"Preview:\n{preview}"
    )

    return meta, df


def extract_column_sum_from_question(question: str) -> Optional[str]:
    """
    Detect column name for sum queries.
    Examples:
    - 'sum of the value column'
    - 'sum of the "price" column'
    - 'what is the total of sales column'
    """

    q = question.lower()

    patterns = [
        r"sum of the ['\"]?([\w\s]+?)['\"]?\s*column",
        r"total of the ['\"]?([\w\s]+?)['\"]?\s*column",
        r"sum of ['\"]?([\w\s]+?)['\"]?\s*column",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match.group(1).strip()

    return None


def classify_question_type(question: str) -> str:
    q = question.lower()

    numeric_keywords = [
        "sum", "total", "count", "average",
        "mean", "median", "percentage",
        "difference", "ratio", "how many"
    ]

    for word in numeric_keywords:
        if word in q:
            return "numeric"

    return "llm"



