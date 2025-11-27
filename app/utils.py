import re
import io
from typing import List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup


def find_submit_url_from_text(text: str) -> Optional[str]:
    """
    Heuristic: find any http/https URL containing 'submit'.
    Works on both HTML and plain text.
    """
    urls = re.findall(r"https?://[^\s\"'>]+", text)
    for u in urls:
        if "submit" in u:
            return u.strip()
    return None


def find_download_links_from_html(html: str) -> List[str]:
    """
    Find potential data download links (.csv, .json, .xlsx, .xls, .txt)
    from anchor tags.
    """
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(ext in href.lower() for ext in [".csv", ".json", ".xlsx", ".xls", ".txt"]):
            links.append(href)
    return links


def normalize_url(base_url: str, href: str) -> str:
    """
    Resolve relative href against base_url in a simple way.
    """
    if href.startswith("http://") or href.startswith("https://"):
        return href

    # If href is relative, attach to base
    from urllib.parse import urljoin
    return urljoin(base_url, href)


def download_and_load_data(url: str) -> Tuple[str, pd.DataFrame]:
    """
    Download a file and load it into pandas.
    Returns (meta_description, df).
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    data = resp.content

    df: pd.DataFrame

    # Heuristics for format
    if "csv" in content_type or url.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    elif "json" in content_type or url.lower().endswith(".json"):
        df = pd.read_json(io.BytesIO(data))
    elif "excel" in content_type or url.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        # Last resort: try CSV
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception as e:
            raise ValueError(f"Unsupported data format at {url}: {e}")

    # Build a compact description to send to the LLM if needed
    preview = df.head(10).to_markdown(index=False)
    meta = f"URL: {url}\nShape: {df.shape}\nColumns: {list(df.columns)}\nPreview:\n{preview}"
    return meta, df


def extract_column_sum_from_question(question: str) -> Optional[str]:
    """
    Very simple heuristic: try to detect 'sum of the "value" column'
    or similar phrases in the question and return the column name.
    """
    q = question.lower()

    # Try to match patterns like: sum of the "value" column
    m = re.search(r'sum of the [\"“](.+?)[\"”] column', question, re.IGNORECASE)
    if m:
        return m.group(1)

    # Fallback: look for 'sum of the X column' without quotes
    m2 = re.search(r"sum of the ([a-zA-Z0-9_]+) column", q)
    if m2:
        return m2.group(1)

    return None
