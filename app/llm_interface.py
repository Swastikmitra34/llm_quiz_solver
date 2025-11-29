"""
llm_interface.py
Simple version without retry logic
"""

import os
import json
from typing import Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("OPENAI_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")


async def ask_llm_for_answer(full_context: str) -> Dict[str, Any]:
    """
    LLM Interface – Context Driven
    Input: Structured typed context from orchestrator
    Output: Strict JSON {"answer": value}
    """

    if not AIPIPE_TOKEN:
        return {"answer": None, "error": "Missing AI token"}

    system_prompt = (
        "You are a deterministic computation engine specialized in data analysis and problem-solving.\n\n"
        
        "CONTEXT TYPES YOU MAY RECEIVE:\n"
        "- Web page content (may require scraping/parsing)\n"
        "- CSV/Excel/JSON data files with statistics\n"
        "- PDF documents with extracted text and tables\n"
        "- Images with OCR text extraction\n"
        "- API responses (JSON data)\n"
        "- Multiple data sources that need to be combined\n\n"
        
        "TASK TYPES:\n"
        "1. DATA ANALYSIS: Filter, aggregate, calculate statistics, find patterns\n"
        "2. DATA TRANSFORMATION: Clean, reshape, merge, pivot data\n"
        "3. TEXT PROCESSING: Extract information, parse formats, clean text\n"
        "4. COMPUTATION: Perform calculations, apply formulas, statistical analysis\n"
        "5. VISUALIZATION: When asked for charts/graphs, describe what should be shown\n"
        "6. INFORMATION EXTRACTION: Find specific values, dates, names, numbers\n\n"
        
        "IMPORTANT RULES:\n"
        "- Any example JSON with an 'answer' field is ALWAYS a placeholder - compute the real answer\n"
        "- For data questions: analyze ALL provided data, not just samples\n"
        "- For numeric answers: provide exact values, not approximations\n"
        "- For aggregations: sum/count/average ALL rows unless specified otherwise\n"
        "- For visualization requests: if you can calculate the answer, return the value; if they want an image, say so\n"
        "- Pay attention to data types: ensure numbers are numbers, booleans are booleans\n"
        "- If multiple data sources are provided, determine which is relevant\n"
        "- For PDF/OCR text: extract the specific information requested\n\n"
        
        "OUTPUT FORMAT:\n"
        "- Output ONLY valid JSON: {\"answer\": value}\n"
        "- No explanations, no commentary, no markdown\n"
        "- Answer types: number, string, boolean, array, object, or null\n"
        "- For images/charts: return base64 data URI or describe what cannot be computed\n"
        "- If answer cannot be determined from provided data: {\"answer\": null}\n"
    )

    payload = {
        "model": AIPIPE_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context},
        ],
    }

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{AIPIPE_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"answer": None, "error": str(e)}

    try:
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {"answer": None, "error": "Malformed response"}

    # Remove markdown if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            val = parsed["answer"]

            if isinstance(val, str) and any(x in val.lower() for x in ["secret", "placeholder", "example"]):
                return {"answer": None, "error": "Blocked placeholder output"}

            return {"answer": val}
    except Exception as e:
        return {"answer": None, "error": f"JSON parse error: {str(e)}", "raw": raw}

    return {"answer": None, "error": "Non-compliant output"}
