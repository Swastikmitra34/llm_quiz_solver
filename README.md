# LLM Quiz Solver (Tools in Data Science – Project)

FastAPI service that:
- Receives a quiz URL
- Loads it in a headless browser
- Uses LLM + pandas to solve data questions
- Submits answers to the given submit endpoint
- Handles chained quizzes within 3 minutes

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
                    ┌────────────────────────────┐
                    │  POST /quiz received      │
                    │  email, secret, url       │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │ Validate JSON + Secret     │
                    │ 400 / 403 if invalid       │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │ Load Quiz Page             │
                    │ Playwright (JS-rendered)   │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────┐
        │ PARSE QUESTION + TEXT CONTENT                 │
        └──────────────┬─────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────┐
        │ 1. Detect Instruction URLs                     │
        │    - Regex scan for http(s) links              │
        │    - Visit those links                         │
        │    - Extract secret/token/code patterns        │
        │    - If found → answer locked                  │
        └──────────────┬─────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────┐
        │ 2. Detect Data Files                           │
        │    - CSV / Excel / JSON                        │
        │    - Load via pandas                           │
        │    - Apply numeric logic (sum, total etc.)     │
        │    - If computed → answer locked               │
        └──────────────┬─────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────┐
        │ 3. Deterministic Text Extraction               │
        │    - Direct answer present in HTML?            │
        │    - If yes → parse and lock                   │
        └──────────────┬─────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────┐
        │ 4. LLM FALLBACK (Last Resort)                  │
        │    - Only if no hard logic succeeded           │
        │    - Interprets complex natural language       │
        └──────────────┬─────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────┐
        │ Submit Answer                                  │
        │ POST JSON to submit URL                        │
        └──────────────┬─────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────┐
        │ Check response                                 │
        │ - If new URL → loop back to Load Quiz          │
        │ - If none → finish quiz                        │
        └────────────────────────────────────────────────┘
