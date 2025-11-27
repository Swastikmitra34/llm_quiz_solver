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
