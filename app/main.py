from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import time

from .quiz_solver import solve_quiz

load_dotenv()

app = FastAPI()   # <-- THIS is what uvicorn is searching for

SECRET = os.getenv("SECRET", "CHANGE_ME")


class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "LLM Quiz Solver is running"}


@app.post("/quiz")
async def handle_quiz(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        payload = QuizPayload(**body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e.errors()}")

    if payload.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    start_time = time.time()

    result = await solve_quiz(
        email=payload.email,
        secret=payload.secret,
        start_url=payload.url,
        start_time=start_time,
        timeout_seconds=170,
    )

    return result
