"""
main.py
-------
FastAPI app. Two endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from app.schemas import ChatRequest, ChatResponse
from app.agent import run_agent
from app import retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading retrieval index...")
    retrieval.load()
    log.info("Server ready.")
    yield
    log.info("Server shutting down.")


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    if len(request.messages) > 8:
        raise HTTPException(status_code=400, detail="conversation exceeds 8 turn limit")
    try:
        return run_agent(request)
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        return ChatResponse(
            reply="Something went wrong. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
@app.get("/debug")
def debug():
    import os
    return {
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
    }