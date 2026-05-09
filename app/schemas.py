"""
schemas.py
----------
Pydantic models for the /chat API.
The response schema is NON-NEGOTIABLE — deviating breaks the automated evaluator.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str   # single-letter code: A, P, K, B, S, C, D, E, M


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False