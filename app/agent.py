"""
agent.py
--------
The agent orchestrator.

Flow for every /chat call:
  1. Classify intent from conversation history
  2. Build retrieval query (if needed)
  3. Retrieve top-K candidates from FAISS
  4. Build prompt with catalog context
  5. Call Groq LLM → get JSON response
  6. Validate + sanitise response
  7. Return ChatResponse
"""

import json
import os
import re
import logging
from groq import Groq
from dotenv import load_dotenv

from app.schemas import ChatRequest, ChatResponse, Recommendation
from app import retrieval
from app.prompts import SYSTEM_PROMPT, build_catalog_context, build_messages

load_dotenv()
log = logging.getLogger(__name__)

# ── Groq client ──────────────────────────────────────────────────────────────
_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Valid catalog URLs (loaded once) ─────────────────────────────────────────
_valid_urls: set[str] = set()

def _get_valid_urls() -> set[str]:
    global _valid_urls
    if not _valid_urls:
        _valid_urls = {c["url"] for c in retrieval.get_all()}
    return _valid_urls


# ── Intent classification ────────────────────────────────────────────────────
def _is_vague(messages: list[dict]) -> bool:
    """
    Returns True ONLY if the query has zero role/skill signal
    AND it's the first user turn.
    """
    user_turns = [m for m in messages if m["role"] == "user"]
    
    # After first turn, never consider vague — just recommend
    if len(user_turns) > 1:
        return False
    
    if not user_turns:
        return True
    
    last = user_turns[-1]["content"].lower()
    
    # If message is longer than 10 words, it has enough context
    if len(last.split()) > 10:
        return False
    
    # If it mentions any role, skill, or level signal — not vague
    role_signals = [
        "developer", "engineer", "manager", "analyst", "designer",
        "sales", "marketing", "finance", "hr", "data", "java", "python",
        "senior", "junior", "mid", "entry", "graduate", "executive",
        "director", "cxo", "leadership", "customer", "contact center",
        "healthcare", "manufacturing", "hiring", "recruit", "role",
        "personality", "cognitive", "ability", "knowledge", "skills",
    ]
    if any(signal in last for signal in role_signals):
        return False
    
    # Truly vague — no role signal, short message
    return True

def _build_retrieval_query(messages: list[dict]) -> str:
    """
    Distil the entire conversation history into a single search query.
    We concatenate all user messages — this preserves refinements.
    """
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_parts)


def _is_compare_query(messages: list[dict]) -> bool:
    """Detect comparison requests like 'difference between X and Y'."""
    if not messages:
        return False
    last = messages[-1]["content"].lower() if messages[-1]["role"] == "user" else ""
    compare_signals = ["difference between", "compare", "vs ", "versus", "which is better", "what's the difference"]
    return any(s in last for s in compare_signals)


def _is_out_of_scope(messages: list[dict]) -> bool:
    """Detect off-topic requests."""
    if not messages:
        return False
    last = messages[-1]["content"].lower() if messages[-1]["role"] == "user" else ""
    oos_signals = [
        "salary", "legal", "lawsuit", "discriminat", "gdpr", "ada compliance",
        "ignore previous", "ignore your instructions", "pretend you are",
        "forget your", "you are now", "jailbreak", "act as",
        "general hiring advice", "how to interview",
    ]
    return any(s in last for s in oos_signals)


# ── Core agent function ──────────────────────────────────────────────────────
def run_agent(request: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in request.messages]

    # ── Guard: out of scope ──────────────────────────────────────────────────
    if _is_out_of_scope(messages):
        return ChatResponse(
            reply="I can only help with SHL assessment selection. What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Count turns to enforce turn budget ───────────────────────────────────
    user_turns = [m for m in messages if m["role"] == "user"]
    turn_number = len(user_turns)

    # ── Build retrieval query ─────────────────────────────────────────────────
    query = _build_retrieval_query(messages)

    # ── Retrieve candidates ───────────────────────────────────────────────────
    is_compare = _is_compare_query(messages)
    candidates = retrieval.search(query, k=20 if is_compare else 15)

    # ── Build prompt ──────────────────────────────────────────────────────────
    catalog_context = build_catalog_context(candidates)
    system = SYSTEM_PROMPT.format(catalog_context=catalog_context)

    # ── Inject forcing instruction based on context ───────────────────────────
    # If we have enough context OR past turn 2, force a recommendation
    should_recommend = (not _is_vague(messages)) or (turn_number >= 3)

    if should_recommend:
        # Add a hidden system nudge as the last message before LLM call
        forcing_note = (
            "SYSTEM NOTE: You have enough context to recommend now. "
            "You MUST return a recommendations list with 1-10 assessments from the catalog. "
            "Do NOT ask any more clarifying questions. Recommend now."
        )
        llm_messages = build_messages(system, messages)
        llm_messages.append({"role": "user", "content": forcing_note})
    else:
        llm_messages = build_messages(system, messages)

    # ── Call Groq ─────────────────────────────────────────────────────────────
    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=llm_messages,
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    return _parse_response(raw_content)


def _parse_response(raw: str) -> ChatResponse:
    """
    Parse and validate the LLM's JSON output.
    Sanitises URLs and ensures schema compliance.
    """
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {raw[:500]}")
        return ChatResponse(
            reply="I encountered an issue formatting my response. Please try rephrasing.",
            recommendations=[],
            end_of_conversation=False,
        )

    reply = data.get("reply", "")
    end   = bool(data.get("end_of_conversation", False))
    raw_recs = data.get("recommendations", [])

    # Validate and sanitise recommendations
    valid_urls = _get_valid_urls()
    recommendations = []

    if isinstance(raw_recs, list):
        for r in raw_recs[:10]:   # hard cap at 10
            if not isinstance(r, dict):
                continue
            url  = r.get("url", "")
            name = r.get("name", "")
            test_type = r.get("test_type", "K")

            # CRITICAL: only allow URLs from the real catalog
            if url not in valid_urls:
                log.warning(f"Rejected hallucinated URL: {url}")
                # Try to fix: find the real URL by name
                match = retrieval.get_by_name(name)
                if match:
                    url = match["url"]
                    test_type = match.get("test_type", "K")
                else:
                    continue   # drop it entirely

            recommendations.append(Recommendation(
                name=name,
                url=url,
                test_type=test_type,
            ))

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end,
    )