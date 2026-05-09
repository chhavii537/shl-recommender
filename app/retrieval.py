"""
retrieval.py
------------
Loads the FAISS index + chunks at startup (once).
Exposes search(query, k) → list of matching assessment dicts.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "faiss.index"
CHUNKS_PATH = "chunks.json"
MODEL_NAME  = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once when the app starts
_index:  faiss.Index         = None
_chunks: list[dict]          = None
_model:  SentenceTransformer = None


def load():
    """
    Called once at FastAPI startup (lifespan event).
    Loads index, chunks, and embedding model into memory.
    """
    global _index, _chunks, _model

    print("Loading embedding model...")
    _model = SentenceTransformer(MODEL_NAME)

    print(f"Loading FAISS index from {INDEX_PATH}...")
    _index = faiss.read_index(INDEX_PATH)

    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        _chunks = json.load(f)

    print(f"Retrieval ready. {_index.ntotal} assessments indexed.")


def search(query: str, k: int = 15) -> list[dict]:
    """
    Semantic search over the catalog.

    Args:
        query: natural language query distilled from conversation
        k:     number of candidates to return (default 15, LLM picks best 1-10)

    Returns:
        list of assessment dicts (from chunks.json), best match first
    """
    if _index is None or _model is None:
        raise RuntimeError("Retrieval not initialised. Call load() first.")

    # Encode and normalise query
    q = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    # Search
    scores, ids = _index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue   # FAISS returns -1 for empty slots
        item = dict(_chunks[idx])
        item["_score"] = float(score)
        results.append(item)

    return results


def get_by_name(name: str) -> dict | None:
    """
    Exact or fuzzy lookup by assessment name.
    Used for compare queries: 'difference between OPQ and GSA'.
    """
    name_lower = name.lower().strip()
    for chunk in _chunks:
        if name_lower in chunk["name"].lower():
            return chunk
    return None


def get_all() -> list[dict]:
    """Return all chunks — used for safety validation of URLs."""
    return _chunks or []