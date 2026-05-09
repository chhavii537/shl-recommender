"""
build_index.py
--------------
Run ONCE after you have catalog.json.
Reads catalog.json → embeds each assessment → saves faiss.index + chunks.json

Usage:
    python build_index.py

Output:
    faiss.index   - vector index (binary)
    chunks.json   - maps index position → assessment dict
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CATALOG_PATH = "catalog.json"
INDEX_PATH   = "faiss.index"
CHUNKS_PATH  = "chunks.json"
MODEL_NAME   = "all-MiniLM-L6-v2"   # fast, 384-dim, good semantic quality


def make_text(item: dict) -> str:
    """
    Convert one catalog item into a single searchable text string.
    More context = better retrieval. We combine all useful fields.
    """
    keys = ", ".join(item.get("keys", []))
    levels = ", ".join(item.get("job_levels", []))
    langs = ", ".join(item.get("languages", []))
    remote = "remote testing available" if item.get("remote") == "yes" else ""
    adaptive = "adaptive/IRT" if item.get("adaptive") == "yes" else ""

    parts = [
        item.get("name", ""),
        item.get("description", ""),
        f"Test types: {keys}" if keys else "",
        f"Job levels: {levels}" if levels else "",
        f"Languages: {langs}" if langs else "",
        f"Duration: {item.get('duration', '')}" if item.get("duration") else "",
        remote,
        adaptive,
    ]
    return " | ".join(p for p in parts if p)


def build_index():
    print("Loading catalog...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Building text chunks...")
    chunks = []
    texts  = []
    for item in catalog:
        text = make_text(item)
        texts.append(text)
        # Store only what the agent needs to return / display
        chunks.append({
            "name":        item["name"],
            "url":         item["link"],
            "description": item.get("description", ""),
            "keys":        item.get("keys", []),
            "job_levels":  item.get("job_levels", []),
            "languages":   item.get("languages", []),
            "duration":    item.get("duration", ""),
            "remote":      item.get("remote", ""),
            "adaptive":    item.get("adaptive", ""),
            # primary test type code for the response schema
            "test_type":   _primary_type_code(item.get("keys", [])),
            "_text":       text,   # kept for debugging only
        })

    print(f"Embedding {len(texts)} items (this takes ~30 seconds)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # L2-normalise so cosine similarity = inner product (faster search)
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dim   = embeddings.shape[1]           # 384 for MiniLM
    index = faiss.IndexFlatIP(dim)        # Inner Product on normalised = cosine
    index.add(embeddings)

    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Index has {index.ntotal} vectors of dimension {dim}.")
    print("Test search: 'Java developer stakeholder management'")
    q = model.encode(["Java developer stakeholder management"], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, 5)
    for score, idx in zip(scores[0], ids[0]):
        print(f"  {score:.3f}  {chunks[idx]['name']}")


# ── helpers ──────────────────────────────────────────────────────────────────

# Map "keys" label → single-letter test type code used in the API response
_KEY_TO_CODE = {
    "Ability & Aptitude":           "A",
    "Assessment Exercises":         "E",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Knowledge & Skills":           "K",
    "Motivation":                   "M",
    "Personality & Behavior":       "P",
    "Simulations":                  "S",
}

def _primary_type_code(keys: list[str]) -> str:
    """Return the first matching single-letter code, or 'K' as fallback."""
    for k in keys:
        if k in _KEY_TO_CODE:
            return _KEY_TO_CODE[k]
    return "K"


if __name__ == "__main__":
    build_index()