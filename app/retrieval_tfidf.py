"""
retrieval_tfidf.py
------------------
Lightweight TF-IDF retrieval — no PyTorch, no heavy models.
Fits in 512MB RAM easily.
"""

import json
import math
import re
import numpy as np
import faiss
from collections import Counter

INDEX_PATH  = "faiss.index"
CHUNKS_PATH = "chunks.json"
VOCAB_PATH  = "vocab.json"
DIM         = 512

_index:  faiss.Index = None
_chunks: list[dict]  = None
_vocab:  dict        = None


def load():
    global _index, _chunks, _vocab
    print(f"Loading FAISS index...")
    _index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        _chunks = json.load(f)
    with open(VOCAB_PATH) as f:
        _vocab = json.load(f)
    print(f"Retrieval ready. {_index.ntotal} assessments indexed.")


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def encode_query(query: str, vocab: dict = None, dim: int = DIM) -> np.ndarray:
    if vocab is None:
        vocab = _vocab
    tokens = tokenize(query)
    tf = Counter(tokens)
    total = len(tokens) or 1
    V = max(vocab.values()) + 1 if vocab else dim

    vec = np.zeros(V, dtype=np.float32)
    for word, count in tf.items():
        if word in vocab:
            vec[vocab[word]] = count / total

    # Same random projection as build time
    np.random.seed(42)
    if V > dim:
        proj = np.random.randn(V, dim).astype(np.float32) / math.sqrt(dim)
        vec = vec @ proj

    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.reshape(1, -1).astype(np.float32)


def search(query: str, k: int = 15) -> list[dict]:
    if _index is None:
        raise RuntimeError("Retrieval not initialised. Call load() first.")
    q = encode_query(query)
    scores, ids = _index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(_chunks[idx])
        item["_score"] = float(score)
        results.append(item)
    return results


def get_by_name(name: str) -> dict | None:
    name_lower = name.lower().strip()
    for chunk in _chunks:
        if name_lower in chunk["name"].lower():
            return chunk
    return None


def get_all() -> list[dict]:
    return _chunks or []