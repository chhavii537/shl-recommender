"""
build_index.py
--------------
Builds a lightweight search index from catalog.json.
Uses TF-IDF instead of sentence-transformers to save memory on free hosting.
Run once locally: python build_index.py
"""

import json
import numpy as np
import faiss
import re
from collections import Counter
import math

CATALOG_PATH = "catalog.json"
INDEX_PATH   = "faiss.index"
CHUNKS_PATH  = "chunks.json"

_KEY_TO_CODE = {
    "Ability & Aptitude":             "A",
    "Assessment Exercises":           "E",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
    "Knowledge & Skills":             "K",
    "Motivation":                     "M",
    "Personality & Behavior":         "P",
    "Simulations":                    "S",
}

def primary_type_code(keys):
    for k in keys:
        if k in _KEY_TO_CODE:
            return _KEY_TO_CODE[k]
    return "K"

def make_text(item):
    keys   = " ".join(item.get("keys", []))
    levels = " ".join(item.get("job_levels", []))
    langs  = " ".join(item.get("languages", []))
    remote   = "remote" if item.get("remote") == "yes" else ""
    adaptive = "adaptive" if item.get("adaptive") == "yes" else ""
    return " ".join(filter(None, [
        item.get("name", ""),
        item.get("description", ""),
        keys, levels, langs, remote, adaptive
    ]))

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def build_tfidf(texts, dim=512):
    """Build a simple TF-IDF matrix projected to fixed dimension."""
    # Build vocabulary
    df = Counter()
    tokenized = []
    for text in texts:
        tokens = set(tokenize(text))
        tokenized.append(tokenize(text))
        df.update(tokens)

    N = len(texts)
    vocab = {w: i for i, w in enumerate(
        w for w, c in df.most_common(10000) if c >= 1
    )}
    V = len(vocab)

    # Build TF-IDF matrix
    matrix = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        total = len(tokens) or 1
        for word, count in tf.items():
            if word in vocab:
                j = vocab[word]
                tfidf = (count / total) * math.log(N / (df[word] + 1))
                matrix[i, j] = tfidf

    # Project to fixed dim using random projection (saves memory vs full vocab)
    np.random.seed(42)
    if V > dim:
        proj = np.random.randn(V, dim).astype(np.float32) / math.sqrt(dim)
        matrix = matrix @ proj

    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    return matrix, vocab

def build_index():
    print("Loading catalog...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    texts  = [make_text(item) for item in catalog]
    chunks = [{
        "name":        item["name"],
        "url":         item["link"],
        "description": item.get("description", ""),
        "keys":        item.get("keys", []),
        "job_levels":  item.get("job_levels", []),
        "languages":   item.get("languages", []),
        "duration":    item.get("duration", ""),
        "remote":      item.get("remote", ""),
        "adaptive":    item.get("adaptive", ""),
        "test_type":   primary_type_code(item.get("keys", [])),
        "_text":       texts[i],
    } for i, item in enumerate(catalog)]

    print("Building TF-IDF vectors...")
    embeddings, vocab = build_tfidf(texts, dim=512)

    print("Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save vocab alongside index for query-time encoding
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Done. {index.ntotal} vectors, dim={dim}")

    # Test search
    print("\nTest: 'Java developer stakeholder'")
    from retrieval_tfidf import encode_query
    q = encode_query("Java developer stakeholder", vocab, dim=512)
    scores, ids = index.search(q, 5)
    for score, idx in zip(scores[0], ids[0]):
        print(f"  {score:.3f}  {chunks[idx]['name']}")

if __name__ == "__main__":
    build_index()