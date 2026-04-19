"""
RAG ablation experiments for MarketMind final report.

Three ablations across AAPL, JNJ, XOM, WMT, JPM:
  1. Retrieval k  — k=1, k=3, k=5
  2. Chunk size   — 256/32 vs 512/64 (current)
  3. Query string — fixed vs ticker-specific

Uses local sec_cache/ files; no network requests.

Run: python experiments/rag_ablation.py
Writes: experiments/results/rag_ablation_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import faiss
from dotenv import load_dotenv
from huggingface_hub import login as _hf_login
from sentence_transformers import SentenceTransformer

load_dotenv()
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    _hf_login(token=_hf_token, add_to_git_credential=False)

from agents.data_collection.sec_agent import SECAgent

_TICKERS = ["AAPL", "JNJ", "XOM", "WMT", "JPM"]
_MODEL   = "all-MiniLM-L6-v2"
_FIXED_QUERY = (
    "material risk factors, revenue guidance, earnings surprise, "
    "forward outlook, capital expenditure, debt obligations"
)
_TICKER_QUERIES = {
    "AAPL": "What are the key risks, revenue drivers, and forward outlook for Apple?",
    "JNJ":  "What are the key risks, revenue drivers, and forward outlook for Johnson & Johnson?",
    "XOM":  "What are the key risks, revenue drivers, and forward outlook for ExxonMobil?",
    "WMT":  "What are the key risks, revenue drivers, and forward outlook for Walmart?",
    "JPM":  "What are the key risks, revenue drivers, and forward outlook for JPMorgan Chase?",
}


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def _build_index(text: str, doc_id: str, model: SentenceTransformer,
                 chunk_size: int, overlap: int):
    chunks = _chunk_text(text, chunk_size, overlap)
    if not chunks:
        return [], [], None
    embs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return chunks, [doc_id] * len(chunks), idx


def _retrieve(query: str, model: SentenceTransformer, chunks, doc_ids, index, top_k: int):
    if index is None or index.ntotal == 0:
        return []
    q = np.array(model.encode([query], normalize_embeddings=True, show_progress_bar=False), dtype="float32")
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q, k)
    return [
        {"text": chunks[i], "doc_id": doc_ids[i], "score": float(scores[0][r])}
        for r, i in enumerate(indices[0]) if i < len(chunks)
    ]


def main():
    print("Loading SentenceBERT model...")
    model = SentenceTransformer(_MODEL)

    print("Fetching SEC filings from cache...")
    agent = SECAgent()
    filing_texts = {}
    for ticker in _TICKERS:
        try:
            r = agent.fetch(ticker, "2024-01-01", "2026-01-01")
            filing_texts[ticker] = r["data"][0]["full_text"] if r["data"] else ""
        except Exception as e:
            print(f"  Warning: could not fetch {ticker}: {e}")
            filing_texts[ticker] = ""

    results = []

    # ── Ablation 1: Retrieval k ────────────────────────────────────────────────
    print("\nAblation 1: Retrieval k")
    for ticker in _TICKERS:
        text = filing_texts[ticker]
        if not text:
            continue
        chunks, doc_ids, index = _build_index(text, ticker, model, 512, 64)
        for k in [1, 3, 5]:
            hits = _retrieve(_FIXED_QUERY, model, chunks, doc_ids, index, top_k=k)
            results.append({
                "ticker": ticker,
                "ablation": "retrieval_k",
                "variant": f"k={k}",
                "retrieved_chunks": [h["text"] for h in hits],
                "scores": [h["score"] for h in hits],
            })
        print(f"  {ticker}: k=1/3/5 done")

    # ── Ablation 2: Chunk size ─────────────────────────────────────────────────
    print("\nAblation 2: Chunk size")
    for ticker in _TICKERS:
        text = filing_texts[ticker]
        if not text:
            continue
        for cs, ov, label in [(256, 32, "256/32"), (512, 64, "512/64")]:
            chunks, doc_ids, index = _build_index(text, ticker, model, cs, ov)
            hits = _retrieve(_FIXED_QUERY, model, chunks, doc_ids, index, top_k=3)
            results.append({
                "ticker": ticker,
                "ablation": "chunk_size",
                "variant": label,
                "retrieved_chunks": [h["text"] for h in hits],
                "scores": [h["score"] for h in hits],
            })
        print(f"  {ticker}: chunk size variants done")

    # ── Ablation 3: Query string ───────────────────────────────────────────────
    print("\nAblation 3: Query string")
    for ticker in _TICKERS:
        text = filing_texts[ticker]
        if not text:
            continue
        chunks, doc_ids, index = _build_index(text, ticker, model, 512, 64)
        for query, label in [(_FIXED_QUERY, "fixed"), (_TICKER_QUERIES[ticker], "ticker_specific")]:
            hits = _retrieve(query, model, chunks, doc_ids, index, top_k=3)
            results.append({
                "ticker": ticker,
                "ablation": "query_string",
                "variant": label,
                "query": query,
                "retrieved_chunks": [h["text"] for h in hits],
                "scores": [h["score"] for h in hits],
            })
        print(f"  {ticker}: query variants done")

    # ── Write results ──────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "results" / "rag_ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary: avg top-1 cosine score per variant ──")
    from collections import defaultdict
    variant_scores = defaultdict(list)
    for r in results:
        if r["scores"]:
            variant_scores[(r["ablation"], r["variant"])].append(r["scores"][0])

    print(f"{'Ablation':<20} {'Variant':<20} {'Avg top-1 score':>16}")
    print("-" * 58)
    for (abl, var), scores in sorted(variant_scores.items()):
        print(f"{abl:<20} {var:<20} {np.mean(scores):>16.4f}")


if __name__ == "__main__":
    main()
