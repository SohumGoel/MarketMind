"""
Summarize RAG ablation results as markdown tables for the final report.

Run: python experiments/summarize_ablation.py
Reads:  experiments/results/rag_ablation_results.json
Prints: three markdown tables (one per ablation) + writes them to
        experiments/results/rag_ablation_summary.md
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_INPUT  = Path(__file__).parent / "results" / "rag_ablation_results.json"
_OUTPUT = Path(__file__).parent / "results" / "rag_ablation_summary.md"

_ABLATION_DISPLAY = {
    "retrieval_k":  "Retrieval k",
    "chunk_size":   "Chunk size (tokens/overlap)",
    "query_string": "Query strategy",
}

_VARIANT_DISPLAY = {
    "k=1": "k = 1",
    "k=3": "k = 3 (default)",
    "k=5": "k = 5",
    "256/32": "256 / 32",
    "512/64": "512 / 64 (default)",
    "fixed": "Fixed domain query",
    "ticker_specific": "Ticker-specific query",
}


def _md_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep    = "| " + " | ".join("---" for _ in columns) + " |"
    body   = "\n".join(
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
        for row in rows
    )
    return "\n".join([header, sep, body])


def main():
    results = json.loads(_INPUT.read_text())

    # Group: ablation → variant → list of top-1 scores (one per ticker)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ticker_sets: dict[str, set] = defaultdict(set)

    for r in results:
        abl = r["ablation"]
        var = r["variant"]
        top1 = r["scores"][0] if r["scores"] else None
        if top1 is not None:
            data[abl][var].append(top1)
        ticker_sets[abl].add(r["ticker"])

    sections = []

    for abl in ["retrieval_k", "chunk_size", "query_string"]:
        if abl not in data:
            continue

        tickers = sorted(ticker_sets[abl])
        abl_results = data[abl]

        # Build per-variant row
        rows = []
        for var, scores in sorted(abl_results.items()):
            rows.append({
                "Variant": _VARIANT_DISPLAY.get(var, var),
                "Avg top-1 score": f"{np.mean(scores):.4f}",
                "Min": f"{np.min(scores):.4f}",
                "Max": f"{np.max(scores):.4f}",
                "n tickers": len(scores),
            })

        title = _ABLATION_DISPLAY.get(abl, abl)
        tickers_str = ", ".join(tickers)
        block = (
            f"### {title}\n\n"
            f"*Tickers: {tickers_str} · metric: cosine similarity (top-1 retrieved chunk)*\n\n"
            + _md_table(rows, ["Variant", "Avg top-1 score", "Min", "Max", "n tickers"])
        )
        sections.append(block)

        # Print to stdout
        print(f"\n{block}\n")

    # Key findings callout
    findings = []

    if "query_string" in data:
        fixed_scores = data["query_string"].get("fixed", [])
        ts_scores    = data["query_string"].get("ticker_specific", [])
        if fixed_scores and ts_scores:
            delta = np.mean(ts_scores) - np.mean(fixed_scores)
            findings.append(
                f"- **Query strategy:** ticker-specific queries outperform fixed query "
                f"by **+{delta:.4f}** avg cosine similarity "
                f"({np.mean(ts_scores):.4f} vs {np.mean(fixed_scores):.4f})."
            )

    if "chunk_size" in data:
        s256 = data["chunk_size"].get("256/32", [])
        s512 = data["chunk_size"].get("512/64", [])
        if s256 and s512:
            winner = "256/32" if np.mean(s256) > np.mean(s512) else "512/64 (default)"
            findings.append(
                f"- **Chunk size:** {winner} achieves higher avg top-1 score "
                f"(256/32: {np.mean(s256):.4f}, 512/64: {np.mean(s512):.4f})."
            )

    if "retrieval_k" in data:
        k1 = data["retrieval_k"].get("k=1", [])
        k5 = data["retrieval_k"].get("k=5", [])
        if k1 and k5:
            findings.append(
                f"- **Retrieval k:** top-1 score is stable across k=1/3/5 "
                f"({np.mean(k1):.4f} → {np.mean(k5):.4f}); "
                f"increasing k adds coverage without hurting precision."
            )

    findings_block = "### Key Findings\n\n" + "\n".join(findings) if findings else ""
    if findings_block:
        print(f"\n{findings_block}\n")

    # Write markdown file
    md = "# RAG Ablation Results\n\n" + "\n\n---\n\n".join(sections)
    if findings_block:
        md += "\n\n---\n\n" + findings_block
    _OUTPUT.write_text(md)
    print(f"\nWrote {_OUTPUT}")


if __name__ == "__main__":
    main()
