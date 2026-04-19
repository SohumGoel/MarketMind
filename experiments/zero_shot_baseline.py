"""
Zero-shot baseline vs full pipeline comparison.

Runs the same LLM (GPT-4o via gateway) on two prompt types:
  1. Zero-shot: ticker + company + date range only — no data
  2. Full pipeline: price + news + SEC/RAG assembled prompt

Uses actual yfinance price data as ground truth.

Usage:
    python experiments/zero_shot_baseline.py

Output:
    experiments/results/zero_shot_baseline_results.json
    experiments/results/zero_shot_baseline_summary.md
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from openai import OpenAI

from agents.pipeline import build_prompt
from evaluation.metrics import extract_direction_from_output

RESULTS_DIR = Path(__file__).parent / "results"

GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
GATEWAY_MODEL          = "gpt-5.4"
GATEWAY_MODEL_FALLBACKS = ["claude-sonnet-4-20250514-v1:0", "gemini-2.5-pro", "gpt-5-mini"]

TICKERS = [
    {"ticker": "AAPL", "company": "Apple Inc.",      "sector": "Technology"},
    {"ticker": "MSFT", "company": "Microsoft Corp.", "sector": "Technology"},
    {"ticker": "JPM",  "company": "JPMorgan Chase",  "sector": "Financials"},
    {"ticker": "NVDA", "company": "NVIDIA Corp.",    "sector": "Technology"},
    {"ticker": "TSLA", "company": "Tesla Inc.",      "sector": "Consumer Cyclical"},
]

DATE_RANGES = [
    ("2024-10-07", "2024-10-11"),
    ("2024-11-04", "2024-11-08"),
]

_SYSTEM = (
    "You are a stock market analyst. Predict whether a stock will go up, down, "
    "or remain neutral for the given week. Be concise and end your answer with "
    "exactly one word on its own line: up, down, or neutral."
)


def get_ground_truth(ticker: str, start_date: str, end_date: str) -> tuple[float | None, str]:
    """Fetch actual price change from yfinance and return (pct_change, direction)."""
    try:
        hist = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if hist.empty:
            return None, "unknown"
        open_price  = float(hist["Open"].iloc[0])
        close_price = float(hist["Close"].iloc[-1])
        pct = 100 * (close_price - open_price) / open_price
        if pct > 0.5:
            direction = "up"
        elif pct < -0.5:
            direction = "down"
        else:
            direction = "neutral"
        return round(pct, 2), direction
    except Exception:
        return None, "unknown"


def run_zero_shot(client: OpenAI, ticker: str, company: str, start_date: str, end_date: str) -> dict:
    """Call LLM with bare minimum prompt — no data."""
    prompt = (
        f"Predict the stock price direction for {company} ({ticker}) "
        f"for the trading week of {start_date} to {end_date}.\n\n"
        f"Answer with up, down, or neutral."
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    raw = None
    for model in [GATEWAY_MODEL] + GATEWAY_MODEL_FALLBACKS:
        try:
            resp = client.chat.completions.create(model=model, max_tokens=100, temperature=0.0, messages=messages)
            raw = resp.choices[0].message.content.strip()
            break
        except Exception:
            continue
    if not raw:
        return {"direction": "unknown", "reasoning": "All models failed."}
    direction = extract_direction_from_output(raw)
    return {"direction": direction, "reasoning": raw}


def run_full_pipeline(client: OpenAI, ticker: str, start_date: str, end_date: str, av_key: str | None, gateway_key: str) -> dict:
    """Build full data prompt then call LLM."""
    from agents.synthesis.evaluator_agent import EvaluatorAgent
    prompt_dict = build_prompt(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        av_api_key=av_key,
        gateway_key=gateway_key,
    )
    agent = EvaluatorAgent(backend="gateway", api_key=gateway_key, gateway_model=GATEWAY_MODEL)
    result = agent.predict(prompt_dict, max_new_tokens=600)
    return {"direction": result["direction"], "reasoning": result["reasoning"]}


def run_comparison(gateway_key: str, av_key: str | None) -> list[dict]:
    client = OpenAI(api_key=gateway_key, base_url=GATEWAY_BASE_URL)
    results = []

    for t in TICKERS:
        for start_date, end_date in DATE_RANGES:
            ticker, company = t["ticker"], t["company"]
            print(f"\n[{ticker}] {start_date} → {end_date}")

            pct, truth = get_ground_truth(ticker, start_date, end_date)
            print(f"  Ground truth: {truth} ({pct:+.2f}%)" if pct is not None else "  Ground truth: unknown")

            record = {
                "ticker": ticker,
                "company": company,
                "sector": t["sector"],
                "start_date": start_date,
                "end_date": end_date,
                "actual_pct_change": pct,
                "ground_truth": truth,
            }

            # Zero-shot
            print("  Running zero-shot...")
            try:
                zs = run_zero_shot(client, ticker, company, start_date, end_date)
                record["zero_shot_direction"] = zs["direction"]
                record["zero_shot_reasoning"] = zs["reasoning"]
                record["zero_shot_correct"]   = zs["direction"] == truth if truth != "unknown" else None
                print(f"    → {zs['direction']} ({'✓' if record['zero_shot_correct'] else '✗'})")
            except Exception as e:
                print(f"  ⚠ Zero-shot failed: {e}")
                record["zero_shot_direction"] = "error"
                record["zero_shot_reasoning"] = str(e)
                record["zero_shot_correct"]   = None

            time.sleep(1)

            # Full pipeline
            print("  Running full pipeline...")
            try:
                fp = run_full_pipeline(client, ticker, start_date, end_date, av_key, gateway_key)
                record["pipeline_direction"] = fp["direction"]
                record["pipeline_reasoning"] = fp["reasoning"]
                record["pipeline_correct"]   = fp["direction"] == truth if truth != "unknown" else None
                print(f"    → {fp['direction']} ({'✓' if record['pipeline_correct'] else '✗'})")
            except Exception as e:
                print(f"  ⚠ Pipeline failed: {e}")
                record["pipeline_direction"] = "error"
                record["pipeline_reasoning"] = str(e)
                record["pipeline_correct"]   = None

            results.append(record)
            time.sleep(1)

    return results


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / "zero_shot_baseline_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {json_path}")

    md_path = RESULTS_DIR / "zero_shot_baseline_summary.md"
    _write_markdown(results, md_path)
    print(f"Summary → {md_path}")


def _write_markdown(results: list[dict], path: Path) -> None:
    zs_correct = [r["zero_shot_correct"] for r in results if r.get("zero_shot_correct") is not None]
    fp_correct = [r["pipeline_correct"]  for r in results if r.get("pipeline_correct")  is not None]

    zs_acc = sum(zs_correct) / len(zs_correct) if zs_correct else 0
    fp_acc = sum(fp_correct) / len(fp_correct)  if fp_correct  else 0

    lines = [
        "# Zero-shot Baseline vs Full Pipeline",
        "",
        f"**Model:** {GATEWAY_MODEL} (CMU AI Gateway, same model for both conditions)",
        f"**Date generated:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Results Table",
        "",
        "| Ticker | Date | Actual Δ% | Truth | Zero-shot | ✓ | Full Pipeline | ✓ |",
        "|--------|------|-----------|-------|-----------|---|---------------|---|",
    ]

    for r in results:
        actual = f"{r['actual_pct_change']:+.1f}%" if r.get("actual_pct_change") is not None else "N/A"
        zs_ok = "✓" if r.get("zero_shot_correct") else ("✗" if r.get("zero_shot_correct") is False else "—")
        fp_ok = "✓" if r.get("pipeline_correct")  else ("✗" if r.get("pipeline_correct")  is False else "—")
        lines.append(
            f"| {r['ticker']} | {r['start_date']} | {actual} | {r['ground_truth']} "
            f"| {r.get('zero_shot_direction','—')} | {zs_ok} "
            f"| {r.get('pipeline_direction','—')} | {fp_ok} |"
        )

    lines += [
        "",
        "## Accuracy Summary",
        "",
        f"| Condition | Correct | Total | Accuracy |",
        f"|-----------|---------|-------|----------|",
        f"| Zero-shot (no data) | {sum(zs_correct)} | {len(zs_correct)} | {zs_acc:.0%} |",
        f"| Full pipeline (price + news + SEC/RAG) | {sum(fp_correct)} | {len(fp_correct)} | {fp_acc:.0%} |",
        "",
        f"> **Key finding:** Full pipeline {'outperforms' if fp_acc > zs_acc else 'matches'} zero-shot "
        f"by {abs(fp_acc - zs_acc):.0%} ({fp_acc:.0%} vs {zs_acc:.0%}), "
        f"using the same underlying LLM ({GATEWAY_MODEL}).",
        "",
        "## Sample Reasoning",
        "",
    ]

    for r in results[:3]:
        actual = f"{r['actual_pct_change']:+.2f}%" if r.get("actual_pct_change") is not None else "N/A"
        lines += [
            f"### {r['ticker']} ({r['start_date']}) — Actual: {actual} ({r['ground_truth']})",
            "",
            f"**Zero-shot ({r.get('zero_shot_direction','—')}):**",
            f"> {r.get('zero_shot_reasoning','')[:300].strip()}",
            "",
            f"**Full pipeline ({r.get('pipeline_direction','—')}):**",
            f"> {r.get('pipeline_reasoning','')[:300].strip()}...",
            "",
        ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    gateway_key = os.environ.get("CMU_AI_GATEWAY_KEY")
    if not gateway_key:
        print("ERROR: CMU_AI_GATEWAY_KEY not set.")
        sys.exit(1)

    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

    results = run_comparison(gateway_key, av_key)
    save_results(results)
    print("\nDone.")
