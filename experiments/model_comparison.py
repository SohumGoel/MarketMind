"""
Fine-tuned vs zero-shot baseline comparison for MarketMind report.

Runs both a gateway model (zero-shot) and an optional fine-tuned/pre-trained HF model
on the same set of tickers and date ranges, then saves a side-by-side comparison.

Usage:
    # Gateway only (zero-shot baseline — works right now, no local model needed)
    python experiments/model_comparison.py --gateway_only

    # Gateway vs pre-trained FinGPT HF model (e.g. FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)
    python experiments/model_comparison.py --hf_model FinGPT/fingpt-forecaster_dow30_llama2-7b_lora

    # Gateway vs Yash's local fine-tuned checkpoint
    python experiments/model_comparison.py --local_checkpoint /path/to/checkpoint

Output:
    experiments/results/model_comparison_results.json
    experiments/results/model_comparison_summary.md
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

from agents.pipeline import build_prompt
from agents.synthesis.evaluator_agent import EvaluatorAgent
from evaluation.metrics import extract_direction_from_output

RESULTS_DIR = Path(__file__).parent / "results"

# 5 tickers across sectors — all have sec_cache entries for fast pipeline runs
TICKERS = [
    {"ticker": "AAPL", "company": "Apple Inc.",       "sector": "Technology"},
    {"ticker": "MSFT", "company": "Microsoft Corp.",   "sector": "Technology"},
    {"ticker": "JPM",  "company": "JPMorgan Chase",    "sector": "Financials"},
    {"ticker": "NVDA", "company": "NVIDIA Corp.",      "sector": "Technology"},
    {"ticker": "TSLA", "company": "Tesla Inc.",        "sector": "Consumer Cyclical"},
]

# 2 date ranges per ticker = 10 examples total
DATE_RANGES = [
    ("2024-10-07", "2024-10-11"),
    ("2024-11-04", "2024-11-08"),
]

# Gateway model to use for zero-shot baseline
GATEWAY_MODEL          = "gpt-5.4"
GATEWAY_MODEL_FALLBACKS = ["claude-sonnet-4-20250514-v1:0", "gemini-2.5-pro", "gpt-5-mini"]


def run_gateway(prompt_dict: dict, gateway_key: str) -> dict:
    agent = EvaluatorAgent(
        backend="gateway",
        api_key=gateway_key,
        gateway_model=GATEWAY_MODEL,
    )
    result = agent.predict(prompt_dict, max_new_tokens=600)
    return {
        "direction": result["direction"],
        "reasoning": result["reasoning"],
    }


def run_local(prompt_dict: dict, model_path: str) -> dict:
    agent = EvaluatorAgent(backend="huggingface", model_path=model_path)
    result = agent.predict(prompt_dict, max_new_tokens=600)
    return {
        "direction": result["direction"],
        "reasoning": result["reasoning"],
    }


def direction_matches(pred: str, actual_pct_change: float) -> bool:
    """Check if predicted direction matches actual price movement."""
    if actual_pct_change > 0.5:
        truth = "up"
    elif actual_pct_change < -0.5:
        truth = "down"
    else:
        truth = "neutral"
    return pred == truth


def run_comparison(gateway_key: str, model_path: str | None, av_key: str | None) -> list[dict]:
    results = []

    for ticker_info in TICKERS:
        ticker = ticker_info["ticker"]
        for start_date, end_date in DATE_RANGES:
            print(f"\n[{ticker}] {start_date} → {end_date}")

            # Build prompt once, reuse for both models
            try:
                prompt_dict = build_prompt(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    av_api_key=av_key,
                    gateway_key=gateway_key,
                )
            except Exception as e:
                print(f"  ⚠ Pipeline failed: {e}")
                continue

            # Get actual price change from prompt raw data
            prices = prompt_dict.get("raw", {}).get("price", {}).get("data", {}).get("prices", [])
            if prices:
                actual_pct = 100 * (prices[-1]["close"] - prices[0]["open"]) / prices[0]["open"]
            else:
                actual_pct = None

            record = {
                "ticker": ticker,
                "company": ticker_info["company"],
                "sector": ticker_info["sector"],
                "start_date": start_date,
                "end_date": end_date,
                "actual_pct_change": round(actual_pct, 2) if actual_pct is not None else None,
            }

            # Zero-shot gateway
            print(f"  Running zero-shot ({GATEWAY_MODEL})...")
            try:
                gw = run_gateway(prompt_dict, gateway_key)
                record["gateway_direction"] = gw["direction"]
                record["gateway_reasoning"] = gw["reasoning"]
                record["gateway_correct"] = direction_matches(gw["direction"], actual_pct) if actual_pct is not None else None
                print(f"    → {gw['direction']}")
            except Exception as e:
                print(f"  ⚠ Gateway failed: {e}")
                record["gateway_direction"] = "error"
                record["gateway_reasoning"] = str(e)
                record["gateway_correct"] = None

            time.sleep(1)  # avoid rate limiting

            # Fine-tuned / pre-trained model (optional)
            if model_path:
                print(f"  Running fine-tuned ({model_path})...")
                try:
                    ft = run_local(prompt_dict, model_path)
                    record["finetuned_direction"] = ft["direction"]
                    record["finetuned_reasoning"] = ft["reasoning"]
                    record["finetuned_correct"] = direction_matches(ft["direction"], actual_pct) if actual_pct is not None else None
                    print(f"    → {ft['direction']}")
                except Exception as e:
                    print(f"  ⚠ Fine-tuned model failed: {e}")
                    record["finetuned_direction"] = "error"
                    record["finetuned_reasoning"] = str(e)
                    record["finetuned_correct"] = None

            results.append(record)

    return results


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / "model_comparison_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    md_path = RESULTS_DIR / "model_comparison_summary.md"
    _write_markdown(results, md_path)
    print(f"Summary saved → {md_path}")


def _write_markdown(results: list[dict], path: Path) -> None:
    has_finetuned = any("finetuned_direction" in r for r in results)

    lines = [
        "# Model Comparison: Fine-tuned vs Zero-shot Baseline",
        "",
        f"**Zero-shot model:** {GATEWAY_MODEL} (CMU AI Gateway)",
        f"**Fine-tuned model:** {'Yes — see results' if has_finetuned else 'Not run — gateway_only mode'}",
        f"**Date generated:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    if has_finetuned:
        lines += [
            "## Results Table",
            "",
            "| Ticker | Date Range | Actual Δ% | Zero-shot | Correct | Fine-tuned | Correct |",
            "|--------|------------|-----------|-----------|---------|------------|---------|",
        ]
        for r in results:
            actual = f"{r['actual_pct_change']:+.1f}%" if r.get("actual_pct_change") is not None else "N/A"
            gw_correct = "✓" if r.get("gateway_correct") else ("✗" if r.get("gateway_correct") is False else "—")
            ft_correct = "✓" if r.get("finetuned_correct") else ("✗" if r.get("finetuned_correct") is False else "—")
            lines.append(
                f"| {r['ticker']} | {r['start_date']} | {actual} "
                f"| {r.get('gateway_direction','—')} | {gw_correct} "
                f"| {r.get('finetuned_direction','—')} | {ft_correct} |"
            )
    else:
        lines += [
            "## Zero-shot Baseline Results",
            "",
            "| Ticker | Date Range | Actual Δ% | Zero-shot Direction | Correct |",
            "|--------|------------|-----------|---------------------|---------|",
        ]
        for r in results:
            actual = f"{r['actual_pct_change']:+.1f}%" if r.get("actual_pct_change") is not None else "N/A"
            gw_correct = "✓" if r.get("gateway_correct") else ("✗" if r.get("gateway_correct") is False else "—")
            lines.append(
                f"| {r['ticker']} | {r['start_date']} | {actual} "
                f"| {r.get('gateway_direction','—')} | {gw_correct} |"
            )

    # Accuracy summary
    gw_correct_list = [r["gateway_correct"] for r in results if r.get("gateway_correct") is not None]
    if gw_correct_list:
        gw_acc = sum(gw_correct_list) / len(gw_correct_list)
        lines += ["", f"**Zero-shot accuracy:** {gw_acc:.0%} ({sum(gw_correct_list)}/{len(gw_correct_list)})"]

    if has_finetuned:
        ft_correct_list = [r["finetuned_correct"] for r in results if r.get("finetuned_correct") is not None]
        if ft_correct_list:
            ft_acc = sum(ft_correct_list) / len(ft_correct_list)
            lines.append(f"**Fine-tuned accuracy:** {ft_acc:.0%} ({sum(ft_correct_list)}/{len(ft_correct_list)})")

    # Qualitative reasoning samples (first 3)
    lines += ["", "## Sample Reasoning Outputs", ""]
    for r in results[:3]:
        lines += [
            f"### {r['ticker']} ({r['start_date']})",
            f"Actual: {r.get('actual_pct_change', 'N/A')}%",
            "",
            f"**Zero-shot ({r.get('gateway_direction','—')}):**",
            f"> {r.get('gateway_reasoning','')[:400].strip()}...",
            "",
        ]
        if "finetuned_reasoning" in r:
            lines += [
                f"**Fine-tuned ({r.get('finetuned_direction','—')}):**",
                f"> {r.get('finetuned_reasoning','')[:400].strip()}...",
                "",
            ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway_only", action="store_true",
                        help="Run only zero-shot gateway baseline (no local model needed)")
    parser.add_argument("--hf_model", type=str, default=None,
                        help="HuggingFace repo ID for pre-trained FinGPT model (e.g. FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="Path to Yash's local fine-tuned checkpoint directory")
    args = parser.parse_args()

    gateway_key = os.environ.get("CMU_AI_GATEWAY_KEY")
    if not gateway_key:
        print("ERROR: CMU_AI_GATEWAY_KEY not set in environment.")
        sys.exit(1)

    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

    model_path = args.local_checkpoint or args.hf_model

    if not args.gateway_only and not model_path:
        print("Pass --gateway_only, --hf_model <repo_id>, or --local_checkpoint <path>.")
        sys.exit(1)

    results = run_comparison(gateway_key, model_path, av_key)
    save_results(results)

    print("\nDone.")
