"""
MarketMind data pipeline.

Orchestrates all agents into a single FinGPT-format prompt ready for inference.

Usage:
    from agents.pipeline import build_prompt
    prompt = build_prompt("AAPL", "2024-01-01", "2024-01-07", av_api_key="...")
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from agents.data_collection.price_agent import PriceAgent
from agents.data_collection.news_agent import NewsAgent
from agents.data_collection.sec_agent import SECAgent
from agents.retrieval.rag_pipeline import RAGPipeline

_LOG_DIR = Path(__file__).parent.parent / "logs"
logger = logging.getLogger(__name__)


def _ticker_rag_query(company_name: str) -> str:
    return f"What are the key risks, revenue drivers, and forward outlook for {company_name}?"


_FINGPT_SYSTEM = (
    "You are a seasoned stock market analyst. Your task is to list the positive "
    "developments and potential concerns for companies based on relevant news and "
    "basic financial data from the past weeks, then make a prediction about the "
    "companies' stock price movement for the upcoming week.\n\n"
    "[Positive Developments]:\n1. ...\n\n"
    "[Potential Concerns]:\n1. ...\n\n"
    "[Prediction & Analysis]:\n..."
)


def _format_price_summary(price_data: dict) -> str:
    prices = price_data["prices"]
    financials = price_data["financials"]
    if not prices:
        return "No price data available."

    first, last = prices[0], prices[-1]
    pct_change = 100 * (last["close"] - first["open"]) / first["open"] if first["open"] else 0

    lines = [
        f"Period: {first['date']} to {last['date']}",
        f"Open: ${first['open']:.2f}  |  Close: ${last['close']:.2f}  |  Change: {pct_change:+.2f}%",
        f"52-week High: ${financials.get('week_52_high', 'N/A')}  |  Low: ${financials.get('week_52_low', 'N/A')}",
        f"Market Cap: {financials.get('market_cap', 'N/A')}  |  P/E: {financials.get('pe_ratio', 'N/A')}  |  EPS: {financials.get('eps', 'N/A')}",
        f"Sector: {financials.get('sector', 'N/A')}  |  Industry: {financials.get('industry', 'N/A')}",
    ]
    return "\n".join(lines)


def _format_news_headlines(articles: list[dict], max_articles: int = 10) -> str:
    if not articles:
        return "No news articles found for this period."
    lines = []
    for i, a in enumerate(articles[:max_articles], 1):
        sentiment = a.get("sentiment_label", "Neutral")
        lines.append(f"{i}. [{sentiment}] {a['title']}")
        if a.get("summary"):
            lines.append(f"   {a['summary'][:200]}")
    return "\n".join(lines)


def _format_rag_passages(passages: list[dict], max_passages: int = 3) -> str:
    if not passages:
        return "No SEC filing passages retrieved."
    lines = []
    for i, p in enumerate(passages[:max_passages], 1):
        lines.append(f"{i}. [Source: {p['doc_id']}]\n   {p['text'][:300]}")
    return "\n".join(lines)


def build_prompt(
    ticker: str,
    start_date: str,
    end_date: str,
    av_api_key: str = None,
    sec_lookback_days: int = 90,
    rag_top_k: int = 3,
) -> dict:
    """
    Fetch all data and assemble a FinGPT-format prompt for a given ticker and week.

    Args:
        ticker:             Stock ticker (e.g. 'AAPL')
        start_date:         Week start date 'YYYY-MM-DD'
        end_date:           Week end date 'YYYY-MM-DD'
        av_api_key:         Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)
        sec_lookback_days:  How many days back to search for SEC filings
        rag_top_k:          Number of RAG passages to retrieve from SEC docs

    Returns:
        dict with keys:
            'prompt'  — full ChatML-formatted string ready for model inference
            'ticker', 'start_date', 'end_date'
            'raw'     — raw data from each agent (for debugging)
    """
    raw = {}

    # Price data (always available via yfinance)
    price_result = PriceAgent().fetch(ticker, start_date, end_date)
    raw["price"] = price_result
    company_name = price_result["data"]["financials"].get("company_name", ticker)
    description  = price_result["data"]["financials"].get("description", "")[:500]

    # News headlines
    try:
        news_result = NewsAgent(api_key=av_api_key).fetch(ticker, start_date, end_date)
        raw["news"] = news_result
        articles = news_result["data"]
    except Exception as e:
        raw["news"] = {"error": str(e)}
        articles = []

    # SEC filings + RAG
    sec_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=sec_lookback_days)).strftime("%Y-%m-%d")
    rag = RAGPipeline()
    try:
        sec_result = SECAgent().fetch(ticker, sec_start, end_date)
        raw["sec"] = sec_result
        for filing in sec_result["data"]:
            if filing.get("full_text"):
                rag.index_document(filing["full_text"], doc_id=f"{filing['form_type']}_{filing['filed_at']}")
    except Exception as e:
        raw["sec"] = {"error": str(e)}

    rag_passages = rag.retrieve(query=_ticker_rag_query(company_name), top_k=rag_top_k)

    # Assemble FinGPT-style user prompt
    user_content = f"""[Company Introduction]
{company_name} ({ticker})
{description}

[Basic Financials]
{_format_price_summary(price_result['data'])}

[News Headlines and Summaries]
{_format_news_headlines(articles)}

[Relevant SEC Filing Passages]
{_format_rag_passages(rag_passages)}

Based on the above information, provide your analysis and prediction for {ticker}'s stock price movement for the week following {end_date}."""

    prompt = (
        f"<|im_start|>system\n{_FINGPT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "prompt": prompt,
        "raw": raw,
        "_rag_chunks": rag_passages,
        "_news_count": len(articles),
    }


def log_result(prompt_dict: dict, direction: str, llm_output: str, model_backend: str) -> None:
    """Append one inference record to logs/run_log.jsonl."""
    _LOG_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "ticker": prompt_dict.get("ticker"),
        "start_date": prompt_dict.get("start_date"),
        "end_date": prompt_dict.get("end_date"),
        "model_backend": model_backend,
        "rag_chunks": prompt_dict.get("_rag_chunks", []),
        "news_count": prompt_dict.get("_news_count", 0),
        "direction": direction,
        "llm_output": llm_output,
    }
    with (_LOG_DIR / "run_log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
