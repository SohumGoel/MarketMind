"""
News headline fetching agent using Alpha Vantage NEWS_SENTIMENT endpoint.

Requires: ALPHA_VANTAGE_API_KEY environment variable.
Free tier: 25 requests/day. Set AV_API_KEY in your .env file.
"""

import logging
import os
import time
import requests
from datetime import datetime
from agents.base_agent import BaseAgent

_AV_BASE = "https://www.alphavantage.co/query"
logger = logging.getLogger(__name__)


class NewsAgent(BaseAgent):
    """
    Fetches recent news headlines and sentiment scores for a ticker
    using Alpha Vantage NEWS_SENTIMENT endpoint.

    Falls back to a time-filtered slice of the API response since AV
    does not support date range filtering natively — we filter client-side.
    """

    def __init__(self, api_key: str = None):
        # Store the explicitly passed key; env keys are loaded lazily only if needed
        self._explicit_key = api_key
        if not api_key and not os.environ.get("ALPHA_VANTAGE_API_KEY"):
            raise ValueError(
                "Alpha Vantage API key required. "
                "Set ALPHA_VANTAGE_API_KEY env var or pass api_key to NewsAgent()."
            )

    def _key_candidates(self):
        """Yield unique keys one at a time — env keys loaded only when needed."""
        seen = set()
        for k in [
            self._explicit_key,
            os.environ.get("ALPHA_VANTAGE_API_KEY"),
            os.environ.get("ALPHA_VANTAGE_API_KEY2"),
            os.environ.get("ALPHA_VANTAGE_API_KEY3"),
        ]:
            if k and k not in seen:
                seen.add(k)
                yield k

    def _fetch_with_key(self, api_key: str, params: dict) -> dict:
        params = {**params, "apikey": api_key}
        resp = requests.get(_AV_BASE, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if "feed" not in raw:
            raise RuntimeError(raw.get("Note") or raw.get("Information") or str(raw))
        return raw

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        av_time_from = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%dT0000")
        av_time_to   = datetime.strptime(end_date,   "%Y-%m-%d").strftime("%Y%m%dT2359")

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": av_time_from,
            "time_to": av_time_to,
            "limit": 25,
            "sort": "RELEVANCE",
        }

        raw = None
        last_err = None
        for i, key in enumerate(self._key_candidates(), 1):
            try:
                logger.info(f"News fetch attempt {i} for {ticker} (key ...{key[-4:]})")
                raw = self._fetch_with_key(key, params)
                logger.info(f"News fetch succeeded with key {i} for {ticker}")
                break
            except Exception as e:
                logger.warning(f"News key {i} failed for {ticker}: {e}")
                last_err = e
                time.sleep(2)

        if raw is None:
            raise RuntimeError(f"All Alpha Vantage keys exhausted. Last error: {last_err}")

        articles = []
        for item in raw["feed"]:
            ticker_sentiment = next(
                (t for t in item.get("ticker_sentiment", []) if t["ticker"] == ticker),
                {}
            )
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "published_at": item.get("time_published", ""),
                "sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score", 0.0)),
                "sentiment_label": ticker_sentiment.get("ticker_sentiment_label", "Neutral"),
                "source": item.get("source", ""),
            })

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "source": "alpha_vantage",
            "data": articles,
        }
