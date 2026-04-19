"""
Stock price and volume data fetching agent using yfinance (no API key needed).
"""

import yfinance as yf
from agents.base_agent import BaseAgent


class PriceAgent(BaseAgent):
    """
    Fetches OHLCV price data and basic financials for a ticker via yfinance.

    Output data includes daily OHLCV rows plus a summary of key financial
    indicators (PE ratio, market cap, 52-week range) to match FinGPT prompt fields.
    """

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        stock = yf.Ticker(ticker)

        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError(f"No price data found for {ticker} between {start_date} and {end_date}")

        price_rows = [
            {
                "date": str(date.date()),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            }
            for date, row in hist.iterrows()
        ]

        info = stock.info
        financials = {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "company_name": info.get("longName", ticker),
            "description": info.get("longBusinessSummary", ""),
        }

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "source": "yfinance",
            "data": {
                "prices": price_rows,
                "financials": financials,
            },
        }
