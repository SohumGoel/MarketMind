"""
News headline fetching agent.

TODO (Sohum): Implement using Alpha Vantage NEWS_SENTIMENT endpoint
or Finnhub news API.

Requires: ALPHA_VANTAGE_API_KEY in .env (or environment variable)

Reference:
    Alpha Vantage: https://www.alphavantage.co/documentation/#news-sentiment
    Finnhub: https://finnhub.io/docs/api/company-news
"""

from agents.base_agent import BaseAgent


class NewsAgent(BaseAgent):
    """
    Fetches recent news headlines and summaries for a given ticker.

    Output format:
        {
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "source": "alpha_vantage",
            "data": [
                {
                    "title": "...",
                    "summary": "...",
                    "url": "...",
                    "published_at": "...",
                    "sentiment_score": 0.0,
                },
                ...
            ]
        }
    """

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        raise NotImplementedError(
            "NewsAgent not yet implemented. "
            "TODO (Sohum): Implement using Alpha Vantage / Finnhub API."
        )
