"""
Social media sentiment fetching agent (Reddit, X/Twitter, Yahoo Finance).

TODO (Sohum): Implement using Reddit PRAW API and/or Yahoo Finance scraping.

Reference:
    PRAW (Reddit): https://praw.readthedocs.io/
    Yahoo Finance news: accessible via yfinance ticker.news
"""

from agents.base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    """
    Fetches social media sentiment signals for a given ticker.

    Targets: Reddit (r/stocks, r/investing, r/wallstreetbets),
             Yahoo Finance comment sentiment.

    Output format:
        {
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "source": "reddit",
            "data": [
                {
                    "platform": "reddit",
                    "subreddit": "r/stocks",
                    "text": "...",
                    "score": 150,
                    "created_at": "2024-01-03T12:00:00",
                },
                ...
            ]
        }
    """

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        raise NotImplementedError(
            "SentimentAgent not yet implemented. "
            "TODO (Sohum): Implement using Reddit PRAW API and/or Yahoo Finance."
        )
