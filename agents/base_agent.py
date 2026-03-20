"""
Abstract base class for all MarketMind data collection agents.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    All data collection agents implement this interface.
    Each agent is responsible for fetching data for a single ticker
    over a given time window and returning a structured evidence dict.
    """

    @abstractmethod
    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        """
        Fetch data for a given ticker and time window.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: ISO date string 'YYYY-MM-DD'
            end_date: ISO date string 'YYYY-MM-DD'

        Returns:
            dict with at minimum:
                {
                    "ticker": str,
                    "start_date": str,
                    "end_date": str,
                    "source": str,
                    "data": Any,
                }
        """
        ...
