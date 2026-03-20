"""
Stock price and volume data fetching agent.

TODO (Sohum): Implement using yfinance (free) or MarketStack / Alpha Vantage
for OHLCV data.

Reference:
    yfinance: https://github.com/ranaroussi/yfinance
    MarketStack: https://marketstack.com/documentation
    Alpha Vantage TIME_SERIES: https://www.alphavantage.co/documentation/#time-series-data
"""

from agents.base_agent import BaseAgent


class PriceAgent(BaseAgent):
    """
    Fetches OHLCV (open, high, low, close, volume) price data for a ticker.

    Output format:
        {
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "source": "yfinance",
            "data": [
                {
                    "date": "2024-01-02",
                    "open": 185.0,
                    "high": 187.5,
                    "low": 184.0,
                    "close": 186.0,
                    "volume": 50000000,
                },
                ...
            ]
        }
    """

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        raise NotImplementedError(
            "PriceAgent not yet implemented. "
            "TODO (Sohum): Implement using yfinance or MarketStack API."
        )
