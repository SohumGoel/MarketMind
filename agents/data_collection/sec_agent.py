"""
SEC filing fetching agent (10-K, 10-Q, 8-K).

TODO (Sohum): Implement using SEC EDGAR full-text search API.

Reference:
    SEC EDGAR: https://efts.sec.gov/LATEST/search-index?q=%22AAPL%22&dateRange=custom
    python-edgar: https://github.com/sec-edgar/sec-edgar
"""

from agents.base_agent import BaseAgent


class SECAgent(BaseAgent):
    """
    Fetches recent SEC filings for a given ticker.

    Fetches 10-K (annual), 10-Q (quarterly), and 8-K (material events).
    Raw filing text is later passed to the RAG pipeline for chunking/retrieval.

    Output format:
        {
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "source": "sec_edgar",
            "data": [
                {
                    "form_type": "8-K",
                    "filed_at": "2024-01-03",
                    "url": "...",
                    "full_text": "...",
                },
                ...
            ]
        }
    """

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        raise NotImplementedError(
            "SECAgent not yet implemented. "
            "TODO (Sohum): Implement using SEC EDGAR full-text search API."
        )
