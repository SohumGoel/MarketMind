"""
SEC filing fetching agent using SEC EDGAR API (no API key needed).

Checks sec_cache/{ticker}.html first before making any network requests.
Uses edgartools to extract MD&A and Risk Factors sections only.
"""

import time
import logging
from pathlib import Path
import requests

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_EDGAR_FILING = "https://www.sec.gov/Archives/edgar/data"
_HEADERS = {"User-Agent": "MarketMind research project contact@cmu.edu"}
_SEC_CACHE_DIR = Path(__file__).parent.parent.parent / "sec_cache"

logger = logging.getLogger(__name__)


def _ticker_to_cik(ticker: str) -> str:
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Could not find CIK for ticker: {ticker}")


def _extract_sections(html_text: str, form_type: str) -> str:
    from edgar.documents import parse_html
    doc = parse_html(html_text)
    available = doc.get_available_sec_sections()
    sections = []

    is_10k = "10-K" in form_type
    risk_key = "part_i_item_1a" if is_10k else "part_ii_item_1a"
    mda_key  = "part_ii_item_7" if is_10k else "part_i_item_2"

    if risk_key in available:
        risk = doc.get_sec_section(risk_key)
        if risk:
            sections.append("=== RISK FACTORS (Item 1A) ===\n" + str(risk)[:15_000])

    if mda_key in available:
        mda = doc.get_sec_section(mda_key)
        if mda:
            sections.append("=== MANAGEMENT DISCUSSION & ANALYSIS ===\n" + str(mda)[:15_000])

    return "\n\n".join(sections) if sections else ""


class SECAgent:
    """
    Fetches SEC filings (10-K, 10-Q) for a given ticker.

    Cache-first: checks sec_cache/{ticker}.html before hitting EDGAR.
    Extracts MD&A + Risk Factors via edgartools for cleaner RAG input.
    """

    def __init__(self, form_types: list[str] = None, max_filings: int = 3):
        self.form_types = form_types or ["10-K", "10-Q"]
        self.max_filings = max_filings

    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict:
        cache_path = _SEC_CACHE_DIR / f"{ticker.upper()}.html"

        if cache_path.exists():
            logger.info(f"SEC cache hit for {ticker} — reading {cache_path}")
            html_text = cache_path.read_text(encoding="utf-8", errors="ignore")
            from edgar.documents import parse_html as _parse_html
            _doc = _parse_html(html_text)
            _avail = _doc.get_available_sec_sections()
            form_type = "10-Q" if "part_ii_item_1a" in _avail and "part_ii_item_7" not in _avail else "10-K"
            full_text = _extract_sections(html_text, form_type)
            return {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "source": "sec_cache",
                "data": [{
                    "form_type": form_type,
                    "filed_at": "cached",
                    "url": str(cache_path),
                    "full_text": full_text,
                }],
            }

        # Live EDGAR fetch (fallback for non-demo tickers)
        logger.info(f"No SEC cache for {ticker} — fetching live from EDGAR")
        cik = _ticker_to_cik(ticker)

        resp = requests.get(
            f"{_EDGAR_SUBMISSIONS}/CIK{cik}.json",
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})

        forms        = recent.get("form", [])
        filed_dates  = recent.get("filingDate", [])
        accessions   = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for form, filed, acc, doc in zip(forms, filed_dates, accessions, primary_docs):
            if form not in self.form_types:
                continue
            if not (start_date <= filed <= end_date):
                continue

            acc_clean = acc.replace("-", "")
            url = f"{_EDGAR_FILING}/{int(cik)}/{acc_clean}/{doc}"
            time.sleep(0.15)
            try:
                doc_resp = requests.get(url, headers=_HEADERS, timeout=20)
                doc_resp.raise_for_status()
                full_text = _extract_sections(doc_resp.text, form)
            except Exception:
                full_text = ""

            filings.append({
                "form_type": form,
                "filed_at": filed,
                "url": url,
                "full_text": full_text,
            })

            if len(filings) >= self.max_filings:
                break

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "source": "sec_edgar",
            "data": filings,
        }
