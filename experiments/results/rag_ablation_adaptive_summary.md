# RAG Query Ablation — Fixed vs Ticker-Specific vs Adaptive

**Model:** all-MiniLM-L6-v2 (SentenceBERT)  |  **Adaptive query model:** gpt-5-mini
**Date:** 2026-04-19
**Tickers:** AAPL, JNJ, WMT, JPM, XOM

## Average Top-1 Cosine Similarity

| Query Variant | Avg Top-1 Score | Δ vs Fixed |
|---------------|-----------------|------------|
| fixed | 0.5028 | +0.0000 |
| ticker_specific | 0.5486 | +0.0458 |
| adaptive | 0.4862 | -0.0166 |

## Per-Ticker Results

| Ticker | Fixed | Ticker-Specific | Adaptive | Best |
|--------|-------|-----------------|----------|------|
| AAPL | 0.5089 | 0.5689 | 0.5607 | **ticker_specific** |
| JNJ | 0.4798 | 0.4226 | 0.3144 | **fixed** |
| WMT | 0.5201 | 0.5603 | 0.5721 | **adaptive** |
| JPM | 0.5024 | 0.6425 | 0.4974 | **ticker_specific** |

## Adaptive Queries Generated

- **AAPL:** Apple revenue growth product sales services segment performance financial results quarterly earnings
- **JNJ:** Johnson & Johnson financial performance revenue earnings net income operating cash flow debt equity ratio
- **WMT:** Walmart revenue growth comparable sales e-commerce digital transformation supply chain efficiency consumer spending retail operations
- **JPM:** JPMorgan Chase financial performance revenue net income assets liabilities capital ratios risk management

## Key Finding

> Ticker-specific template queries achieve the highest average top-1 cosine similarity (0.5486, +0.046 vs fixed), outperforming both the fixed baseline and adaptive LLM-generated queries. Adaptive queries (0.4862, -0.017 vs fixed) underperform despite clean generation — likely because news-grounded specificity (e.g. "California environmental pressure", "e-commerce digital transformation") diverges from the broad financial terminology used in SEC filings (net sales, operating income, risk factors, liquidity). This suggests SEC retrieval favours stable financial vocabulary over current-event specificity. Ticker-specific templates strike the best balance: company-scoped but financially grounded. The adaptive approach remains useful as a runtime signal for the LLM prompt context, even if it does not improve cosine-based retrieval scores.