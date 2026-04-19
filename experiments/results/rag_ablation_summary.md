# RAG Ablation Results

### Retrieval k

*Tickers: AAPL, JNJ, JPM, WMT · metric: cosine similarity (top-1 retrieved chunk)*

| Variant | Avg top-1 score | Min | Max | n tickers |
| --- | --- | --- | --- | --- |
| k = 1 | 0.5028 | 0.4798 | 0.5201 | 4 |
| k = 3 (default) | 0.5028 | 0.4798 | 0.5201 | 4 |
| k = 5 | 0.5028 | 0.4798 | 0.5201 | 4 |

---

### Chunk size (tokens/overlap)

*Tickers: AAPL, JNJ, JPM, WMT · metric: cosine similarity (top-1 retrieved chunk)*

| Variant | Avg top-1 score | Min | Max | n tickers |
| --- | --- | --- | --- | --- |
| 256 / 32 | 0.5058 | 0.4798 | 0.5250 | 4 |
| 512 / 64 (default) | 0.5028 | 0.4798 | 0.5201 | 4 |

---

### Query strategy

*Tickers: AAPL, JNJ, JPM, WMT · metric: cosine similarity (top-1 retrieved chunk)*

| Variant | Avg top-1 score | Min | Max | n tickers |
| --- | --- | --- | --- | --- |
| Fixed domain query | 0.5028 | 0.4798 | 0.5201 | 4 |
| Ticker-specific query | 0.5486 | 0.4226 | 0.6425 | 4 |

---

### Key Findings

- **Query strategy:** ticker-specific queries outperform fixed query by **+0.0458** avg cosine similarity (0.5486 vs 0.5028).
- **Chunk size:** 256/32 achieves higher avg top-1 score (256/32: 0.5058, 512/64: 0.5028).
- **Retrieval k:** top-1 score is stable across k=1/3/5 (0.5028 → 0.5028); increasing k adds coverage without hurting precision.