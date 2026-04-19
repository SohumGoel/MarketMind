# RAG Pipeline Ablation Study (Report Section)

## 4. RAG Pipeline Ablation Study

We evaluate three design choices in our Retrieval-Augmented Generation (RAG) pipeline over SEC 10-K/10-Q filings for four tickers (AAPL, JNJ, JPM, WMT). All experiments use SentenceBERT (`all-MiniLM-L6-v2`) with FAISS IndexFlatIP. The metric is cosine similarity between the query embedding and the top-1 retrieved chunk.

### 4.1 Retrieval k

**Table 1: Effect of retrieval k on top-1 cosine similarity**

| k | Avg Top-1 Score |
|---|---|
| k = 1 | 0.5028 |
| k = 3 (default) | 0.5028 |
| k = 5 | 0.5028 |

Top-1 similarity is stable across all values of k, confirming that increasing k adds passage coverage without degrading the quality of the best-matched chunk. We use k = 3 in the final system as a balance between context richness and prompt length.

### 4.2 Chunk Size

**Table 2: Effect of chunk size on top-1 cosine similarity**

| Chunk Size / Overlap | Avg Top-1 Score |
|---|---|
| 256 / 32 | 0.5058 |
| 512 / 64 (default) | 0.5028 |

Smaller chunks (256 words) yield marginally higher top-1 similarity (+0.003). However, 512-word chunks provide more contiguous financial context per passage, which is preferable for downstream LLM reasoning over dense SEC prose. We retain 512/64 as the default.

### 4.3 Query Strategy

**Table 3: Effect of query strategy on top-1 cosine similarity**

| Query Variant | Avg Top-1 Score | Δ vs Fixed |
|---|---|---|
| Fixed domain query | 0.5028 | — |
| Ticker-specific query | **0.5486** | **+0.046** |
| Adaptive (LLM-generated) | 0.4862 | -0.017 |

**Table 4: Per-ticker breakdown (query strategy)**

| Ticker | Fixed | Ticker-Specific | Adaptive | Best |
|---|---|---|---|---|
| AAPL | 0.5089 | 0.5689 | 0.5607 | Ticker-specific |
| JNJ | 0.4798 | 0.4226 | 0.3144 | Fixed |
| WMT | 0.5201 | 0.5603 | 0.5721 | Adaptive |
| JPM | 0.5024 | 0.6425 | 0.4974 | Ticker-specific |

Ticker-specific natural language queries ("What are the key risks, revenue drivers, and forward outlook for {company}?") outperform the fixed domain query by +0.046 average cosine similarity. Adaptive LLM-generated queries — grounded in recent news headlines — underperform despite producing grammatically clean output, likely because news-specific terminology (e.g., "e-commerce digital transformation", "California environmental pressure") diverges from the stable financial vocabulary used in SEC filings (net sales, operating income, liquidity, risk factors). This suggests that for SEC retrieval, company-scoped but financially grounded language generalises better than current-event specificity. We adopt the ticker-specific query strategy in the final system.

### 4.4 Summary

| Design Choice | Default Setting | Justification |
|---|---|---|
| Retrieval k | 3 | Stable top-1 quality; adds coverage |
| Chunk size | 512 / 64 overlap | Richer financial context per passage |
| Query strategy | Ticker-specific template | +0.046 cosine vs fixed; best average |
