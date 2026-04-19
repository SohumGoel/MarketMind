# MarketMind

MarketMind helps retail investors understand weekly stock price movements using a fine-tuned Qwen3-8B model, live data collection (news, price, SEC filings), and RAG over financial documents. A user enters a company name and gets a structured Buy/Hold/Sell recommendation with analyst-style reasoning.

---

## Repo Structure

```
MarketMind/
├── app.py                        # ⭐ Streamlit demo app — run this for the demo
├── demo_cache/                   # Pre-saved model outputs for offline/cached demo mode
├── agents/
│   ├── pipeline.py               # Single entry point: fetches all data → FinGPT prompt
│   ├── base_agent.py
│   ├── data_collection/
│   │   ├── price_agent.py        # yfinance — OHLCV + financials (no API key)
│   │   ├── news_agent.py         # Alpha Vantage news + sentiment
│   │   └── sec_agent.py          # SEC EDGAR 10-K/10-Q/8-K (no API key)
│   ├── retrieval/
│   │   └── rag_pipeline.py       # SentenceBERT + FAISS over SEC documents
│   └── synthesis/
│       └── evaluator_agent.py    # Model inference (gateway / local / HF Hub)
├── configs/
│   ├── base_config.yaml          # All training defaults
│   └── ablations/                # 8 ablation configs, each overrides base
├── data/
│   ├── dataset.py                # FinGPT Dow30 dataset loading + label tokenization
│   └── formatting.py             # Prompt formatting (Llama [INST] → Qwen3 ChatML)
├── training/
│   ├── train.py                  # Entry point: python training/train.py --config ...
│   ├── sft_trainer.py            # Core fine-tuning logic (BnB, LoRA, SFTTrainer)
│   ├── classification_head.py    # ClsHeadSFTTrainer — combined SFT + cls loss
│   └── debug_cls.py              # Debug utilities for cls token validation
├── evaluation/
│   ├── evaluate.py               # Eval script: loads checkpoint, runs on test set
│   └── metrics.py                # Direction extraction, coarse accuracy, F1
├── scripts/
│   ├── run_ablations.sh          # Run all 8 ablations sequentially
│   └── run_eval_all.sh           # Eval all checkpoints, write results_summary.json
├── experiments/
│   ├── rag_ablation.py           # RAG ablations: retrieval k, chunk size, query string
│   ├── summarize_ablation.py     # Parse ablation results → markdown tables
│   ├── model_comparison.py       # Fine-tuned vs zero-shot baseline comparison (for report)
│   └── results/                  # Outputs from ablations and model comparison
├── notebooks/
│   ├── train_colab.ipynb         # ⭐ Self-contained Colab training notebook
│   └── explore_dataset.ipynb     # Optional EDA notebook (CPU, no GPU needed)
├── tests/
│   ├── test_dataset.py
│   └── test_metrics.py
├── outputs/                      # Gitignored — checkpoints written here at runtime
├── logs/                         # Gitignored — run logs written at demo runtime
├── sec_cache/                    # Pre-downloaded SEC filings for 5 demo tickers
└── requirements.txt
```

---

## Demo App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sidebar controls which backend is used for inference:

| Mode | What runs | When to use |
|---|---|---|
| **CMU AI Gateway** | Calls Claude/GPT-4o via CMU proxy — enter your gateway key + pick a model | Live demo, best quality, ~2s latency, no GPU |
| **Cached demo** | Reads `demo_cache/TICKER.json` — no API calls, no pipeline | Fallback if internet is unreliable |
| **Local model** | Loads finetuned Qwen3-8B from a local checkpoint path | After Yash's checkpoint is available |
| **HuggingFace Hub** | Loads model from a HF repo ID | If model is pushed to HF |

In all live modes (Gateway / Local / HF), the full data pipeline runs first: price via yfinance → news via Alpha Vantage → SEC filings via EDGAR (with cache-first check) → adaptive RAG retrieval → model inference.

**SEC caching:** The first time you run a ticker, its 10-K/10-Q is fetched from EDGAR and cached locally. Subsequent runs read from `sec_cache/`. Cached files (5 demo tickers) are committed to the repo for fast demo runs.

**Adaptive RAG query:** If your gateway key is set, the RAG pipeline generates a company/sector/headline-aware retrieval query via `gpt-4o-mini` (instead of using a fixed string). Falls back silently to the fixed query if the call fails or no key is available.

**How the model swap works:** The sidebar dropdown sets `model_mode` in `app.py`, which passes a different `backend` argument to `EvaluatorAgent` in `agents/synthesis/evaluator_agent.py`. The interface is always `agent.predict(prompt_dict)` — only the backend changes. To swap in Yash's finetuned checkpoint, select "Local model" and paste the checkpoint path; no other code changes needed.

**Environment variables** (set in shell or `.env`):
```bash
export CMU_AI_GATEWAY_KEY=your_key
export ALPHA_VANTAGE_API_KEY=your_key
export WANDB_API_KEY=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
export HF_TOKEN=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
```

**Adding cached outputs** for demo fallback — save any model result as `demo_cache/TICKER.json`:
```json
{
  "ticker": "MSFT",
  "direction": "up",
  "reasoning": "[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...",
  "raw": { "news": { "data": [...] } }
}
```

---

## Fine-tuning (Yash — Colab only)

Open `notebooks/train_colab.ipynb` on Colab (A100 recommended). Edit the config cell to pick an ablation, run all cells. Checkpoints save to Google Drive. Once done, push the checkpoint to HF Hub or share the folder so it can be plugged into the demo app.

---

## Training Reference (Yash — GPU cluster only)

All training is done via the Colab notebook. These local scripts exist as reference and for cluster runs — not needed on Mac.

```bash
# Single run
python training/train.py --config configs/ablations/qlora_r32.yaml
python training/train.py --config configs/ablations/qlora_r32.yaml --smoke_test  # 5-step verify

# All 8 ablations sequentially
bash scripts/run_ablations.sh

# Evaluate a checkpoint
python evaluation/evaluate.py --checkpoint outputs/runs/qlora_4bit_r32/checkpoint-best --baseline

# Aggregate all results
bash scripts/run_eval_all.sh  # writes outputs/results_summary.json
```

Ablation matrix: 8 configs in `configs/ablations/` covering 4-bit QLoRA (r16/32/64), 8-bit LoRA (r16/32/64), full fine-tune, and classification head variant. All log to WandB.

---

## Work Division

| Area | Owner | Status |
|---|---|---|
| Fine-tuning training loop + ablations | Yash | Done |
| Evaluation pipeline | Yash | Done |
| DistilBERT downstream classifier | Yash | In progress |
| Model hosting / checkpoint export | Yash | In progress |
| Data collection agents (price, news, SEC) | Sohum | Done |
| RAG pipeline (SentenceBERT + FAISS) | Sohum | Done |
| Streamlit demo app | Sohum | Done |
| Final report | Both | In progress |

---

## Current System State

Here's what's live:

**Full pipeline flow:**
```
User input (ticker + date range)
  → PriceAgent       yfinance — OHLC, P/E, market cap, sector
  → NewsAgent        Alpha Vantage NEWS_SENTIMENT — 25 headlines, sentiment labels
                     3 API keys in .env (KEY, KEY2, KEY3), auto-rotates on rate limit
  → SECAgent         Reads sec_cache/{TICKER}.html first (5 demo tickers pre-cached)
                     Falls back to live EDGAR fetch for other tickers
                     Extracts MD&A + Risk Factors via edgartools (not raw HTML)
  → RAGPipeline      SentenceBERT (all-MiniLM-L6-v2) + FAISS IndexFlatIP
                     Chunks at 512 words / 64 overlap, retrieves top-3 passages
                     Query: "What are the key risks, revenue drivers, and forward outlook for {company}?"
  → Prompt assembly  FinGPT ChatML format with price + news + SEC passages
  → EvaluatorAgent   CMU AI Gateway (primary) → local checkpoint → HF Hub
                     Primary inference model: claude-sonnet-4-20250514-v1:0
                     3 fallback models configured per use case
  → Direction parse  extract_direction_from_output() in evaluation/metrics.py
                     Regex keyword extraction → up / down / neutral
  → UI               Streamlit — two-column layout, BUY/HOLD/SELL card, news sentiment badges
```

**If news fails** (all keys rate-limited): pipeline continues, prompt says "No news articles found", LLM uses price + SEC only — output is still coherent.

**Plugging in your model:** In the app sidebar, select "Local model" and paste your checkpoint path, or "HuggingFace Hub" and paste the repo ID. No code changes needed — `EvaluatorAgent` handles both via the same `predict()` interface.

**Direction extraction:** `evaluation/metrics.py → extract_direction_from_output()`. Currently keyword regex. If you train Model 2 (classifier on reasoning text → 12 magnitude classes), it replaces this function.

**Run logs:** Every inference run appends to `logs/run_log.jsonl` (gitignored). Contains ticker, dates, model backend, RAG chunks, news count, direction, and full LLM output. This is the training data source for Model 2.

**Experiments for report:**
- `python experiments/rag_ablation.py` — RAG design ablations (done, results in `experiments/results/`)
- `python experiments/zero_shot_baseline.py` — full pipeline vs zero-shot LLM comparison
- `python experiments/model_comparison.py --local_checkpoint <path>` — fine-tuned vs zero-shot

---

## Report: Experiments & Evaluation

**Model comparison:** Run `python experiments/model_comparison.py --gateway_only` to generate a side-by-side fine-tuned vs zero-shot baseline table. Pass `--hf_model <repo_id>` or `--local_checkpoint <path>` to include a pre-trained/finetuned model. Outputs formatted markdown + JSON to `experiments/results/`.

**RAG ablations:** Run `python experiments/rag_ablation.py` to measure retrieval k, chunk size, and query string variants. Results in `experiments/results/rag_ablation_results.json`.

---

## Future Work

1. **FinBERT embeddings for RAG** — swap `all-MiniLM-L6-v2` for `yiyanghkust/finbert-tone` in `rag_pipeline.py` for domain-specific financial embeddings.
2. **Finetuned DistilBERT classifier (Model 2)** — replace keyword-based direction extraction with a 12-class classifier (up/down × 6 magnitude buckets) trained on Qwen3-8B reasoning outputs using accumulated run-log data.
3. **"Teach Me" educational module** — NER on reasoning text to surface financial terms, then fetch plain-English definitions via LLM and relevant YouTube explainers via MCP.
4. **Persistent vector index** — replace in-memory FAISS with ChromaDB or PGVector for multi-ticker production use with incremental updates.
5. **Follow-up chat** — store analysis context in session state, allow 2-3 follow-up questions per ticker.