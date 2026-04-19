"""
MarketMind — Streamlit demo app.

Run: streamlit run app.py

Model modes (sidebar):
  CMU AI Gateway  — calls Claude/GPT-4o via CMU proxy, no GPU needed (best for demo)
  Cached demo     — pre-saved outputs, zero latency, no keys needed
  Local model     — loads finetuned Qwen3-8B checkpoint from disk
  HuggingFace Hub — loads model from a HF repo ID
"""

import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import yfinance as yf

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

st.set_page_config(page_title="MarketMind", page_icon="📈", layout="wide")

from ui.styles import DARK_THEME_CSS
from ui.components import (
    parse_reasoning,
    render_company_header,
    render_price_chart,
    render_key_metrics,
    render_signal_inputs,
    render_prediction_card,
    render_key_drivers,
    render_news_headlines,
    render_full_reasoning,
)

_DEMO_MODE = os.environ.get("MARKETMIND_DEMO") == "1"
CACHED_DIR = Path(__file__).parent / "demo_cache"

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_price_chart(ticker: str):
    return yf.Ticker(ticker).history(period="3mo")


@st.cache_resource(show_spinner=False)
def load_local_model(model_path: str):
    from agents.synthesis.evaluator_agent import EvaluatorAgent
    return EvaluatorAgent(backend="local", model_path=model_path)


def ticker_from_name(query: str) -> str:
    try:
        quotes = yf.Search(query, max_results=1).quotes
        return quotes[0].get("symbol", query.upper()) if quotes else query.upper()
    except Exception:
        return query.upper()


def load_cached(ticker: str) -> dict | None:
    path = CACHED_DIR / f"{ticker.upper()}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_mode = st.selectbox(
        "Model mode",
        ["CMU AI Gateway", "Cached demo", "Local model", "HuggingFace Hub"],
        index=1 if _DEMO_MODE else 0,
    )

    gateway_key  = os.environ.get("CMU_AI_GATEWAY_KEY", "")
    av_key       = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    gateway_model, model_path = "", ""

    if model_mode == "CMU AI Gateway":
        if not gateway_key:
            gateway_key = st.text_input("CMU AI Gateway key", type="password")
        else:
            st.caption("🔑 Gateway key loaded from environment")
        gateway_model = st.selectbox("Model", [
            "claude-sonnet-4-20250514-v1:0",
            "claude-opus-4-20250514-v1:0",
            "gpt-5.4",
            "gemini-2.5-pro",
            "claude-haiku-4-5-20251001-v1:0",
            "gpt-5-mini",
            "gemini-2.5-flash",
        ])
    elif model_mode == "Local model":
        model_path = st.text_input("Checkpoint path", placeholder="/path/to/checkpoint")
    elif model_mode == "HuggingFace Hub":
        model_path = st.text_input("HF repo ID", placeholder="username/marketmind-qwen3-8b")

    if not av_key:
        av_key = st.text_input(
            "Alpha Vantage API key", type="password",
            help="Required for live news. Free key at alphavantage.co (25 req/day).",
        )
    else:
        st.caption("🔑 Alpha Vantage key loaded from environment")

    st.divider()
    end_date   = st.date_input("Week end date", value=date.today() - timedelta(days=1))
    start_date = end_date - timedelta(days=6)
    st.caption(f"Analysis window: {start_date} → {end_date}")
    st.divider()
    if model_mode == "Cached demo":
        st.caption("📦 Demo Cache · no live API calls")
    else:
        st.caption("🟢 Live mode")
    st.caption("MarketMind · CMU 11-766 · Spring 2026")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
    <span style="font-size:1.6rem; font-weight:800; color:#22c55e; letter-spacing:-0.02em;">▲ MarketMind</span>
    <span style="font-size:0.78rem; color:#475569; letter-spacing:0.08em; text-transform:uppercase;">
        AI Market Intelligence
    </span>
</div>
""", unsafe_allow_html=True)

if _DEMO_MODE:
    st.info("📦 Running in cached demo mode — no live API calls")

query = st.text_input(
    "Company or ticker",
    placeholder="Enter company name or ticker — e.g. Apple, MSFT, Tesla",
    label_visibility="collapsed",
)
running = st.session_state.get("running", False)
go_btn = st.button("Analyze →", type="primary", width="stretch", disabled=running)

if go_btn and query.strip():
    _raw = query.strip()
    if len(_raw) > 60:
        st.error("Input too long — enter a company name or ticker symbol.")
        st.stop()
    if re.search(r"[<>{};`]", _raw):
        st.error("Invalid input.")
        st.stop()
    _clean = re.sub(r"[^\w\s\.\-&']", "", _raw).strip()
    if not _clean:
        st.error("Invalid input — enter a company name or ticker symbol.")
        st.stop()
    st.session_state["running"] = True
    st.session_state["pending_query"] = _clean
    st.rerun()

if not st.session_state.get("running") or "pending_query" not in st.session_state:
    st.stop()

_clean = st.session_state["pending_query"]

with st.spinner("Resolving ticker..."):
    ticker = ticker_from_name(_clean)

# ── Company header row ─────────────────────────────────────────────────────────
try:
    _info   = yf.Ticker(ticker).info
    _price  = _info.get("currentPrice") or _info.get("regularMarketPrice")
    _name   = _info.get("longName") or _info.get("shortName") or ticker
    _exch   = _info.get("exchange", "")
    _pchg   = _info.get("regularMarketChangePercent", 0) * 100 if _info.get("regularMarketChangePercent") else None
    _mcap   = _info.get("marketCap", 0)
    _pe     = _info.get("trailingPE")
    _52lo   = _info.get("fiftyTwoWeekLow")
    _52hi   = _info.get("fiftyTwoWeekHigh")
    _sector = _info.get("sector", "")
except Exception:
    _info = {}; _price = None; _name = ticker; _exch = ""; _pchg = None
    _mcap = 0; _pe = None; _52lo = None; _52hi = None; _sector = ""

_price_str = f"${_price:.2f}" if _price else "N/A"
_pchg_str  = f"{_pchg:+.1f}%" if _pchg is not None else ""
_pchg_col  = "#22c55e" if (_pchg or 0) >= 0 else "#ef4444"

render_company_header(_name, ticker, _exch, _price_str, _pchg_str, _pchg_col, end_date)

# ── Price chart + key metrics ──────────────────────────────────────────────────
col_chart, col_metrics = st.columns([3, 1])

with col_chart:
    try:
        hist = fetch_price_chart(ticker)
        render_price_chart(hist)
    except Exception as e:
        st.warning(f"Price chart unavailable: {e}")

with col_metrics:
    render_key_metrics(_price_str, _mcap, _pe, _52lo, _52hi, _sector)

st.divider()

# ── Model inference ────────────────────────────────────────────────────────────

news_items   = []
articles     = []
rag_passages = []
filings      = []

if model_mode == "Cached demo":
    cached = load_cached(ticker)
    if not cached:
        st.warning(
            f"No cached output for **{ticker}**. "
            f"Add `demo_cache/{ticker}.json` or switch to CMU AI Gateway mode."
        )
        st.stop()
    direction  = cached.get("direction", "unknown")
    reasoning  = cached.get("reasoning", "")
    news_items = cached.get("raw", {}).get("news", {}).get("data", [])
    articles   = news_items
    st.session_state["running"] = False

else:
    from datetime import datetime, timedelta as _td
    from agents.data_collection.price_agent import PriceAgent
    from agents.data_collection.news_agent import NewsAgent
    from agents.data_collection.sec_agent import SECAgent
    from agents.retrieval.rag_pipeline import RAGPipeline
    from agents.pipeline import _FINGPT_SYSTEM, _format_price_summary, _format_news_headlines, _format_rag_passages, log_result, _ticker_rag_query
    from agents.synthesis.evaluator_agent import EvaluatorAgent

    if model_mode == "CMU AI Gateway" and not gateway_key:
        st.error("Enter your CMU AI Gateway key in the sidebar.")
        st.stop()
    if model_mode in ("Local model", "HuggingFace Hub") and not model_path:
        st.error("Enter a model path or HF repo ID in the sidebar.")
        st.stop()

    _log = logging.getLogger("marketmind.pipeline")

    with st.status("Running pipeline...", expanded=True) as status:

        def _step(placeholder, msg):
            placeholder.write(msg)
            _log.info(msg.replace("⏳ ", "").replace("✅ ", "").replace("⚠️ ", "WARN: "))

        s1 = st.empty(); _step(s1, "⏳ Fetching price data & financials...")
        try:
            price_result = PriceAgent().fetch(ticker, str(start_date), str(end_date))
            company_name = price_result["data"]["financials"].get("company_name", ticker)
            description  = price_result["data"]["financials"].get("description", "")[:500]
            prices = price_result["data"]["prices"]
            if prices:
                last_close = prices[-1]["close"]
                first_open = prices[0]["open"]
                pct = 100 * (last_close - first_open) / first_open if first_open else 0
                _step(s1, f"✅ Price data fetched — ${last_close:.2f} ({pct:+.1f}% on the week)")
            else:
                _step(s1, "✅ Price data fetched")
        except Exception as e:
            _log.error(f"Price fetch failed: {e}")
            _step(s1, f"⚠️ Price data unavailable: {e}")
            price_result = {"data": {"prices": [], "financials": {}}}
            company_name, description = ticker, ""

        s2 = st.empty(); _step(s2, "⏳ Fetching news headlines...")
        try:
            news_result = NewsAgent(api_key=av_key or None).fetch(ticker, str(start_date), str(end_date))
            articles    = news_result["data"]
            news_items  = articles
            _step(s2, f"✅ News fetched — {len(articles)} headline{'s' if len(articles) != 1 else ''}")
        except Exception as e:
            _log.warning(f"News fetch failed (all keys exhausted): {e}")
            _step(s2, "⚠️ News unavailable — proceeding without headlines")
            articles, news_items = [], []

        s3 = st.empty(); _step(s3, "⏳ Loading SEC filing...")
        try:
            sec_start  = (datetime.strptime(str(start_date), "%Y-%m-%d") - _td(days=90)).strftime("%Y-%m-%d")
            sec_result = SECAgent().fetch(ticker, sec_start, str(end_date))
            filings    = sec_result["data"]
            source     = sec_result.get("source", "")
            if filings:
                form      = filings[0].get("form_type", "filing")
                src_label = "cached" if source == "sec_cache" else "live"
                _step(s3, f"✅ SEC filing loaded — {ticker} {form} ({src_label})")
            else:
                _step(s3, "⚠️ No SEC filings found")
        except Exception as e:
            _log.error(f"SEC fetch failed: {e}")
            _step(s3, f"⚠️ SEC filing unavailable: {e}")
            filings = []; sec_result = {}

        s4 = st.empty(); _step(s4, "⏳ Running RAG extraction...")
        try:
            rag = RAGPipeline()
            for filing in filings:
                if filing.get("full_text"):
                    rag.index_document(filing["full_text"], doc_id=f"{filing['form_type']}_{filing.get('filed_at','?')}")
            rag_passages = rag.retrieve(query=_ticker_rag_query(company_name), top_k=3)
            _step(s4, f"✅ RAG complete — {len(rag_passages)} passage{'s' if len(rag_passages) != 1 else ''} extracted")
        except Exception as e:
            _log.error(f"RAG failed: {e}")
            _step(s4, f"⚠️ RAG failed: {e}")
            rag_passages = []

        s5 = st.empty(); _step(s5, "⏳ Assembling prompt...")
        user_content = f"""[Company Introduction]
{company_name} ({ticker})
{description}

[Basic Financials]
{_format_price_summary(price_result['data'])}

[News Headlines and Summaries]
{_format_news_headlines(articles)}

[Relevant SEC Filing Passages]
{_format_rag_passages(rag_passages)}

Based on the above information, provide your analysis and prediction for {ticker}'s stock price movement for the week following {str(end_date)}."""

        prompt = (
            f"<|im_start|>system\n{_FINGPT_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_dict = {
            "ticker": ticker, "start_date": str(start_date), "end_date": str(end_date),
            "prompt": prompt, "raw": {"price": price_result, "news": {"data": articles}, "sec": sec_result if filings else {}},
            "_rag_chunks": rag_passages, "_news_count": len(articles),
        }
        _step(s5, "✅ Prompt assembled")

        s6 = st.empty(); _step(s6, "⏳ Generating analysis...")
        try:
            if model_mode == "CMU AI Gateway":
                agent = EvaluatorAgent(backend="gateway", api_key=gateway_key, gateway_model=gateway_model)
            else:
                agent = load_local_model(model_path)
            result    = agent.predict(prompt_dict)
            direction = result["direction"]
            reasoning = result["reasoning"]
            _step(s6, "✅ Analysis complete")
        except Exception as e:
            st.session_state["running"] = False
            st.error(f"Inference error: {e}")
            st.stop()

        status.update(label="Pipeline complete ✓", state="complete", expanded=False)

    st.session_state["running"] = False

    try:
        backend_label = (
            "gateway"     if model_mode == "CMU AI Gateway" else
            "local"       if model_mode == "Local model"    else
            "huggingface"
        )
        log_result(prompt_dict, direction, reasoning, backend_label)
    except Exception:
        pass

# ── Signal inputs card ─────────────────────────────────────────────────────────

_form_label = filings[0].get("form_type", "—") if filings else "—"
_rag_label  = "RAG extraction complete ✓" if rag_passages else "No passages retrieved"
_news_label = f"{len(articles)} articles" if articles else "Unavailable"
_news_sub   = (
    f"{sum(1 for a in articles if 'Bullish' in a.get('sentiment_label',''))} bullish · "
    f"{sum(1 for a in articles if 'Bearish' in a.get('sentiment_label',''))} bearish"
) if articles else "—"

render_signal_inputs(_news_label, _news_sub, _form_label, _rag_label, _price_str, _pchg_str)

# ── Prediction card ────────────────────────────────────────────────────────────

sections = parse_reasoning(reasoning)
render_prediction_card(direction, ticker, start_date, end_date, model_mode)

# ── Key drivers + news ────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    render_key_drivers(sections)

with col_right:
    render_news_headlines(news_items)

# ── Full analyst reasoning (always visible) ────────────────────────────────────
render_full_reasoning(sections, reasoning)

# ── Action buttons ─────────────────────────────────────────────────────────────
st.divider()
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("🔄 Refresh analysis", use_container_width=True, help="Re-run the same ticker"):
        st.session_state["running"] = True
        st.rerun()

with btn_col2:
    if st.button("🔍 Analyze new stock", use_container_width=True, help="Clear and start over"):
        st.session_state.pop("pending_query", None)
        st.session_state["running"] = False
        st.rerun()
