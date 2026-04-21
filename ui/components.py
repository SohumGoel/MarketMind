import re
import plotly.graph_objects as go
import streamlit as st

DIRECTION_COLOR = {"up": "#22c55e", "down": "#ef4444", "neutral": "#f59e0b", "unknown": "#94a3b8"}
DIRECTION_LABEL = {"up": "BUY",     "down": "SELL",    "neutral": "HOLD",    "unknown": "UNKNOWN"}
DIRECTION_ARROW = {"up": "▲",       "down": "▼",        "neutral": "◆",       "unknown": "—"}

SENTIMENT_COLOR = {
    "Bullish":          "#22c55e",
    "Somewhat-Bullish": "#86efac",
    "Bearish":          "#ef4444",
    "Somewhat-Bearish": "#fca5a5",
    "Neutral":          "#f59e0b",
}


def parse_reasoning(text: str) -> dict:
    pattern = r"\[Positive Developments\](.*?)\[Potential Concerns\](.*?)\[Prediction & Analysis\](.*)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return {
            "positives": m.group(1).strip(),
            "concerns":  m.group(2).strip(),
            "analysis":  m.group(3).strip(),
            "raw": text,
        }
    return {"positives": "", "concerns": "", "analysis": "", "raw": text}


def _bullet_lines(text: str) -> list[str]:
    return [l.strip().lstrip("0123456789.-• ") for l in text.splitlines() if l.strip()]


def render_company_header(name, ticker, exchange, price_str, pchg_str, pchg_col, end_date):
    st.markdown(f"""
<div class="mm-card" style="margin-bottom:16px;">
    <div style="display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;">
        <span style="font-size:1.5rem; font-weight:700; color:#e2e8f0;">{name}</span>
        <span style="font-size:0.85rem; color:#475569;">{ticker} · {exchange}</span>
    </div>
    <div style="display:flex; align-items:baseline; gap:14px; margin-top:8px;">
        <span style="font-size:2rem; font-weight:700; color:#22c55e;">{price_str}</span>
        <span style="font-size:0.95rem; color:{pchg_col}; font-weight:600;">{pchg_str}</span>
        <span style="font-size:0.78rem; color:#475569;">Analysis Date: {end_date} · EOD</span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_price_chart(hist):
    fig = go.Figure(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444", name="Price",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=260,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#64748b", rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#64748b"),
        font=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_key_metrics(price_str, mcap, pe, lo52, hi52, sector):
    st.markdown('<div class="mm-card-label">Key Metrics</div>', unsafe_allow_html=True)

    def _row(label, val):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:6px 0;
                    border-bottom:1px solid #1e293b;">
            <span style="font-size:0.78rem; color:#64748b;">{label}</span>
            <span style="font-size:0.82rem; font-weight:600; color:#e2e8f0;">{val}</span>
        </div>""", unsafe_allow_html=True)

    _row("Price",      price_str)
    _row("Market Cap", f"${mcap/1e9:.1f}B" if mcap else "N/A")
    _row("P/E Ratio",  f"{pe:.1f}" if pe else "N/A")
    _row("52W Range",  f"${lo52} – ${hi52}" if lo52 else "N/A")
    _row("Sector",     sector or "N/A")


def render_signal_inputs(news_label, news_sub, form_label, rag_label, price_str, pchg_str):
    st.markdown(f"""
<div class="mm-card-label" style="margin-bottom:8px;">SIGNAL INPUTS RETRIEVED</div>
<div class="signal-grid">
    <div class="signal-cell">
        <div class="s-label">News</div>
        <div class="s-value">{news_label}</div>
        <div class="s-sub">{news_sub}</div>
    </div>
    <div class="signal-cell">
        <div class="s-label">Filings</div>
        <div class="s-value">{form_label}</div>
        <div class="s-sub">{rag_label}</div>
    </div>
    <div class="signal-cell">
        <div class="s-label">Price Action</div>
        <div class="s-value">{price_str}</div>
        <div class="s-sub">{pchg_str} on the week</div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_prediction_card(direction, ticker, start_date, end_date, model_mode):
    color = DIRECTION_COLOR[direction]
    label = DIRECTION_LABEL[direction]
    arrow = DIRECTION_ARROW[direction]
    st.markdown(f"""
<div class="pred-card" style="background:{color}14; border:2px solid {color};">
    <div>
        <div class="pred-label" style="color:{color};">{arrow} {label}</div>
        <div class="pred-meta">{ticker} · {start_date} → {end_date} &nbsp;·&nbsp; {model_mode}</div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_key_drivers(sections):
    st.markdown('<div class="mm-card-label">KEY DRIVERS</div>', unsafe_allow_html=True)
    if sections["positives"] or sections["concerns"]:
        pos_lines = [l for l in _bullet_lines(sections["positives"]) if l and l != ":"]
        con_lines = [l for l in _bullet_lines(sections["concerns"])  if l and l != ":"]

        for line in pos_lines[:3]:
            st.markdown(f"""
            <div class="driver-card" style="border-color:#22c55e;">
                <div class="driver-title" style="color:#22c55e;">POSITIVE</div>
                <div class="driver-body">{line}</div>
            </div>""", unsafe_allow_html=True)

        for line in con_lines[:2]:
            st.markdown(f"""
            <div class="driver-card" style="border-color:#f59e0b;">
                <div class="driver-title" style="color:#f59e0b;">CONCERN</div>
                <div class="driver-body">{line}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(sections["raw"])


def render_news_headlines(news_items):
    if not news_items:
        return
    st.markdown('<div class="mm-card-label">NEWS HEADLINES</div>', unsafe_allow_html=True)
    for article in news_items[:6]:
        sc    = SENTIMENT_COLOR.get(article.get("sentiment_label", "Neutral"), "#94a3b8")
        title = article.get("title", "")
        summ  = article.get("summary", "")[:160]
        st.markdown(f"""
        <div class="news-item">
            <div class="news-title">
                <span style="color:{sc}; margin-right:6px;">●</span>{title}
            </div>
            <div class="news-summary">{summ}</div>
        </div>""", unsafe_allow_html=True)


def render_full_reasoning(sections, reasoning):
    # If markers weren't found, fall back to simple rendering
    if not (sections["positives"] or sections["concerns"] or sections["analysis"]):
        st.markdown('<div class="mm-card-label" style="margin-top:20px;">FULL ANALYST REASONING</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mm-card"><div style="color:#cbd5e1; line-height:1.8;">{reasoning.replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)
        return

    _HEADERS = {
        "[Positive Developments]": ("POSITIVE DEVELOPMENTS", "#22c55e"),
        "[Potential Concerns]":    ("POTENTIAL CONCERNS",    "#f59e0b"),
        "[Prediction & Analysis]": ("PREDICTION & ANALYSIS", "#60a5fa"),
    }

    lines = reasoning.splitlines()
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("", ":"):
            html_lines.append("<br>")
            continue
        matched = False
        for marker, (label, color) in _HEADERS.items():
            if marker in stripped:
                html_lines.append(
                    f'<div style="margin-top:18px; margin-bottom:6px; '
                    f'font-size:0.95rem; letter-spacing:0.10em; font-weight:700; '
                    f'color:{color}; text-transform:uppercase; '
                    f'border-left:3px solid {color}; padding-left:8px;">{label}</div>'
                )
                matched = True
                break
        if not matched:
            formatted = re.sub(r'\*\*(up|buy)\*\*',           r'<strong style="color:#22c55e;">\1</strong>',  stripped, flags=re.IGNORECASE)
            formatted = re.sub(r'\*\*(down|sell)\*\*',        r'<strong style="color:#ef4444;">\1</strong>',  formatted, flags=re.IGNORECASE)
            formatted = re.sub(r'\*\*(neutral|hold)\*\*',     r'<strong style="color:#f59e0b;">\1</strong>',  formatted, flags=re.IGNORECASE)
            formatted = re.sub(r'Prediction:\s*(up|buy)',      r'Prediction: <strong style="color:#22c55e;">\1</strong>', formatted, flags=re.IGNORECASE)
            formatted = re.sub(r'Prediction:\s*(down|sell)',   r'Prediction: <strong style="color:#ef4444;">\1</strong>', formatted, flags=re.IGNORECASE)
            formatted = re.sub(r'Prediction:\s*(neutral|hold)',r'Prediction: <strong style="color:#f59e0b;">\1</strong>', formatted, flags=re.IGNORECASE)
            formatted = re.sub(r'\*\*(.+?)\*\*',              r'<strong style="color:#e2e8f0;">\1</strong>',  formatted)
            html_lines.append(f'<div style="color:#cbd5e1; line-height:1.75; margin:2px 0;">{formatted}</div>')

    st.markdown('<div class="mm-card-label" style="margin-top:20px;">FULL ANALYST REASONING</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-card">{"".join(html_lines)}</div>', unsafe_allow_html=True)
