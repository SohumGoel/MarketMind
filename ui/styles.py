DARK_THEME_CSS = """
<style>
/* Base dark background */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #0d1117; border-right: 1px solid #1e293b; }
[data-testid="stHeader"]           { background: #0d1117; }
section.main > div                 { background: #0d1117; }

/* Typography */
h1, h2, h3, h4, p, label, div     { color: #e2e8f0 !important; }
.stCaption, small                  { color: #64748b !important; }

/* Input */
input[type="text"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: #22c55e !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] > button:disabled {
    background: #1e293b !important;
    color: #475569 !important;
}

/* Cards */
.mm-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.mm-card-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b !important;
    margin-bottom: 6px;
}

/* Signal row */
.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.signal-cell {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 16px;
}
.signal-cell .s-label {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
}
.signal-cell .s-value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
}
.signal-cell .s-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
}

/* Prediction card */
.pred-card {
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.pred-label {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}
.pred-meta {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Driver cards */
.driver-card {
    border-left: 3px solid;
    background: #161b22;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.driver-title {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 4px;
}
.driver-body {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* News items */
.news-item {
    border-bottom: 1px solid #1e293b;
    padding: 10px 0;
}
.news-title { font-size: 0.88rem; font-weight: 600; color: #e2e8f0; }
.news-summary { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

/* Divider */
hr { border-color: #1e293b !important; }

/* Plotly */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* Status box */
[data-testid="stStatusWidget"] { background: #161b22 !important; border: 1px solid #21262d !important; }
</style>
"""
