import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import io

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


st.set_page_config(
    page_title="Dashboard Peramalan",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
#  INJEKSI CSS — FORMAL RED, WHITE & CREAM PROFESSIONAL THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700;800&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --crimson:      #9B1C1C;
    --crimson-deep: #7A1515;
    --crimson-mid:  #B91C1C;
    --crimson-soft: #DC2626;
    --crimson-pale: #FEE2E2;
    --gold:         #B45309;
    --gold-light:   #D97706;
    --cream:        #FDF8F0;
    --cream-deep:   #F5EDD8;
    --cream-mid:    #EFE3C8;
    --ivory:        #FAFAF7;
    --white:        #FFFFFF;
    --charcoal:     #1C1917;
    --brown:        #44403C;
    --warm-gray:    #78716C;
    --border:       rgba(155,28,28,0.18);
    --border-cream: rgba(180,83,9,0.22);
    --text:         #1C1917;
    --text-mid:     #44403C;
    --text-dim:     #78716C;
    --font:         'Source Sans 3', sans-serif;
    --display:      'Playfair Display', serif;
    --mono:         'JetBrains Mono', monospace;
    --shadow-red:   rgba(155,28,28,0.15);
    --shadow-warm:  rgba(180,83,9,0.12);
}

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── APP BACKGROUND ── */
.stApp {
    background: linear-gradient(160deg, #FDF8F0 0%, #FAF4E8 40%, #F5EDD8 100%) !important;
    font-family: var(--font) !important;
}

/* Subtle damask/linen texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(155,28,28,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(180,83,9,0.04) 0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 5 L55 20 L55 40 L30 55 L5 40 L5 20 Z' fill='none' stroke='rgba(155,28,28,0.025)' stroke-width='0.5'/%3E%3C/svg%3E");
    background-size: cover, cover, 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #7A1515 0%, #9B1C1C 45%, #7A1515 100%) !important;
    border-right: 3px solid rgba(180,83,9,0.4) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FCD34D !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    margin-bottom: 0.6rem !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: rgba(253,248,240,0.85) !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(252,211,77,0.25) !important;
    margin: 1rem 0 !important;
}

/* Sidebar selectbox / inputs */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(122,21,21,0.6) !important;
    border: 1px solid rgba(252,211,77,0.3) !important;
    border-radius: 6px !important;
    color: #FDF8F0 !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(122,21,21,0.6) !important;
    border: 1px solid rgba(252,211,77,0.3) !important;
    border-radius: 6px !important;
    color: #FDF8F0 !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: rgba(253,248,240,0.85) !important;
}

/* Sidebar slider */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: #FCD34D !important;
}

/* ── TYPOGRAPHY (MAIN) ── */
h1 {
    color: var(--charcoal) !important;
    font-family: var(--display) !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    letter-spacing: -0.01em !important;
    line-height: 1.2 !important;
}

h2 {
    color: var(--crimson-deep) !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.01em !important;
    border-bottom: 2px solid var(--border) !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 1rem !important;
}

h3 {
    color: var(--crimson) !important;
    font-family: var(--display) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
}

p, li, .stMarkdown p {
    color: var(--text-mid) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}

label { color: var(--text-dim) !important; font-size: 0.82rem !important; }

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF 0%, #FDF8F0 100%) !important;
    border: 1px solid rgba(155,28,28,0.15) !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px var(--shadow-red) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--crimson-deep), var(--crimson-soft), var(--gold-light));
}

[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 60px; height: 60px;
    border-radius: 50% 0 0 0;
    background: rgba(155,28,28,0.04);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px var(--shadow-red) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--warm-gray) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-family: var(--font) !important;
}

[data-testid="stMetricValue"] {
    color: var(--crimson-deep) !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    font-family: var(--mono) !important;
}

[data-testid="stMetricDelta"] { color: var(--gold) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--crimson-deep) 0%, var(--crimson-mid) 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    font-family: var(--font) !important;
    letter-spacing: 0.06em !important;
    border: 1px solid rgba(180,83,9,0.3) !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px var(--shadow-red) !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--crimson-mid) 0%, var(--crimson-soft) 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 20px rgba(155,28,28,0.3) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--crimson-deep) 0%, #C41E1E 100%) !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.6rem !important;
    width: 100% !important;
    border: 1px solid rgba(252,211,77,0.4) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-light) 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 3px 12px var(--shadow-warm) !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 20px rgba(180,83,9,0.3) !important;
}

/* ── DATAFRAME / TABLES ── */
[data-testid="stDataFrame"],
div[data-testid="stElementContainer"] > div[data-testid="stDataFrame"] {
    background: var(--white) !important;
    border: 1px solid rgba(155,28,28,0.12) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px var(--shadow-red) !important;
}

.stDataFrame thead tr th {
    background: linear-gradient(135deg, var(--crimson-deep), var(--crimson)) !important;
    color: var(--cream) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    border-bottom: 2px solid rgba(180,83,9,0.5) !important;
    padding: 0.75rem 1rem !important;
    font-family: var(--font) !important;
}

.stDataFrame tbody tr td {
    background: var(--white) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid rgba(155,28,28,0.06) !important;
    padding: 0.55rem 1rem !important;
}

.stDataFrame tbody tr:nth-child(even) td {
    background: #FDF8F0 !important;
}

.stDataFrame tbody tr:hover td {
    background: #FEE2E2 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--white) !important;
    border-radius: 8px !important;
    padding: 0.3rem !important;
    gap: 0.2rem !important;
    border: 1px solid rgba(155,28,28,0.15) !important;
    box-shadow: 0 2px 8px var(--shadow-red) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    font-family: var(--font) !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--crimson-deep), var(--crimson-mid)) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(155,28,28,0.35) !important;
}

/* ── SELECT / RADIO / SLIDER (MAIN) ── */
.stSelectbox > div > div {
    background: var(--white) !important;
    border: 1px solid rgba(155,28,28,0.2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-size: 0.85rem !important;
}

.stRadio > div {
    gap: 0.5rem !important;
}

.stRadio label {
    color: var(--text-mid) !important;
    font-size: 0.84rem !important;
}

[data-testid="stSlider"] > div > div > div {
    background: var(--crimson) !important;
}

/* ── INFO / WARNING / SUCCESS BOXES ── */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
    font-size: 0.85rem !important;
}

[data-testid="stInfo"] {
    background: rgba(155,28,28,0.07) !important;
    border-left: 4px solid var(--crimson) !important;
    color: var(--text) !important;
}

[data-testid="stSuccess"],
div[data-testid="stAlert"][data-type="success"] {
    background: rgba(180,83,9,0.08) !important;
    border-left: 4px solid var(--gold) !important;
    color: var(--text) !important;
}

[data-testid="stWarning"] {
    background: rgba(180,83,9,0.08) !important;
    border-left: 4px solid var(--gold-light) !important;
    color: var(--text) !important;
}

/* ── CONTAINERS / BORDERS ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white) !important;
    border: 1px solid rgba(155,28,28,0.12) !important;
    border-radius: 10px !important;
    padding: 1.2rem !important;
    box-shadow: 0 2px 10px var(--shadow-red) !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #FDF8F0 !important;
    border: 1px solid rgba(155,28,28,0.15) !important;
    border-radius: 6px !important;
    color: var(--crimson-deep) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    font-family: var(--font) !important;
}

.streamlit-expanderHeader:hover {
    background: #FEE2E2 !important;
    border-color: rgba(155,28,28,0.3) !important;
}

.streamlit-expanderContent {
    background: var(--ivory) !important;
    border: 1px solid rgba(155,28,28,0.12) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(253,248,240,0.8) !important;
    border: 2px dashed rgba(252,211,77,0.5) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(252,211,77,0.9) !important;
    background: rgba(254,226,226,0.3) !important;
}

/* ── NUMBER INPUT (MAIN) ── */
[data-testid="stNumberInput"] input {
    background: var(--white) !important;
    border: 1px solid rgba(155,28,28,0.2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.88rem !important;
}

[data-testid="stTextInput"] input {
    background: var(--white) !important;
    border: 1px solid rgba(155,28,28,0.2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-size: 0.85rem !important;
}

/* ── MAIN TITLE HEADER BLOCK ── */
.title-block {
    background: linear-gradient(135deg, #7A1515 0%, #9B1C1C 50%, #7A1515 100%);
    border: 1px solid rgba(180,83,9,0.4);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(122,21,21,0.3);
}

.title-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #FCD34D, #F59E0B, #FCD34D, #B45309);
}

.title-block::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(252,211,77,0.07) 0%, transparent 70%);
}

/* Decorative left bar on title block */
.title-block .deco-left {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, #FCD34D, #F59E0B, #B45309);
    border-radius: 14px 0 0 14px;
}

.title-block h1 {
    margin: 0 0 0.4rem 0 !important;
    font-size: 1.85rem !important;
    color: #FFFFFF !important;
    font-family: var(--display) !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.title-block p {
    margin: 0 !important;
    font-size: 0.9rem !important;
    color: rgba(253,248,240,0.82) !important;
}

.title-badge {
    display: inline-block;
    background: rgba(252,211,77,0.18);
    border: 1px solid rgba(252,211,77,0.45);
    color: #FCD34D !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.25rem 0.8rem;
    border-radius: 100px;
    margin-bottom: 0.85rem;
    font-family: var(--font) !important;
}

/* ── SECTION LABELS ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}

.section-label-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--crimson), var(--gold));
    flex-shrink: 0;
}

/* ── PLOTLY CHART WRAPPER ── */
[data-testid="stPlotlyChart"] {
    background: var(--white) !important;
    border: 1px solid rgba(155,28,28,0.12) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px var(--shadow-red) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: rgba(155,28,28,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--crimson); }

/* ── BEST METHOD BADGE ── */
.best-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: linear-gradient(135deg, rgba(155,28,28,0.07), rgba(180,83,9,0.05));
    border: 1px solid rgba(155,28,28,0.25);
    border-left: 4px solid var(--crimson);
    border-radius: 8px;
    padding: 0.85rem 1.4rem;
    margin: 1rem 0;
    font-family: var(--font);
}

.best-badge span {
    color: var(--crimson-deep) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}

/* ── DIVIDER ── */
hr {
    border-color: rgba(155,28,28,0.12) !important;
}

/* ── STALE CAPTION / SMALL TEXT ── */
.stCaption, small {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
}

/* ── COLUMN HEADERS IN MAIN ── */
[data-testid="column"] h3 {
    color: var(--crimson) !important;
    font-family: var(--display) !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FUNGSI PROSES DAN PERHITUNGAN DASAR  (tidak diubah)
# ─────────────────────────────────────────────────────────────────────────────

def clean_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().astype(float)


def safe_mape(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def calculate_error_table(periods, actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    error = actual - forecast
    abs_error = np.abs(error)
    squared_error = error ** 2
    ape = np.where(actual != 0, np.abs(error / actual) * 100, np.nan)
    result_df = pd.DataFrame({
        "Periode": periods, "Aktual": actual, "Forecast": forecast,
        "Error": error, "Absolute Error": abs_error,
        "Squared Error": squared_error, "APE (%)": ape
    })
    metrics = {"MAD": np.mean(abs_error), "MSE": np.mean(squared_error), "MAPE": safe_mape(actual, forecast)}
    return result_df, metrics


def split_train_test(values: np.ndarray, test_percentage: int):
    n = len(values)
    test_size = max(1, int(round(n * test_percentage / 100)))
    test_size = min(test_size, n - 2)
    train = values[:-test_size]
    test = values[-test_size:]
    return train, test, test_size


def parse_weights(weight_text: str):
    try:
        weights = [float(x.strip()) for x in weight_text.split(",") if x.strip() != ""]
        weights = [w for w in weights if w > 0]
        if len(weights) == 0:
            return [0.2, 0.3, 0.5]
        total = sum(weights)
        return [w / total for w in weights]
    except Exception:
        return [0.2, 0.3, 0.5]


def make_period_labels(df: pd.DataFrame, period_col):
    if period_col is None:
        return [f"Periode {i}" for i in range(1, len(df) + 1)], None
    raw_period = df[period_col]
    parsed = pd.to_datetime(raw_period, errors="coerce")
    valid_ratio = parsed.notna().mean()
    if valid_ratio >= 0.7:
        return parsed.dt.strftime("%Y-%m-%d").fillna(raw_period.astype(str)).tolist(), parsed
    return raw_period.astype(str).tolist(), None


def make_future_labels(period_dates, existing_labels, horizon: int):
    if period_dates is not None and period_dates.notna().sum() >= 2:
        valid_dates = period_dates.dropna().reset_index(drop=True)
        try:
            inferred_freq = pd.infer_freq(valid_dates)
        except Exception:
            inferred_freq = None
        last_date = valid_dates.iloc[-1]
        if inferred_freq is not None:
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=inferred_freq)[1:]
            return future_dates.strftime("%Y-%m-%d").tolist()
        delta = valid_dates.iloc[-1] - valid_dates.iloc[-2]
        future_dates = [last_date + (i * delta) for i in range(1, horizon + 1)]
        return [d.strftime("%Y-%m-%d") for d in future_dates]
    return [f"Periode {len(existing_labels) + i}" for i in range(1, horizon + 1)]


# ─────────────────────────────────────────────────────────────────────────────
#  FUNGSI GRAFIK PLOTLY — Formal Red & Cream Theme
# ─────────────────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(253,248,240,0.4)",
    font=dict(family="Source Sans 3, sans-serif", color="#78716C", size=11),
    xaxis=dict(
        gridcolor="rgba(155,28,28,0.07)",
        linecolor="rgba(155,28,28,0.15)",
        tickfont=dict(color="#78716C", size=10),
        title_font=dict(color="#44403C"),
        zerolinecolor="rgba(155,28,28,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(155,28,28,0.07)",
        linecolor="rgba(155,28,28,0.15)",
        tickfont=dict(color="#78716C", size=10),
        title_font=dict(color="#44403C"),
        zerolinecolor="rgba(155,28,28,0.1)",
    ),
    legend=dict(
        bgcolor="rgba(253,248,240,0.92)",
        bordercolor="rgba(155,28,28,0.2)",
        borderwidth=1,
        font=dict(color="#1C1917", size=11),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(122,21,21,0.92)",
        bordercolor="rgba(180,83,9,0.5)",
        font=dict(color="#FDF8F0", size=11, family="JetBrains Mono, monospace"),
    ),
    margin=dict(l=20, r=20, t=50, b=30),
)


def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=actual, mode="lines+markers", name="Aktual",
        line=dict(color="#9B1C1C", width=2.5),
        marker=dict(size=6, color="#9B1C1C", line=dict(color="#FFFFFF", width=1.2)),
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=forecast, mode="lines+markers", name="Forecast",
        line=dict(color="#B45309", width=2.5),
        marker=dict(size=6, color="#B45309", symbol="diamond", line=dict(color="#FFFFFF", width=1.2)),
    ))
    layout = dict(PLOT_LAYOUT)
    layout["title"] = dict(text=title, font=dict(color="#1C1917", size=14, family="Playfair Display, serif"), x=0.02)
    layout["xaxis"] = dict(PLOT_LAYOUT["xaxis"], title="Periode")
    layout["yaxis"] = dict(PLOT_LAYOUT["yaxis"], title="Nilai")
    fig.update_layout(**layout)
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_periods, y=actual_values, mode="lines+markers", name="Data Historis",
        line=dict(color="#9B1C1C", width=2.5),
        marker=dict(size=5, color="#9B1C1C", line=dict(color="#FFFFFF", width=1)),
    ))
    if residual_std > 0:
        upper_bound = future_forecast + (1.96 * residual_std)
        lower_bound = np.clip(future_forecast - (1.96 * residual_std), 0, None)
        fig.add_trace(go.Scatter(
            x=future_periods + future_periods[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill="toself",
            fillcolor="rgba(180,83,9,0.10)",
            line=dict(color="rgba(180,83,9,0.25)", width=1),
            name="Interval Keyakinan 95%",
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=future_periods, y=future_forecast, mode="lines+markers", name="Proyeksi",
        line=dict(color="#B45309", width=2.5, dash="dash"),
        marker=dict(size=7, color="#D97706", symbol="star", line=dict(color="#FFFFFF", width=1.5)),
    ))
    layout = dict(PLOT_LAYOUT)
    layout["title"] = dict(text="Grafik Proyeksi Nilai Masa Depan", font=dict(color="#1C1917", size=14, family="Playfair Display, serif"), x=0.02)
    layout["xaxis"] = dict(PLOT_LAYOUT["xaxis"], title="Periode")
    layout["yaxis"] = dict(PLOT_LAYOUT["yaxis"], title="Nilai")
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITMA METODE PERAMALAN  (tidak diubah)
# ─────────────────────────────────────────────────────────────────────────────

def forecast_naive(history, horizon, **kwargs):
    if len(history) == 0: return np.zeros(horizon)
    return np.repeat(history[-1], horizon)

def forecast_moving_average(history, horizon, window=3, **kwargs):
    history_list = list(history)
    forecasts = []
    for _ in range(horizon):
        usable_window = min(window, len(history_list))
        pred = np.mean(history_list[-usable_window:])
        forecasts.append(pred)
        history_list.append(pred)
    return np.array(forecasts)

def forecast_weighted_moving_average(history, horizon, weights=None, **kwargs):
    if weights is None: weights = [0.2, 0.3, 0.5]
    history_list = list(history)
    forecasts = []
    for _ in range(horizon):
        usable_window = min(len(weights), len(history_list))
        recent_values = np.array(history_list[-usable_window:], dtype=float)
        recent_weights = np.array(weights[-usable_window:], dtype=float)
        recent_weights = recent_weights / recent_weights.sum()
        pred = np.sum(recent_values * recent_weights)
        forecasts.append(pred)
        history_list.append(pred)
    return np.array(forecasts)

def get_fitted_param(fitted, keys):
    for key in keys:
        value = fitted.params.get(key, None)
        if value is not None: return value
    return None

def format_param(value):
    if value is None or pd.isna(value): return "-"
    try: return f"{float(value):.4f}"
    except Exception: return "-"

def limit_smoothing_param(value, minimum=0.01, maximum=0.99):
    try:
        if value is None or pd.isna(value): return minimum
        value = float(value)
        if value < minimum: return minimum
        value = maximum if value > maximum else value
        return value
    except Exception: return minimum

def forecast_single_exponential_smoothing(history, horizon, optimized=True, alpha=None, **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 3:
        return np.array(forecast_naive(history, horizon)), {}
    try:
        model = SimpleExpSmoothing(history, initialization_method="estimated")
        if optimized:
            fitted_auto = model.fit(optimized=True)
            alpha_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_level"]))
            fitted = model.fit(smoothing_level=alpha_used, optimized=False)
        else:
            alpha_used = limit_smoothing_param(alpha)
            fitted = model.fit(smoothing_level=alpha_used, optimized=False)
        return np.array(fitted.forecast(horizon)), {"Alpha": alpha_used, "Beta": None, "Gamma": None}
    except Exception:
        return np.array(forecast_naive(history, horizon)), {}

def forecast_double_exponential_smoothing(history, horizon, optimized=True, alpha=None, beta=None, **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 4:
        return np.array(forecast_naive(history, horizon)), {}
    try:
        model = ExponentialSmoothing(history, trend="add", seasonal=None, initialization_method="estimated")
        if optimized:
            fitted_auto = model.fit(optimized=True)
            alpha_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_level"]))
            beta_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_trend", "smoothing_slope"]))
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, optimized=False)
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, optimized=False)
        return np.array(fitted.forecast(horizon)), {"Alpha": alpha_used, "Beta": beta_used, "Gamma": None}
    except Exception:
        return np.array(forecast_naive(history, horizon)), {}

def forecast_triple_exponential_smoothing(history, horizon, seasonal_periods=12, optimized=True, alpha=None, beta=None, gamma=None, **kwargs):
    min_data = max(2 * seasonal_periods, seasonal_periods + 4)
    if not STATSMODELS_AVAILABLE or len(history) < min_data:
        return forecast_double_exponential_smoothing(history, horizon, optimized=optimized, alpha=alpha, beta=beta)
    try:
        model = ExponentialSmoothing(history, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated")
        if optimized:
            fitted_auto = model.fit(optimized=True)
            alpha_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_level"]))
            beta_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_trend", "smoothing_slope"]))
            gamma_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_seasonal"]))
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, smoothing_seasonal=gamma_used, optimized=False)
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
            gamma_used = limit_smoothing_param(gamma)
            fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, smoothing_seasonal=gamma_used, optimized=False)
        return np.array(fitted.forecast(horizon)), {"Alpha": alpha_used, "Beta": beta_used, "Gamma": gamma_used}
    except Exception:
        return forecast_double_exponential_smoothing(history, horizon, optimized=optimized, alpha=alpha, beta=beta)

def forecast_linear_trend(history, horizon, **kwargs):
    if len(history) < 2: return forecast_naive(history, horizon)
    x = np.arange(1, len(history) + 1)
    y = np.array(history, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(history) + 1, len(history) + horizon + 1)
    return np.array(intercept + slope * future_x)

def forecast_least_square_quadratic(history, horizon, **kwargs):
    if len(history) < 3: return forecast_linear_trend(history, horizon)
    x = np.arange(1, len(history) + 1)
    y = np.array(history, dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    future_x = np.arange(len(history) + 1, len(history) + horizon + 1)
    return np.array(a * (future_x ** 2) + b * future_x + c)

def forecast_seasonal_naive(history, horizon, seasonal_periods=12, **kwargs):
    if len(history) < seasonal_periods: return forecast_naive(history, horizon)
    history_list = list(history)
    forecasts = []
    for _ in range(horizon):
        pred = history_list[-seasonal_periods]
        forecasts.append(pred)
        history_list.append(pred)
    return np.array(forecasts)

def forecast_arima(history, horizon, arima_order=(1, 1, 1), **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 8: return forecast_naive(history, horizon)
    try:
        model = ARIMA(history, order=arima_order)
        fitted_model = model.fit()
        return np.array(fitted_model.forecast(steps=horizon))
    except Exception:
        return forecast_naive(history, horizon)


FORECAST_METHODS = {
    "Naive Forecast": forecast_naive,
    "Moving Average": forecast_moving_average,
    "Weighted Moving Average": forecast_weighted_moving_average,
    "Single Exponential Smoothing": forecast_single_exponential_smoothing,
    "Double Exponential Smoothing": forecast_double_exponential_smoothing,
    "Triple Exponential Smoothing": forecast_triple_exponential_smoothing,
    "Linear Trend Projection": forecast_linear_trend,
    "Least Square Quadratic Trend": forecast_least_square_quadratic,
    "Seasonal Naive": forecast_seasonal_naive,
    "ARIMA": forecast_arima
}

def run_forecast(method_name, history, horizon, params):
    method_function = FORECAST_METHODS[method_name]
    result = method_function(history, horizon, **params)
    return np.array(result[0] if isinstance(result, tuple) else result, dtype=float)

def run_forecast_with_params(method_name, history, horizon, params):
    method_function = FORECAST_METHODS[method_name]
    result = method_function(history, horizon, **params)
    if isinstance(result, tuple):
        return np.array(result[0], dtype=float), result[1]
    return np.array(result, dtype=float), {}

def evaluate_one_method(method_name, train, test, test_periods, params):
    forecast = run_forecast(method_name, train, len(test), params)
    error_table, metrics = calculate_error_table(test_periods, test, forecast)
    return forecast, error_table, metrics

def evaluate_all_methods(train, test, test_periods, params):
    rows = []
    details = {}
    for method_name in FORECAST_METHODS.keys():
        forecast, error_table, metrics = evaluate_one_method(method_name, train, test, test_periods, params)
        rows.append({"Metode": method_name, "MAD": metrics["MAD"], "MSE": metrics["MSE"], "MAPE": metrics["MAPE"]})
        details[method_name] = {"forecast": forecast, "error_table": error_table, "metrics": metrics}
    comparison_df = pd.DataFrame(rows).sort_values(by=["MAPE", "MAD", "MSE"], ascending=True, na_position="last").reset_index(drop=True)
    return comparison_df, details


def convert_all_to_excel(comparison_df, best_method_name, future_labels, future_forecast):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, index=False, sheet_name='Perbandingan_Metode')
        best_df = pd.DataFrame({"Periode": future_labels, "Forecast Utama": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name='Proyeksi_Metode_Terbaik')
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': '#9B1C1C', 'font_color': '#FFFFFF', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        ws1 = writer.sheets['Perbandingan_Metode']
        ws1.set_row(0, 24)
        for col_num, value in enumerate(comparison_df.columns.values):
            ws1.write(0, col_num, value, header_format)
        for i, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            ws1.set_column(i, i, max_len, text_format if col == "Metode" else num_format)
        ws2 = writer.sheets['Proyeksi_Metode_Terbaik']
        ws2.set_row(0, 24)
        for col_num, value in enumerate(best_df.columns.values):
            ws2.write(0, col_num, value, header_format)
        for i, col in enumerate(best_df.columns):
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            ws2.set_column(i, i, max_len, text_format if col == "Periode" else num_format)
    return output.getvalue()


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Proyeksi')
        workbook = writer.book
        worksheet = writer.sheets['Hasil_Proyeksi']
        worksheet.set_row(0, 24)
        header_format = workbook.add_format({'bold': True, 'bg_color': '#9B1C1C', 'font_color': '#FFFFFF', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 4
            worksheet.set_column(i, i, max_len, text_format if col in ["No", "Periode"] else num_format)
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  INTERFACE UTAMA
# ─────────────────────────────────────────────────────────────────────────────

# Header utama
st.markdown("""
<div class="title-block">
    <div class="deco-left"></div>
    <div class="title-badge">📊 Analytical Intelligence Platform</div>
    <h1>✦ Dashboard Peramalan Data Historis</h1>
    <p>Aplikasi analitik interaktif berbasis sains data untuk menghitung peramalan tingkat lanjut dengan 10 metode statistik.</p>
</div>
""", unsafe_allow_html=True)

if not STATSMODELS_AVAILABLE:
    st.warning("⚠️ Library **statsmodels** belum tersedia. Metode Exponential Smoothing dan ARIMA menggunakan fallback Naive Forecast.")

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🔮 Pengaturan Input")
    uploaded_file = st.file_uploader("Upload data historis (.csv / .xlsx)", type=["csv", "xlsx"])
    st.divider()

    st.markdown("### ⚙️ Pengaturan Evaluasi")
    test_percentage = st.slider("Persentase data uji (%)", min_value=10, max_value=50, value=20, step=5)
    future_horizon = st.number_input("Jumlah periode ke depan", min_value=1, max_value=60, value=6, step=1)
    mode = st.radio("Mode Perhitungan", ["Satu metode", "Bandingkan semua metode"])
    selected_method = st.selectbox("Pilih Metode Utama", list(FORECAST_METHODS.keys()))
    st.divider()

    st.markdown("### 🛠️ Parameter Tambahan")
    st.markdown("**Exponential Smoothing Parameters**")
    smoothing_mode = st.radio("Metode Penyetelan", ["Optimasi otomatis", "Input manual"])

    if smoothing_mode == "Input manual":
        alpha_input = st.slider("Alpha (Level)", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
        beta_input = st.slider("Beta (Trend)", min_value=0.01, max_value=0.99, value=0.20, step=0.01)
        gamma_input = st.slider("Gamma (Seasonality)", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    else:
        alpha_input = beta_input = gamma_input = None

    ma_window = st.number_input("Window Moving Average", min_value=2, max_value=24, value=3, step=1)
    wma_weight_text = st.text_input("Bobot WMA (Pisahkan koma)", value="0.2, 0.3, 0.5")
    seasonal_periods = st.number_input("Seasonal Periods (Musiman)", min_value=2, max_value=52, value=12, step=1)

    st.markdown("**ARIMA Parameters (p, d, q)**")
    arima_p = st.number_input("Order p (AR)", min_value=0, max_value=5, value=1, step=1)
    arima_d = st.number_input("Order d (I)", min_value=0, max_value=2, value=1, step=1)
    arima_q = st.number_input("Order q (MA)", min_value=0, max_value=5, value=1, step=1)

    st.divider()
    process_button = st.button("🚀 Jalankan Proses", type="primary")


# ── HALAMAN AWAL (belum ada file) ──
if uploaded_file is None:
    st.info("💡 **Petunjuk:** Unggah berkas Excel atau CSV di panel kiri untuk memulai analisis peramalan.")

    st.markdown("### 📋 Contoh Struktur Data yang Valid")
    sample = pd.DataFrame({
        "Tanggal": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "Penjualan": [120, 135, 128, 140, 150, 160, 155, 170, 180, 175, 190, 200]
    })
    preview_df = sample.copy()
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.stop()

# ── BACA FILE ──
try:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"File gagal dibaca: {e}"); st.stop()

if df_raw.empty:
    st.error("File tidak memiliki data."); st.stop()

# ── PREVIEW DATA ──
st.markdown("### 📂 Pratinjau Data Unggahan")
preview_df = df_raw.head(20).copy()
preview_df.insert(0, "No", range(1, len(preview_df) + 1))
st.dataframe(preview_df, use_container_width=True, hide_index=True)

# ── PEMILIHAN KOLOM ──
columns = df_raw.columns.tolist()
col1, col2 = st.columns(2)
with col1:
    period_options = ["Tidak ada"] + columns
    default_period_index = period_options.index("Tanggal") if "Tanggal" in period_options else 0
    period_option = st.selectbox("📅 Kolom Indeks Waktu / Periode", period_options, index=default_period_index)
with col2:
    default_value_index = columns.index("Penjualan") if "Penjualan" in columns else 0
    value_col = st.selectbox("📈 Kolom Nilai Aktual (Numerik)", columns, index=default_value_index)

period_col = None if period_option == "Tidak ada" else period_option
df = df_raw.copy()

if period_col is not None:
    temp_date = pd.to_datetime(df[period_col], errors="coerce")
    if temp_date.notna().mean() >= 0.7:
        df["_parsed_period"] = temp_date
        df = df.sort_values("_parsed_period").drop(columns=["_parsed_period"])
    else:
        df = df.sort_values(period_col)

values_series = clean_numeric_series(df[value_col])
if len(values_series) < 6:
    st.error("Data numerik terlalu sedikit! Gunakan minimal **6 baris** data numerik.")
    st.stop()

df = df.loc[values_series.index].reset_index(drop=True)
values = values_series.reset_index(drop=True).values
period_labels, period_dates = make_period_labels(df, period_col)

params = {
    "window": int(ma_window), "weights": parse_weights(wma_weight_text),
    "seasonal_periods": int(seasonal_periods),
    "arima_order": (int(arima_p), int(arima_d), int(arima_q)),
    "optimized": smoothing_mode == "Optimasi otomatis",
    "alpha": alpha_input, "beta": beta_input, "gamma": gamma_input
}

train, test, test_size = split_train_test(values, test_percentage)
train_periods = period_labels[:-test_size]
test_periods = period_labels[-test_size:]

# ── RINGKASAN DISTRIBUSI DATA ──
st.markdown("### 📌 Ringkasan Dataset")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Observasi", len(values))
metric_col2.metric("Dataset Latih (Train)", len(train))
metric_col3.metric("Dataset Uji (Test Validation)", len(test))

st.divider()

# ── TAB UTAMA ──
tab_data, tab_grafik = st.tabs(["🔍  Karakteristik & Tren Data", "📊  Hasil Komputasi Peramalan"])

with tab_data:
    st.markdown("### Analisis Karakteristik Data Historis")
    c_desc, c_roll = st.columns([1, 2])

    with c_desc:
        st.markdown("**Statistik Deskriptif**")
        desc_df = pd.DataFrame(values, columns=[value_col]).describe()
        st.dataframe(desc_df, use_container_width=True)

    with c_roll:
        st.markdown("**Deteksi Pergerakan Tren (Rolling Mean)**")
        roll_df = pd.DataFrame({"Periode": period_labels, "Aktual": values})
        roll_df["Rolling_Mean"] = roll_df["Aktual"].rolling(window=min(3, len(values)), min_periods=1).mean()

        roll_fig = go.Figure()
        roll_fig.add_trace(go.Scatter(
            x=roll_df["Periode"], y=roll_df["Aktual"], name="Aktual", mode="lines",
            line=dict(color="#9B1C1C", width=2)
        ))
        roll_fig.add_trace(go.Scatter(
            x=roll_df["Periode"], y=roll_df["Rolling_Mean"], name="Rolling Mean",
            line=dict(dash="dot", color="#B45309", width=2)
        ))
        rl = dict(PLOT_LAYOUT)
        rl["height"] = 320
        rl["margin"] = dict(l=20, r=20, t=20, b=20)
        roll_fig.update_layout(**rl)
        st.plotly_chart(roll_fig, use_container_width=True)

with tab_grafik:
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(
        x=period_labels, y=values, mode="lines+markers", name="Nilai Aktual",
        line=dict(color="#9B1C1C", width=2.5),
        marker=dict(size=5, color="#9B1C1C", line=dict(color="#FFFFFF", width=1)),
        fill="tozeroy",
        fillcolor="rgba(155,28,28,0.07)",
    ))
    hl = dict(PLOT_LAYOUT)
    hl["title"] = dict(text="Visualisasi Runtun Waktu Historis", font=dict(color="#1C1917", size=14, family="Playfair Display, serif"), x=0.02)
    hl["xaxis"] = dict(PLOT_LAYOUT["xaxis"], title="Periode")
    hl["yaxis"] = dict(PLOT_LAYOUT["yaxis"], title="Nilai")
    history_fig.update_layout(**hl)
    st.plotly_chart(history_fig, use_container_width=True)

    # ── PROSES ──
    if process_button:
        if mode == "Satu metode":
            forecast_test, error_table, metrics = evaluate_one_method(selected_method, train, test, test_periods, params)
            _, used_params = run_forecast_with_params(selected_method, train, len(test), params)
            future_forecast = run_forecast(selected_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))
            residuals = test - forecast_test
            std_error = np.std(residuals)

            st.markdown(f"### 📊 Hasil Analisis — {selected_method}")

            # Metric akurasi
            with st.container(border=True):
                st.markdown("**🎯 Metrik Validasi Tingkat Akurasi**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Akurasi (MAPE)", f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A")
                m2.metric("Error (MAD)", f"{metrics['MAD']:.4f}")
                m3.metric("Error (MSE)", f"{metrics['MSE']:.4f}")

            # Parameter smoothing
            if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"]:
                with st.container(border=True):
                    st.markdown("**⚙️ Nilai Parameter Optimal**")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Alpha (Level)", format_param(used_params.get("Alpha")))
                    p2.metric("Beta (Trend)", format_param(used_params.get("Beta")))
                    p3.metric("Gamma (Seasonality)", format_param(used_params.get("Gamma")))

            tab1, tab2 = st.tabs(["📉  Grafik Evaluasi Model", "🔮  Hasil Proyeksi Masa Depan"])

            with tab1:
                st.plotly_chart(
                    plot_actual_forecast(test_periods, test, forecast_test, "Uji Validasi: Data Aktual vs Estimasi Model"),
                    use_container_width=True
                )
                with st.expander("📋 Lihat Rincian Tabel Komputasi Error"):
                    error_table_view = error_table.copy()
                    error_table_view.insert(0, "No", range(1, len(error_table_view) + 1))
                    st.dataframe(error_table_view, use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("### 🔮 Proyeksi Nilai Masa Depan")
                st.plotly_chart(
                    plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, std_error),
                    use_container_width=True
                )
                st.divider()

                col_tabel, col_download = st.columns([2, 1])
                with col_tabel:
                    st.markdown("**Tabel Angka Hasil Prediksi**")
                    f_df = pd.DataFrame({"Periode": future_labels, "Hasil Forecast": future_forecast})
                    f_df_view = f_df.copy()
                    f_df_view.insert(0, "No", range(1, len(f_df_view) + 1))
                    st.dataframe(f_df_view, use_container_width=True, hide_index=True)

                with col_download:
                    st.markdown("**Unduh Hasil**")
                    excel_data = convert_df_to_excel(f_df)
                    st.download_button(
                        label="📥 Download Hasil (.xlsx)",
                        data=excel_data,
                        file_name=f"Hasil_Proyeksi_{selected_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        else:
            # Mode: Bandingkan semua metode
            comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
            best_method = comparison_df.iloc[0]["Metode"]
            future_forecast = run_forecast(best_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            st.markdown("### 🏆 Perbandingan Akurasi Semua Metode")
            st.caption("Diurutkan otomatis dari model dengan tingkat error (MAPE) terkecil ke terbesar.")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="best-badge">
                <span>✅ Rekomendasi Model Terbaik:</span>
                <span style="color:#B45309 !important; font-size:1rem !important;">{best_method}</span>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(
                plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, 0),
                use_container_width=True
            )

            excel_all = convert_all_to_excel(comparison_df, best_method, future_labels, future_forecast)
            st.download_button(
                label="📥 Download Laporan Perbandingan Lengkap (.xlsx)",
                data=excel_all,
                file_name="Laporan_Perbandingan_Metode_Peramalan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
