import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st  # <--- Pastikan baris ini ada SEBELUM st.set_page_config
import io

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# Sekarang Streamlit sudah di-import, fungsi ini tidak akan error lagi
st.set_page_config(
    page_title="Dashboard Peramalan Modern",
    page_icon="📈",
    layout="wide"
)

# ==============================================================================
# SUNTIKAN CSS - MODERN, FORMAL & COLORFUL PREMIUM SLATE STYLE
# ==============================================================================
st.markdown("""
    <style>
    /* 1. Latar Belakang Utama */
    .stApp {
        background-color: #0b0f19 !important;
        background-image: radial-gradient(at 0% 0%, rgba(30, 41, 59, 0.4) 0, transparent 50%), 
                          radial-gradient(at 50% 0%, rgba(79, 70, 229, 0.1) 0, transparent 50%) !important;
    }

    /* 2. Styling Wadah Konten & Dataframe (Tabel) */
    [data-testid="stDataFrame"], 
    div[data-testid="stElementContainer"] div[data-style="border"],
    .stCustomBlockContainer {
        background-color: #131c2e !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }
    
    /* 3. Tipografi Aman (Menghindari Teks Hilang) */
    h1, h2, h3, h4, .stMarkdown p, label [data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important; 
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    
    /* Judul Utama dengan Gradien Colorful */
    .main-title {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        padding-top: 10px;
    }

    /* 4. Panel Kontrol Kiri (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #070a12 !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Memastikan teks di dalam sidebar tetap terlihat jelas */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    /* 5. Tombol Utama (Gradien & Efek Hover Modern) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4) !important;
    }

    /* 6. Tabs Navigasi Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 6px;
        border-radius: 10px;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        color: #94a3b8 !important;
        padding: 8px 16px !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI PROSES DAN PERHITUNGAN DASAR (TETAP SAMA) ---

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
        "Periode": periods,
        "Aktual": actual,
        "Forecast": forecast,
        "Error": error,
        "Absolute Error": abs_error,
        "Squared Error": squared_error,
        "APE (%)": ape
    })

    metrics = {
        "MAD": np.mean(abs_error),
        "MSE": np.mean(squared_error),
        "MAPE": safe_mape(actual, forecast)
    }

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
            future_dates = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=inferred_freq
            )[1:]
            return future_dates.strftime("%Y-%m-%d").tolist()

        delta = valid_dates.iloc[-1] - valid_dates.iloc[-2]
        future_dates = [last_date + (i * delta) for i in range(1, horizon + 1)]
        return [d.strftime("%Y-%m-%d") for d in future_dates]

    return [f"Periode {len(existing_labels) + i}" for i in range(1, horizon + 1)]


# ==============================================================================
# PERBARUAN: TUNING WARNA GRAFIK PLOTLY AGAR MATCH DENGAN THEME BARU (DARK-SLATE)
# ==============================================================================
PLOTLY_DARK_THEME = dict(
    paper_bgcolor='rgba(19, 28, 46, 1)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#cbd5e1', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
)

def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    # Aktual: Electric Cyan
    fig.add_trace(go.Scatter(x=periods, y=actual, mode="lines+markers", name="Aktual", line=dict(color='#06b6d4', width=3)))
    # Forecast: Neon Pink/Fuchsia
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode="lines+markers", name="Forecast", line=dict(color='#d946ef', width=3)))
    
    fig.update_layout(title=title, xaxis_title="Periode", yaxis_title="Nilai", hovermode="x unified")
    fig.update_layout(PLOTLY_DARK_THEME)
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    
    # Historis: Bright Indigo
    fig.add_trace(go.Scatter(x=all_periods, y=actual_values, mode="lines+markers", name="Aktual Historis", line=dict(color='#6366f1', width=3)))
    
    # Confidence Interval: Semi-transparent Purple Neon
    if residual_std > 0:
        upper_bound = future_forecast + (1.96 * residual_std)
        lower_bound = future_forecast - (1.96 * residual_std)
        lower_bound = np.clip(lower_bound, 0, None)
        
        fig.add_trace(go.Scatter(
            x=future_periods + future_periods[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor='rgba(168, 85, 247, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Interval Keyakinan (95%)"
        ))

    # Garis Proyeksi Utama: Neon Pink Putus-putus
    fig.add_trace(go.Scatter(x=future_periods, y=future_forecast, mode="lines+markers", name="Proyeksi Utama", line=dict(color='#ec4899', width=3, dash='dash')))

    fig.update_layout(
        title="Grafik Proyeksi Nilai Masa Depan",
        xaxis_title="Periode",
        yaxis_title="Nilai",
        hovermode="x unified"
    )
    fig.update_layout(PLOTLY_DARK_THEME)
    return fig
