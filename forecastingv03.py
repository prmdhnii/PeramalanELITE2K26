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

# Suntikan CSS - ENTERPRISE FORMAL STYLE
st.markdown("""
    <style>
    /* 1. Fondasi Font & Background Utama */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #F8FAFC !important;
    }

    /* 2. Sidebar Executive Look */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p,
    [data-testid="stSidebar"] p {
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 6px;
        margin-top: 20px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 3. Layout Uploader Minimalis */
    [data-testid="stFileUploader"] {
        background-color: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #0F172A !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 6px 14px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    
    [data-testid="stFileUploader"] button * {
        font-size: 0px !important;
        color: transparent !important;
        display: none !important;
    }
    
    [data-testid="stFileUploader"] button::after {
        content: "Pilih File" !important;
        color: #FFFFFF !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        display: block !important;
    }

    [data-testid="stFileUploader"] text {
        fill: #64748B !important;
    }
    [data-testid="stFileUploader"] div {
        color: #64748B !important;
        font-size: 0.8rem !important;
    }

    /* 4. Area Konten Utama */
    .main h1 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        font-size: 2rem !important;
        margin-bottom: 5px !important;
    }
    
    .main p {
        color: #475569 !important;
    }

    /* 5. Institusional Metric Cards */
    [data-testid="stMetricValue"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        border-radius: 6px !important;
        padding: 15px 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        border: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        margin-left: 5px !important;
    }

    /* 6. Tombol Utama (Solid Corporate Blue) - DIPERBAIKI AGAR TEKS TERLIHAT JELAS */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;            /* Memaksa teks berwarna putih bersih */
        font-weight: 700 !important;          /* Membuat teks lebih tebal dan tegas */
        font-size: 0.95rem;
        padding: 0.55rem 1rem;
        border: 1px solid #172554 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: background-color 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;            /* Memastikan teks tetap putih saat hover */
        border: 1px solid #1E3A8A !important;
    }

    /* 7. Desain Tabel Data Grid */
    .stDataFrame {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px;
    }

    hr {
        margin: 1.2rem 0 !important;
        border-color: #E2E8F0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI PROSES DAN PERHITUNGAN DASAR ---

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


# --- FUNGSI GRAFIK PLOTLY ---

def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=actual, mode="lines+markers", name="Aktual", line=dict(color='#1E3A8A')))
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode="lines+markers", name="Forecast", line=dict(color='#E11D48')))
    fig.update_layout(title=title, xaxis_title="Periode", yaxis_title="Nilai", hovermode="x unified", template="plotly_white")
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    
    # Historis
    fig.add_trace(go.Scatter(x=all_periods, y=actual_values, mode="lines+markers", name="Aktual Historis", line=dict(color='#1E3A8A')))
    
    # Hitung interval jika standard deviasi tersedia
    if residual_std > 0:
        upper_bound = future_forecast + (1.96 * residual_std)
        lower_bound = future_forecast - (1.96 * residual_std)
        lower_bound = np.clip(lower_bound, 0, None)
        
        fig.add_trace(go.Scatter(
            x=future_periods + future_periods[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor='rgba(225, 29, 72, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Interval Keyakinan (95%)"
        ))

    # Garis Forecast Utama
    fig.add_trace(go.Scatter(x=future_periods, y=future_forecast, mode="lines+markers", name="Proyeksi Utama", line=dict(color='#E11D48', dash='dash')))

    fig.update_layout(
        title="Grafik Proyeksi Nilai Masa Depan",
        xaxis_title="Periode",
        yaxis_title="Nilai",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


# --- ALGORITMA METODE PERAMALAN ---

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
        if value > maximum: return maximum
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
        rows.append({
            "Metode": method_name,
            "MAD": metrics["MAD"],
            "MSE": metrics["MSE"],
            "MAPE": metrics["MAPE"]
        })
        details[method_name] = {"forecast": forecast, "error_table": error_table, "metrics": metrics}
    
    comparison_df = pd.DataFrame(rows).sort_values(by=["MAPE", "MAD", "MSE"], ascending=True, na_position="last").reset_index(drop=True)
    return comparison_df, details


def convert_all_to_excel(comparison_df, best_method_name, future_labels, future_forecast):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, index=False, sheet_name='Perbandingan_Metode')
        
        best_df = pd.DataFrame({"Periode": future_labels, "Forecast Utama": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name='Proyeksi_Metode_Terbaik')
        
        workbook  = writer.book
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#1E3A8A', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        # Format Sheet 1
        ws1 = writer.sheets['Perbandingan_Metode']
        ws1.set_row(0, 24)
        for col_num, value in enumerate(comparison_df.columns.values):
            ws1.write(0, col_num, value, header_format)
            
        for i, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            if col == "Metode":
                ws1.set_column(i, i, max_len, text_format)
            else:
                ws1.set_column(i, i, max_len, num_format)
                
        # Format Sheet 2
        ws2 = writer.sheets['Proyeksi_Metode_Terbaik']
        ws2.set_row(0, 24)
        for col_num, value in enumerate(best_df.columns.values):
            ws2.write(0, col_num, value, header_format)
            
        for i, col in enumerate(best_df.columns):
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            if col == "Periode":
                ws2.set_column(i, i, max_len, text_format)
            else:
                ws2.set_column(i, i, max_len, num_format)
            
    return output.getvalue()


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Proyeksi')
        
        workbook  = writer.book
        worksheet = writer.sheets['Hasil_Proyeksi']
        worksheet.set_row(0, 24)
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#1E3A8A', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 4
            if col in ["No", "Periode"]:
                worksheet.set_column(i, i, max_len, text_format)
            else:
                worksheet.set_column(i, i, max_len, num_format)
                
    return output.getvalue()


# --- INTERFACE UTAMA DASHBOARD ---

st.title("Dashboard Peramalan Data Historis")
st.write("Aplikasi ini menghitung peramalan tingkat lanjut, melakukan validasi model, mendeteksi karakteristik data historis, serta menghitung error metrik.")

if not STATSMODELS_AVAILABLE:
    st.warning("Library statsmodels belum tersedia. Metode Exponential Smoothing dan ARIMA memakai fallback Naive Forecast.")

with st.sidebar:
    st.header("Pengaturan Input")
    uploaded_file = st.file_uploader("Upload data historis", type=["csv", "xlsx"])
    st.divider()

    st.header("Pengaturan Evaluasi")
    test_percentage = st.slider("Persentase data uji", min_value=10, max_value=50, value=20, step=5)
    future_horizon = st.number_input("Jumlah periode forecast masa depan", min_value=1, max_value=60, value=6, step=1)
    mode = st.radio("Mode perhitungan", ["Satu metode", "Bandingkan semua metode"])
    selected_method = st.selectbox("Pilih metode", list(FORECAST_METHODS.keys()))
    st.divider()

    st.header("Parameter Metode")
    st.write("Parameter Exponential Smoothing")
    smoothing_mode = st.radio("Mode parameter smoothing", ["Optimasi otomatis", "Input manual"])

    if smoothing_mode == "Input manual":
        alpha_input = st.slider("Alpha", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
        beta_input = st.slider("Beta", min_value=0.01, max_value=0.99, value=0.20, step=0.01)
        gamma_input = st.slider("Gamma", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    else:
        alpha_input = beta_input = gamma_input = None

    ma_window = st.number_input("Window Moving Average", min_value=2, max_value=24, value=3, step=1)
    wma_weight_text = st.text_input("Bobot WMA", value="0.2, 0.3, 0.5", help="Contoh: 0.2, 0.3, 0.5.")
    seasonal_periods = st.number_input("Seasonal periods", min_value=2, max_value=52, value=12, step=1)

    st.write("Parameter ARIMA")
    arima_p = st.number_input("p", min_value=0, max_value=5, value=1, step=1)
    arima_d = st.number_input("d", min_value=0, max_value=2, value=1, step=1)
    arima_q = st.number_input("q", min_value=0, max_value=5, value=1, step=1)

    process_button = st.button("Proses Peramalan", type="primary")


if uploaded_file is None:
    st.info("Silakan upload file CSV atau Excel terlebih dahulu.")
    st.subheader("Contoh Format Data")
    sample = pd.DataFrame({
        "Tanggal": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "Penjualan": [120, 135, 128, 140, 150, 160, 155, 170, 180, 175, 190, 200]
    })
    preview_df = sample.copy()
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"File gagal dibaca: {e}"); st.stop()

if df_raw.empty:
    st.error("File tidak memiliki data."); st.stop()

st.subheader("Preview Data")
preview_df = df_raw.head(20).copy()
preview_df.insert(0, "No", range(1, len(preview_df) + 1))
st.dataframe(preview_df, use_container_width=True, hide_index=True)

columns = df_raw.columns.tolist()
col1, col2 = st.columns(2)

with col1:
    period_options = ["Tidak ada"] + columns
    default_period_index = period_options.index("Tanggal") if "Tanggal" in period_options else 0
    period_option = st.selectbox("Pilih kolom periode", period_options, index=default_period_index)

with col2:
    default_value_index = columns.index("Penjualan") if "Penjualan" in columns else 0
    value_col = st.selectbox("Pilih kolom nilai aktual", columns, index=default_value_index)

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
    st.error("Data numerik terlalu sedikit. Minimal gunakan 6 baris data historis.")
    st.stop()

df = df.loc[values_series.index].reset_index(drop=True)
values = values_series.reset_index(drop=True).values
period_labels, period_dates = make_period_labels(df, period_col)

params = {
    "window": int(ma_window), "weights": parse_weights(wma_weight_text), "seasonal_periods": int(seasonal_periods),
    "arima_order": (int(arima_p), int(arima_d), int(arima_q)), "optimized": smoothing_mode == "Optimasi otomatis",
    "alpha": alpha_input, "beta": beta_input, "gamma": gamma_input
}

train, test, test_size = split_train_test(values, test_percentage)
train_periods = period_labels[:-test_size]
test_periods = period_labels[-test_size:]

st.subheader("Ringkasan Data")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Jumlah Data", len(values))
metric_col2.metric("Data Latih", len(train))
metric_col3.metric("Data Uji", len(test))


# --- AREA IMPLEMENTASI GRID TAB & PROSES UTAMA ---

tab_data, tab_grafik = st.tabs(["🔍 Analisis Karakteristik Data", "📊 Hasil & Grafik Utama"])

with tab_data:
    st.write("### Analisis Karakteristik Data Historis")
    c_desc, c_roll = st.columns([1, 2])
    
    with c_desc:
        st.write("**Statistik Deskriptif Internal**")
        desc_df = pd.DataFrame(values, columns=[value_col]).describe()
        st.dataframe(desc_df, use_container_width=True)
        
    with c_roll:
        st.write("**Deteksi Tren (Rolling Mean 3 Periode)**")
        roll_df = pd.DataFrame({"Periode": period_labels, "Aktual": values})
        roll_df["Rolling_Mean"] = roll_df["Aktual"].rolling(window=min(3, len(values)), min_periods=1).mean()
        
        roll_fig = go.Figure()
        roll_fig.add_trace(go.Scatter(x=roll_df["Periode"], y=roll_df["Aktual"], name="Aktual", mode="lines"))
        roll_fig.add_trace(go.Scatter(x=roll_df["Periode"], y=roll_df["Rolling_Mean"], name="Tren (Rolling Average)", line=dict(dash='dot', color='orange')))
        roll_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
        st.plotly_chart(roll_fig, use_container_width=True)

with tab_grafik:
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(x=period_labels, y=values, mode="lines+markers", name="Nilai Aktual"))
    history_fig.update_layout(title="Grafik Data Historis", xaxis_title="Periode", yaxis_title="Nilai", hovermode="x unified", template="plotly_white")
    st.plotly_chart(history_fig, use_container_width=True)

    if process_button:
        if mode == "Satu metode":
            forecast_test, error_table, metrics = evaluate_one_method(selected_method, train, test, test_periods, params)
            _, used_params = run_forecast_with_params(selected_method, train, len(test), params)
            future_forecast = run_forecast(selected_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            residuals = test - forecast_test
            std_error = np.std(residuals)

            st.subheader(f"📊 Hasil Analisis: {selected_method}")
            with st.container(border=True):
                st.write("**📈 Performa Model (Error)**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Akurasi (MAPE)", f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A")
                m2.metric("Error (MAD)", f"{metrics['MAD']:.4f}")
                m3.metric("Error (MSE)", f"{metrics['MSE']:.4f}")

            if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"]:
                with st.container(border=True):
                    st.write("**⚙️ Konfigurasi Smoothing (Alpha, Beta, Gamma)**")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Alpha (Level)", format_param(used_params.get("Alpha")), help="Bobot data terbaru")
                    p2.metric("Beta (Trend)", format_param(used_params.get("Beta")), help="Bobot pola tren")
                    p3.metric("Gamma (Seasonality)", format_param(used_params.get("Gamma")), help="Bobot pola musiman")
                    st.caption(f"Metode optimasi: {smoothing_mode}")

            st.write("") 
            tab1, tab2 = st.tabs(["📉 Grafik & Validasi", "🔮 Proyeksi Masa Depan"])

            with tab1:
                st.plotly_chart(plot_actual_forecast(test_periods, test, forecast_test, "Validasi Model: Aktual vs Prediksi"), use_container_width=True)
                with st.expander("Klik untuk cek Tabel Error per Periode"):
                    error_table_view = error_table.copy()
                    error_table_view.insert(0, "No", range(1, len(error_table_view) + 1))
                    st.dataframe(error_table_view, use_container_width=True, hide_index=True)

            with tab2:
                st.write("### 🔮 Proyeksi Tren Masa Depan")
                st.plotly_chart(plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, std_error), use_container_width=True)
                st.divider()

                col_tabel, col_download = st.columns([2, 1])
                with col_tabel:
                    st.write("**Tabel Angka Proyeksi**")
                    f_df = pd.DataFrame({"Periode": future_labels, "Forecast": future_forecast})
                    
                    # Tampilan Grid Proyeksi Tunggal yang Rapi
                    f_df_view = f_df.copy()
                    f_df_view.insert(0, "No", range(1, len(f_df_view) + 1))
                    st.dataframe(f_df_view, use_container_width=True, hide_index=True)

                with col_download:
                    st.write("**Aksi Data**")
                    excel_data = convert_df_to_excel(f_df)
                    st.download_button(
                        label="📥 Download Hasil Proyeksi (Excel)",
                        data=excel_data,
                        file_name=f"proyeksi_{selected_method.lower().replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        elif mode == "Bandingkan semua metode":
            comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
            
            st.subheader("📊 Hasil Perbandingan Akurasi Semua Metode")
            st.write("Tabel diurutkan otomatis dari metode dengan tingkat akurasi tertinggi (MAPE terkecil).")
            
            # Tampilan Grid Perbandingan Tabel Akurasi
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # FITUR 1: DETEKSI OTOMATIS MODEL TERBAIK (Sistem cerdas mencari MAPE terkecil)
            best_method_name = comparison_df.iloc[0]["Metode"]
            best_mape = comparison_df.iloc[0]["MAPE"]
            
            st.success(f"💡 **Rekomendasi Sistem:** Metode **{best_method_name}** dipilih secara otomatis sebagai model terbaik karena memiliki tingkat error MAPE paling rendah ({best_mape:.2f}%).")
            
            # Jalankan peramalan masa depan langsung menggunakan model terbaik
            best_future_forecast = run_forecast(best_method_name, values, int(future_horizon), params)
            best_future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))
            
            # Hitung deviasi standar dari residual model terbaik untuk interval keyakinan grafik
            best_forecast_test = details[best_method_name]["forecast"]
            best_residuals = test - best_forecast_test
            best_std_error = np.std(best_residuals)
            
            # Tampilkan grafik masa depan dari model terbaik terpilih
            st.plotly_chart(plot_future_forecast_with_ci(period_labels, values, best_future_labels, best_future_forecast, best_std_error), use_container_width=True)
            
            # Tombol unduh laporan multi-sheet komparasi
            excel_all_data = convert_all_to_excel(comparison_df, best_method_name, best_future_labels, best_future_forecast)
            st.download_button(
                label="📥 Download Laporan Komparasi Lengkap (Excel)",
                data=excel_all_data,
                file_name="laporan_komparasi_peramalan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
