import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import io

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# ─── NEW: PDF Report Generation ───────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


st.set_page_config(
    page_title="Forecasting Dashboard Elementary Laboratory",
    page_icon="📈",
    layout="wide"
)

# CSS Injection - VIBRANT COLORFUL LIGHT STYLE
st.markdown("""
    <style>
    /* 1. Font & Main Background Foundation */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #F5F7FF 0%, #FAFAFA 100%) !important;
    }

    /* 2. Bright & Fresh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 2px solid #E0E4FF !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p,
    [data-testid="stSidebar"] p {
        color: #4A5568 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #4F46E5 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-bottom: 2px solid #F0F2FF;
        padding-bottom: 8px;
        margin-top: 20px !important;
        letter-spacing: 0.5px;
    }

    /* 3. Colorful File Uploader Layout */
    [data-testid="stFileUploader"] {
        background-color: #F0F4FF !important;
        border: 2px dashed #6366F1 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2) !important;
    }
    
    [data-testid="stFileUploader"] button * {
        font-size: 0px !important;
        color: transparent !important;
        display: none !important;
    }
    
    [data-testid="stFileUploader"] button::after {
        content: "Choose Your File" !important;
        color: #FFFFFF !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        display: block !important;
    }

    [data-testid="stFileUploader"] text {
        fill: #4F46E5 !important;
    }
    [data-testid="stFileUploader"] div {
        color: #4A5568 !important;
        font-weight: 500;
    }

    /* 4. Main Content Area */
    .main h1 {
        background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.3rem !important;
        margin-bottom: 8px !important;
    }
    
    .main p {
        color: #64748B !important;
        font-weight: 500;
    }

    /* 5. Colorful Pastel Metric Cards */
    [data-testid="stMetricValue"] {
        background: #FFFFFF !important;
        color: #1E1B4B !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        border-left: 5px solid #EC4899 !important;
        border-top: 1px solid #EEF2F6 !important;
        border-right: 1px solid #EEF2F6 !important;
        border-bottom: 1px solid #EEF2F6 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4F46E5 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        margin-left: 5px !important;
    }

    /* 6. Pro Main Button (Neon/Bright Coral Style) */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%) !important;
        color: #FFFFFF !important;            
        font-weight: 700 !important;          
        font-size: 1rem;
        padding: 0.65rem 1rem;
        border: none !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.4);
        background: linear-gradient(135deg, #FF6B6B 0%, #FF4B4B 100%) !important;
        color: #FFFFFF !important;            
    }

    /* 7. Data Grid Table Design */
    .stDataFrame {
        background-color: #FFFFFF;
        border: 2px solid #F0F2FF !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
    }

    /* Colorful Streamlit Tabs Customization */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #64748B !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom-color: #4F46E5 !important;
    }

    hr {
        margin: 1.5rem 0 !important;
        border-color: #E2E8F0 !important;
    }

    /* Stepper style */
    .step-card {
        background: linear-gradient(135deg, #F0F4FF 0%, #EEF2FF 100%);
        border-left: 4px solid #4F46E5;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 10px;
    }

    /* 8. Expander Arrow — jelas dan kontras di semua mode */
    [data-testid="stExpander"] {
        border: 1.5px solid #C7D2FE !important;
        border-radius: 10px !important;
        background-color: #FAFBFF !important;
        margin-bottom: 8px !important;
    }

    /* Header baris expander */
    [data-testid="stExpander"] summary {
        background-color: #F0F4FF !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        cursor: pointer !important;
    }

    [data-testid="stExpander"] summary:hover {
        background-color: #E0E7FF !important;
    }

    /* Teks label expander */
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #3730A3 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Arrow/chevron SVG — paksa warna gelap & ukuran besar */
    [data-testid="stExpander"] summary svg {
        fill: #4F46E5 !important;
        color: #4F46E5 !important;
        width: 22px !important;
        height: 22px !important;
        stroke: #4F46E5 !important;
        stroke-width: 1.5px !important;
        opacity: 1 !important;
        flex-shrink: 0 !important;
    }

    /* Pastikan path dalam SVG juga ikut berwarna */
    [data-testid="stExpander"] summary svg path {
        fill: #4F46E5 !important;
        stroke: #4F46E5 !important;
    }

    /* Fallback: semua detail/summary arrow di Streamlit */
    details > summary {
        list-style: none;
    }
    details > summary::marker,
    details > summary::-webkit-details-marker {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


# --- CORE PROCESSING AND CALCULATION FUNCTIONS ---

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
        "Period": periods,
        "Actual": actual,
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
        return [f"Period {i}" for i in range(1, len(df) + 1)], None

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

    return [f"Period {len(existing_labels) + i}" for i in range(1, horizon + 1)]


# --- PLOTLY CHART FUNCTIONS ---

def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=actual, mode="lines+markers", name="Actual", line=dict(color='#4F46E5', width=3)))
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode="lines+markers", name="Forecast", line=dict(color='#EC4899', width=3)))
    fig.update_layout(title=title, xaxis_title="Period", yaxis_title="Value", hovermode="x unified", template="plotly_white")
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_periods, y=actual_values, mode="lines+markers", name="Historical Actual", line=dict(color='#4F46E5', width=3)))
    
    if residual_std > 0:
        upper_bound = future_forecast + (1.96 * residual_std)
        lower_bound = future_forecast - (1.96 * residual_std)
        lower_bound = np.clip(lower_bound, 0, None)
        
        fig.add_trace(go.Scatter(
            x=future_periods + future_periods[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor='rgba(236, 72, 153, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Confidence Interval (95%)"
        ))

    fig.add_trace(go.Scatter(x=future_periods, y=future_forecast, mode="lines+markers", name="Main Projection", line=dict(color='#EC4899', width=3, dash='dash')))

    fig.update_layout(
        title="Future Value Projection Chart",
        xaxis_title="Period",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


# ─── NEW: Boxplot Outlier Chart ────────────────────────────────────────────────
def plot_boxplot_outliers(values, period_labels, value_col_name):
    """Return boxplot + scatter for outlier visualization."""
    arr = np.array(values, dtype=float)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    z_scores = np.abs((arr - mean_val) / std_val) if std_val > 0 else np.zeros_like(arr)
    outlier_mask = z_scores > 3

    fig = go.Figure()
    # Normal points
    normal_x = [period_labels[i] for i in range(len(arr)) if not outlier_mask[i]]
    normal_y = [arr[i] for i in range(len(arr)) if not outlier_mask[i]]
    fig.add_trace(go.Scatter(
        x=normal_x, y=normal_y, mode="markers",
        name="Normal Data", marker=dict(color="#4F46E5", size=8)
    ))
    # Outlier points
    outlier_x = [period_labels[i] for i in range(len(arr)) if outlier_mask[i]]
    outlier_y = [arr[i] for i in range(len(arr)) if outlier_mask[i]]
    if outlier_x:
        fig.add_trace(go.Scatter(
            x=outlier_x, y=outlier_y, mode="markers",
            name="Outliers (Z > 3)", marker=dict(color="#EF4444", size=12, symbol="x")
        ))
    # Mean & std bands
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#10B981", line_width=1.5, annotation_text=f"Mean: {mean_val:.2f}")
    fig.add_hrect(y0=mean_val - 3*std_val, y1=mean_val + 3*std_val, fillcolor="rgba(99,102,241,0.07)", line_width=0, annotation_text="±3σ zone")
    fig.update_layout(
        title=f"Outlier Detection — {value_col_name} (Z-score > 3 flagged)",
        xaxis_title="Period", yaxis_title="Value",
        template="plotly_white", height=320,
        hovermode="x unified"
    )
    return fig


# --- FORECASTING METHOD ALGORITHMS ---

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
        rows.append({
            "Method": method_name,
            "MAD": metrics["MAD"],
            "MSE": metrics["MSE"],
            "MAPE": metrics["MAPE"]
        })
        details[method_name] = {"forecast": forecast, "error_table": error_table, "metrics": metrics}
    
    comparison_df = pd.DataFrame(rows).sort_values(by=["MAPE", "MAD", "MSE"], ascending=True, na_position="last").reset_index(drop=True)
    return comparison_df, details


# ─── EXCEL EXPORT FUNCTIONS ────────────────────────────────────────────────────

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Projection_Results')
        workbook  = writer.book
        worksheet = writer.sheets['Projection_Results']
        worksheet.set_row(0, 24)
        header_format = workbook.add_format({'bold': True, 'bg_color': '#4F46E5', 'font_color': '#FFFFFF', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 4
            if col in ["No", "Period"]:
                worksheet.set_column(i, i, max_len, text_format)
            else:
                worksheet.set_column(i, i, max_len, num_format)
    return output.getvalue()


def convert_full_comparison_excel(comparison_df, details_dict, future_labels, future_forecast, best_method):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#4F46E5", "font_color": "#FFFFFF", "border": 1, "align": "center", "valign": "vcenter"})
        num_fmt    = workbook.add_format({"num_format": "#,##0.00", "border": 1, "align": "right"})
        text_fmt   = workbook.add_format({"border": 1, "align": "left"})
        good_fmt   = workbook.add_format({"bg_color": "#D1FAE5", "border": 1, "align": "left", "bold": True})

        comparison_df.to_excel(writer, index=False, sheet_name="Method_Comparison")
        ws1 = writer.sheets["Method_Comparison"]
        ws1.set_row(0, 24)
        for ci, col in enumerate(comparison_df.columns):
            ws1.write(0, ci, col, header_fmt)
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            ws1.set_column(ci, ci, max_len, text_fmt if col == "Method" else num_fmt)
        best_row_idx = comparison_df[comparison_df["Method"] == best_method].index
        if len(best_row_idx):
            excel_row = int(best_row_idx[0]) + 1
            for ci in range(len(comparison_df.columns)):
                ws1.write(excel_row, ci, comparison_df.iloc[int(best_row_idx[0]), ci], good_fmt)

        best_df = pd.DataFrame({"Period": future_labels, "Forecast": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name="Best_Projection")
        ws2 = writer.sheets["Best_Projection"]
        ws2.set_row(0, 24)
        for ci, col in enumerate(best_df.columns):
            ws2.write(0, ci, col, header_fmt)
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            ws2.set_column(ci, ci, max_len, text_fmt if col == "Period" else num_fmt)

        for method_name, detail in details_dict.items():
            safe_name = method_name[:25].replace("/", "-").replace(":", "")
            et = detail["error_table"].copy()
            et.to_excel(writer, index=False, sheet_name=safe_name)
            ws = writer.sheets[safe_name]
            ws.set_row(0, 22)
            for ci, col in enumerate(et.columns):
                ws.write(0, ci, col, header_fmt)
                max_len = max(et[col].astype(str).map(len).max(), len(col)) + 4
                ws.set_column(ci, ci, max_len, text_fmt if col in ["Period"] else num_fmt)

    return output.getvalue()


# ─── FEATURE 4: MAPE Interpretation ───────────────────────────────────────────
def interpret_mape(mape_value):
    if mape_value is None or (isinstance(mape_value, float) and np.isnan(mape_value)):
        return "N/A", "#94A3B8", "Cannot be calculated"
    if mape_value < 10:
        return "Highly Accurate", "#10B981", "Excellent — forecast error is very low (< 10%)"
    elif mape_value < 20:
        return "Good", "#3B82F6", "Good — forecast error is acceptable (10–20%)"
    elif mape_value < 50:
        return "Reasonable", "#F59E0B", "Reasonable — use with caution (20–50%)"
    else:
        return "Inaccurate", "#EF4444", "Poor — forecast error is too high (> 50%)"


# ─── FEATURE 2: Data Quality Check ────────────────────────────────────────────
def check_data_quality(df, value_col, period_col):
    issues = []
    total = len(df)

    missing = df[value_col].isna().sum()
    if missing > 0:
        issues.append({"type": "⚠️ Missing Values", "detail": f"{missing} missing value(s) in '{value_col}'", "severity": "warning"})

    neg = (pd.to_numeric(df[value_col], errors="coerce") < 0).sum()
    if neg > 0:
        issues.append({"type": "⚠️ Negative Values", "detail": f"{neg} negative value(s) detected — may affect forecast accuracy", "severity": "warning"})

    numeric_vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(numeric_vals) >= 4:
        z_scores = np.abs((numeric_vals - numeric_vals.mean()) / numeric_vals.std())
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            issues.append({"type": "🔴 Outliers Detected", "detail": f"{outliers} extreme outlier(s) (Z-score > 3) found in value column", "severity": "error"})

    if period_col is not None:
        dupes = df[period_col].duplicated().sum()
        if dupes > 0:
            issues.append({"type": "⚠️ Duplicate Periods", "detail": f"{dupes} duplicate period/date value(s) found", "severity": "warning"})

    return issues


# ─── FEATURE 1 & 3: Multi-method comparison chart ──────────────────────────────
def plot_all_methods_comparison(test_periods, test, details_dict):
    colors = [
        "#4F46E5","#EC4899","#10B981","#F59E0B","#EF4444",
        "#06B6D4","#8B5CF6","#F97316","#84CC16","#64748B"
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_periods, y=test, mode="lines+markers",
        name="Actual", line=dict(color="#1E1B4B", width=4),
        marker=dict(size=8, symbol="circle")
    ))
    for i, (method_name, detail) in enumerate(details_dict.items()):
        fig.add_trace(go.Scatter(
            x=test_periods, y=detail["forecast"], mode="lines+markers",
            name=method_name, line=dict(color=colors[i % len(colors)], width=2, dash="dot"),
            marker=dict(size=5), opacity=0.85
        ))
    fig.update_layout(
        title="All Methods vs Actual Data — Validation Period",
        xaxis_title="Period", yaxis_title="Value",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
        height=500
    )
    return fig


# ─── FEATURE 5: Residual Analysis ─────────────────────────────────────────────
def plot_residual_analysis(test_periods, residuals):
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=test_periods, y=residuals, mode="lines+markers",
        name="Residual", line=dict(color="#6366F1", width=2),
        marker=dict(size=6)
    ))
    fig_time.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=2)
    fig_time.update_layout(
        title="Residuals Over Time (Actual − Forecast)",
        xaxis_title="Period", yaxis_title="Residual",
        template="plotly_white", height=300
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals, nbinsx=15,
        marker_color="#A855F7", opacity=0.8,
        name="Residual Distribution"
    ))
    fig_hist.update_layout(
        title="Residual Distribution (Histogram)",
        xaxis_title="Residual Value", yaxis_title="Frequency",
        template="plotly_white", height=300,
        showlegend=False
    )
    return fig_time, fig_hist


# ─── FEATURE 7: Seasonality Decomposition ─────────────────────────────────────
def plot_decomposition(values, period_labels, seasonal_periods=12):
    n = len(values)
    if n < 2 * seasonal_periods:
        return None

    half = seasonal_periods // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.mean(values[i - half: i + half + 1])

    detrended = values - trend
    seasonal = np.full(n, np.nan)
    season_avgs = np.zeros(seasonal_periods)
    counts = np.zeros(seasonal_periods)
    for i in range(n):
        if not np.isnan(detrended[i]):
            season_avgs[i % seasonal_periods] += detrended[i]
            counts[i % seasonal_periods] += 1
    season_avgs = np.where(counts > 0, season_avgs / counts, 0)
    for i in range(n):
        seasonal[i] = season_avgs[i % seasonal_periods]

    residual = values - trend - seasonal

    fig = go.Figure()
    colors_decomp = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444"]
    components = [
        ("Original", values),
        ("Trend", trend),
        ("Seasonal", seasonal),
        ("Residual", residual),
    ]
    for idx, (label, data) in enumerate(components):
        fig.add_trace(go.Scatter(
            x=period_labels, y=data, name=label,
            line=dict(color=colors_decomp[idx], width=2),
            mode="lines"
        ))
    fig.update_layout(
        title=f"Time Series Decomposition (Seasonal Period = {seasonal_periods})",
        xaxis_title="Period", yaxis_title="Value",
        template="plotly_white", height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


# ─── FEATURE 8: Export chart as HTML ──────────────────────────────────────────
def fig_to_html_bytes(fig):
    import io as _io
    buf = _io.StringIO()
    fig.write_html(buf)
    return buf.getvalue().encode("utf-8")


# ─── NEW: Seasonal Insight ─────────────────────────────────────────────────────
def get_seasonal_insight(values, period_labels):
    """Return text summary of highest/lowest demand periods."""
    if len(values) < 2:
        return None
    arr = np.array(values, dtype=float)
    idx_max = int(np.argmax(arr))
    idx_min = int(np.argmin(arr))
    return {
        "max_period": period_labels[idx_max],
        "max_value": arr[idx_max],
        "min_period": period_labels[idx_min],
        "min_value": arr[idx_min],
        "mean_value": float(np.mean(arr)),
        "trend": "upward 📈" if arr[-1] > arr[0] else "downward 📉" if arr[-1] < arr[0] else "flat ➡️"
    }


# ─── NEW: Auto-recommend best method ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def auto_recommend_method(train_tuple, test_tuple, test_periods_tuple, params_hashable):
    """Find the best method by evaluating all on train/test split. Cached."""
    train = np.array(train_tuple)
    test = np.array(test_tuple)
    test_periods = list(test_periods_tuple)
    params = dict(params_hashable)
    
    best_name = "Naive Forecast"
    best_mape = np.inf
    for method_name in FORECAST_METHODS.keys():
        try:
            _, _, metrics = evaluate_one_method(method_name, train, test, test_periods, params)
            mape_val = metrics["MAPE"]
            if not np.isnan(mape_val) and mape_val < best_mape:
                best_mape = mape_val
                best_name = method_name
        except Exception:
            pass
    return best_name, best_mape


# ─── NEW: PDF Report Generation ───────────────────────────────────────────────
def generate_pdf_report(
    method_name, metrics, future_labels, future_forecast,
    values, period_labels, value_col, mape_label, mape_color, mape_desc,
    used_params=None
):
    """Generate a 1-page PDF summary report. Returns bytes."""
    if not REPORTLAB_AVAILABLE:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=18, textColor=colors.HexColor('#4F46E5'),
                                  spaceAfter=4)
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
                                fontSize=10, textColor=colors.HexColor('#64748B'),
                                spaceAfter=12)
    elements.append(Paragraph("Predictive Production Planning for MSMEs", title_style))
    elements.append(Paragraph("Forecasting Analysis Summary Report", sub_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E0E4FF')))
    elements.append(Spacer(1, 0.3*cm))

    # Method & Accuracy section
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                    fontSize=12, textColor=colors.HexColor('#1E1B4B'),
                                    spaceAfter=6, spaceBefore=10)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#374151'),
                                 spaceAfter=4)
    elements.append(Paragraph("📊 Method & Accuracy", section_style))

    mape_val = f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A"
    acc_data = [
        ["Selected Method", method_name],
        ["MAPE (Accuracy)", mape_val],
        ["Accuracy Level", mape_label],
        ["MAD (Mean Abs. Deviation)", f"{metrics['MAD']:.4f}"],
        ["MSE (Mean Sq. Error)", f"{metrics['MSE']:.4f}"],
    ]
    if used_params:
        if used_params.get("Alpha") is not None:
            acc_data.append(["Alpha (Level)", format_param(used_params.get("Alpha"))])
        if used_params.get("Beta") is not None:
            acc_data.append(["Beta (Trend)", format_param(used_params.get("Beta"))])
        if used_params.get("Gamma") is not None:
            acc_data.append(["Gamma (Seasonality)", format_param(used_params.get("Gamma"))])

    acc_table = Table(acc_data, colWidths=[7*cm, 9*cm])
    acc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EEF2FF')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4F46E5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#F8FAFF')]),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#C7D2FE')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#E0E7FF')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(acc_table)
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(f"<i>{mape_desc}</i>", body_style))
    elements.append(Spacer(1, 0.4*cm))

    # Data Statistics
    elements.append(Paragraph("📈 Historical Data Statistics", section_style))
    arr = np.array(values, dtype=float)
    stats_data = [
        ["Total Data Points", str(len(arr))],
        ["Mean", f"{np.mean(arr):.2f}"],
        ["Std. Deviation", f"{np.std(arr):.2f}"],
        ["Min Value", f"{np.min(arr):.2f}"],
        ["Max Value", f"{np.max(arr):.2f}"],
        ["Value Column", value_col],
    ]
    stats_table = Table(stats_data, colWidths=[7*cm, 9*cm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#FDF4FF')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#9333EA')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#FDF4FF')]),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E9D5FF')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#F3E8FF')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.4*cm))

    # Future Forecast Table
    elements.append(Paragraph("🔮 Future Forecast Results", section_style))
    fc_header = [["Period", "Forecast Value", "vs Mean (%)"]]
    fc_rows = []
    mean_val = float(np.mean(arr))
    for lbl, val in zip(future_labels, future_forecast):
        diff_pct = ((val - mean_val) / mean_val * 100) if mean_val != 0 else 0
        sign = "+" if diff_pct >= 0 else ""
        fc_rows.append([str(lbl), f"{val:,.2f}", f"{sign}{diff_pct:.1f}%"])
    fc_data = fc_header + fc_rows
    fc_table = Table(fc_data, colWidths=[6*cm, 5.5*cm, 4.5*cm])
    fc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F4FF')]),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#C7D2FE')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#E0E7FF')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(fc_table)
    elements.append(Spacer(1, 0.5*cm))

    # Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E0E4FF')))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.HexColor('#94A3B8'),
                                   spaceBefore=6, alignment=TA_CENTER)
    import datetime
    elements.append(Paragraph(
        f"Generated by Predictive Production Planning for MSMEs Dashboard — {datetime.date.today().strftime('%B %d, %Y')}",
        footer_style
    ))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


# ─── NEW: Auto-detect Indonesian language recommendation ──────────────────────
def get_production_recommendation_id(future_labels, future_forecast, values):
    """Return production recommendation in simple Indonesian."""
    if len(future_forecast) == 0:
        return ""
    arr = np.array(values, dtype=float)
    mean_hist = float(np.mean(arr))
    next_val = float(future_forecast[0])
    next_period = future_labels[0]
    diff = next_val - mean_hist
    pct = (diff / mean_hist * 100) if mean_hist != 0 else 0
    sign = "naik" if pct >= 0 else "turun"
    sign_icon = "📈" if pct >= 0 else "📉"
    rec = (
        f"Berdasarkan model prediksi, perkiraan permintaan pada **{next_period}** adalah "
        f"**{next_val:,.0f} unit** — {sign} **{abs(pct):.1f}%** dari rata-rata historis "
        f"({mean_hist:,.0f} unit). {sign_icon}\n\n"
        f"**Rekomendasi Produksi:** Rencanakan produksi sekitar **{next_val * 1.05:,.0f} unit** "
        f"(+5% buffer stok pengaman) untuk periode {next_period}."
    )
    return rec


# --- MAIN DASHBOARD INTERFACE ---

# Logo + Title Header
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:18px;margin-bottom:4px;">
        <div>
            <h1 style="margin:0;background:linear-gradient(135deg,#4F46E5 0%,#EC4899 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                font-weight:800;font-size:2.3rem;line-height:1.1;">
                Predictive Production Planning for MSMEs
            </h1>
            <p style="margin:4px 0 0 0;color:#64748B;font-weight:500;font-size:1rem;">
                An interactive data science–based analytics application for advanced forecasting
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if not STATSMODELS_AVAILABLE:
    st.warning("⚠️ The statsmodels library is not available. Exponential Smoothing and ARIMA methods will fall back to Naive Forecast.")

with st.sidebar:
    st.header("🔮 Input Settings")
    uploaded_file = st.file_uploader("Upload Historical Data (.csv / .xlsx)", type=["csv", "xlsx"])
    st.divider()

    st.header("⚙️ Evaluation Settings")
    test_percentage = st.slider("Test Data Percentage (%)", min_value=10, max_value=50, value=20, step=5)
    future_horizon = st.number_input("Number of Future Periods", min_value=1, max_value=120, value=6, step=1)
    mode = st.radio("Calculation Mode", ["Single Method", "Compare All Methods"])
    selected_method = st.selectbox("Select Primary Method", list(FORECAST_METHODS.keys()))
    st.divider()

    st.header("🛠️ Additional Parameters")
    st.write("**Exponential Smoothing Parameters**")
    smoothing_mode = st.radio("Tuning Method", ["Automatic Optimization", "Manual Input"])

    if smoothing_mode == "Manual Input":
        alpha_input = st.slider("Alpha (Level)", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
        beta_input = st.slider("Beta (Trend)", min_value=0.01, max_value=0.99, value=0.20, step=0.01)
        gamma_input = st.slider("Gamma (Seasonality)", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    else:
        alpha_input = beta_input = gamma_input = None

    ma_window = st.number_input("Window Moving Average", min_value=2, max_value=24, value=3, step=1)
    wma_weight_text = st.text_input("WMA Weights (Comma Separated)", value="0.2, 0.3, 0.5")
    seasonal_periods = st.number_input("Seasonal Periods (Seasonal)", min_value=2, max_value=52, value=12, step=1)

    st.write("**ARIMA Parameters (p, d, q)**")
    arima_p = st.number_input("Order p (AR)", min_value=0, max_value=5, value=1, step=1)
    arima_d = st.number_input("Order d (I)", min_value=0, max_value=2, value=1, step=1)
    arima_q = st.number_input("Order q (MA)", min_value=0, max_value=5, value=1, step=1)

    process_button = st.button("🚀 Run Process", type="primary")


# ─── ONBOARDING / LANDING PAGE ─────────────────────────────────────────────────
if uploaded_file is None:
    st.info("💡 Instruction: Please upload your Excel or CSV file in the left panel to start the analysis.")

    # Stepper onboarding
    st.markdown("### 🗺️ How to Use This Dashboard")
    steps = [
        ("Step 1 — Upload Data", "📂", "Upload your .csv or .xlsx file using the left panel. Make sure it has a Date column and a numeric Sales/Value column."),
        ("Step 2 — Configure Settings", "⚙️", "Choose the test split %, number of future periods, calculation mode (Single or Compare All), and additional parameters."),
        ("Step 3 — Run Analysis", "🚀", "Click the **Run Process** button to compute forecasts, residuals, accuracy metrics, and projections."),
        ("Step 4 — Download Results", "📥", "Download your results as Excel (.xlsx), HTML charts, or a PDF summary report."),
    ]
    col_s1, col_s2 = st.columns(2)
    for i, (title, icon, desc) in enumerate(steps):
        col = col_s1 if i % 2 == 0 else col_s2
        col.markdown(
            f"""<div class="step-card">
                <span style="font-size:1.5rem;">{icon}</span>
                <strong style="color:#4F46E5;font-size:0.95rem;"> {title}</strong>
                <p style="color:#4A5568;font-size:0.88rem;margin:4px 0 0 0;">{desc}</p>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📋 Example of a Proper Excel/CSV Table Structure")
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#F0F4FF 0%,#EEF2FF 100%);border:1.5px solid #A5B4FC;
            border-radius:10px;padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.3rem;">📥</span>
            <span style="color:#4A5568;font-weight:500;font-size:0.95rem;">
                Download the data template here: &nbsp;
                <a href="https://docs.google.com/spreadsheets/d/1DVghuZxRV-Xj269UMerp_NLRkCuLys5j/edit?usp=sharing&ouid=104043343223917321673&rtpof=true&sd=true"
                   target="_blank"
                   style="color:#4F46E5;font-weight:700;text-decoration:none;background:#E0E7FF;
                          padding:4px 12px;border-radius:6px;border:1px solid #C7D2FE;">
                    📊 Open Template in Google Drive
                </a>
            </span>
        </div>
        """, unsafe_allow_html=True
    )
    sample = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "Sales": [120, 135, 128, 140, 150, 160, 155, 170, 180, 175, 190, 200]
    })
    preview_df = sample.copy()
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.stop()


# ─── FILE LOADING ──────────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}"); st.stop()

if df_raw.empty:
    st.error("The file contains no data."); st.stop()

st.subheader("📊 Uploaded Data Preview")
total_rows = len(df_raw)
st.markdown(
    f"""<div style="background:linear-gradient(135deg,#EEF2FF 0%,#F5F3FF 100%);border-left:4px solid #4F46E5;
        border-radius:8px;padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
        <span style="font-size:1.2rem;">📂</span>
        <span style="color:#3730A3;font-weight:600;font-size:0.95rem;">
            File loaded successfully — <b>{total_rows} rows</b> detected. All rows are used for analysis.
        </span></div>""",
    unsafe_allow_html=True
)
preview_df = df_raw.copy()
if "No" not in preview_df.columns:
    preview_df.insert(0, "No", range(1, len(preview_df) + 1))
st.dataframe(preview_df, use_container_width=True, hide_index=True, height=min(420, 45 + len(preview_df) * 35))

columns = df_raw.columns.tolist()
col1, col2 = st.columns(2)

with col1:
    period_options = ["None"] + columns
    default_period_index = period_options.index("Date") if "Date" in period_options else 0
    period_option = st.selectbox("Select Time/Period Index Column", period_options, index=default_period_index)

with col2:
    default_value_index = columns.index("Sales") if "Sales" in columns else 0
    value_col = st.selectbox("Select Actual Value Column (Numeric)", columns, index=default_value_index)

period_col = None if period_option == "None" else period_option
df = df_raw.copy()

if period_col is not None:
    temp_date = pd.to_datetime(df[period_col], errors="coerce")
    if temp_date.notna().mean() >= 0.7:
        df["_parsed_period"] = temp_date
        df = df.sort_values("_parsed_period").drop(columns=["_parsed_period"])
    else:
        df = df.sort_values(period_col)

# ─── FEATURE 2: Data Quality Check ────────────────────────────────────────────
quality_issues = check_data_quality(df, value_col, period_col)
if quality_issues:
    with st.expander("⚠️ Data Quality Issues Detected — Click to Review", expanded=True):
        for issue in quality_issues:
            color = "#FEF3C7" if issue["severity"] == "warning" else "#FEE2E2"
            border = "#F59E0B" if issue["severity"] == "warning" else "#EF4444"
            st.markdown(
                f"""<div style="background:{color};border-left:4px solid {border};border-radius:6px;
                    padding:8px 14px;margin-bottom:6px;">
                    <b>{issue['type']}</b> — {issue['detail']}
                </div>""",
                unsafe_allow_html=True
            )
else:
    st.success("✅ Data quality check passed — no issues detected.")

values_series = clean_numeric_series(df[value_col])
if len(values_series) < 6:
    st.error("Too few numeric data points detected! Please use at least 6 rows of numeric data.")
    st.stop()

df = df.loc[values_series.index].reset_index(drop=True)
values = values_series.reset_index(drop=True).values
period_labels, period_dates = make_period_labels(df, period_col)

params = {
    "window": int(ma_window), "weights": parse_weights(wma_weight_text), "seasonal_periods": int(seasonal_periods),
    "arima_order": (int(arima_p), int(arima_d), int(arima_q)), "optimized": smoothing_mode == "Automatic Optimization",
    "alpha": alpha_input, "beta": beta_input, "gamma": gamma_input
}

train, test, test_size = split_train_test(values, test_percentage)
train_periods = period_labels[:-test_size]
test_periods = period_labels[-test_size:]

st.subheader("📌 Data Distribution Summary")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Observations", len(values))
metric_col2.metric("Training Dataset", len(train))
metric_col3.metric("Test Dataset (Validation)", len(test))


# ─── NEW: Auto-Recommend Best Method Button ────────────────────────────────────
with st.expander("🚀 Auto-Recommend Best Method", expanded=False):
    st.write("Click below to automatically evaluate all 10 methods and find the one with the lowest MAPE on your data.")
    if st.button("🔍 Find Best Method Automatically"):
        with st.spinner("Evaluating all methods... please wait"):
            params_hashable = tuple(sorted({
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in params.items()
            }.items()))
            best_name_auto, best_mape_auto = auto_recommend_method(
                tuple(train), tuple(test), tuple(test_periods), params_hashable
            )
        mape_lbl, mape_clr, _ = interpret_mape(best_mape_auto)
        st.markdown(
            f"""<div style="background:{mape_clr}18;border-left:5px solid {mape_clr};border-radius:8px;
                padding:10px 18px;margin:8px 0;">
                <b style="color:{mape_clr};font-size:1rem;">✅ Recommended Method: {best_name_auto}</b><br>
                <span style="color:#4A5568;font-size:0.9rem;">MAPE: {best_mape_auto:.2f}% — {mape_lbl}. 
                Select this method in the sidebar and run the process.</span>
            </div>""",
            unsafe_allow_html=True
        )


# --- MAIN TABS ---
tab_data, tab_decomp, tab_grafik = st.tabs(["🔍 Data Characteristics & Trends", "🌊 Seasonality Decomposition", "📊 Forecasting Computation Results"])

with tab_data:
    st.write("### Historical Data Characteristics Analysis")
    c_desc, c_roll = st.columns([1, 2])
    
    with c_desc:
        st.write("**Main Descriptive Statistics**")
        desc_df = pd.DataFrame(values, columns=[value_col]).describe()
        st.dataframe(desc_df, use_container_width=True)
        
    with c_roll:
        st.write("**Trend Movement Detection (Rolling Mean)**")
        roll_df = pd.DataFrame({"Period": period_labels, "Actual": values})
        roll_df["Rolling_Mean"] = roll_df["Actual"].rolling(window=min(3, len(values)), min_periods=1).mean()
        
        roll_fig = go.Figure()
        roll_fig.add_trace(go.Scatter(x=roll_df["Period"], y=roll_df["Actual"], name="Actual", mode="lines", line=dict(color="#4F46E5")))
        roll_fig.add_trace(go.Scatter(x=roll_df["Period"], y=roll_df["Rolling_Mean"], name="Trend Signal (MA)", line=dict(dash='dot', color='#06B6D4', width=2)))
        roll_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
        st.plotly_chart(roll_fig, use_container_width=True)

    # ─── NEW: Outlier visualization ───────────────────────────────────────────
    st.write("---")
    st.write("**🔴 Outlier Detection — Z-score Analysis**")
    box_fig = plot_boxplot_outliers(values, period_labels, value_col)
    st.plotly_chart(box_fig, use_container_width=True)

    # ─── NEW: Seasonal Insight ─────────────────────────────────────────────────
    insight = get_seasonal_insight(values, period_labels)
    if insight:
        st.write("**🌟 Seasonal Demand Insights**")
        ins_c1, ins_c2, ins_c3 = st.columns(3)
        ins_c1.metric("📈 Peak Demand Period", insight["max_period"], f"{insight['max_value']:,.0f} units")
        ins_c2.metric("📉 Lowest Demand Period", insight["min_period"], f"{insight['min_value']:,.0f} units")
        ins_c3.metric("📊 Overall Trend", insight["trend"], f"Avg: {insight['mean_value']:,.0f}")

    # ─── NEW: Manual Data Entry ────────────────────────────────────────────────
    with st.expander("✏️ Edit / Correct Data Values (Manual Input)", expanded=False):
        st.caption("You can directly edit values below. Changes will be used when you re-run the process.")
        editable_df = pd.DataFrame({"Period": period_labels, value_col: values})
        edited_df = st.data_editor(editable_df, use_container_width=True, num_rows="fixed")
        if not np.array_equal(edited_df[value_col].values, values):
            st.info("✏️ Data has been edited. Re-run the process to use updated values.")


with tab_decomp:
    st.write("### 🌊 Time Series Decomposition (Additive)")
    st.write("Breaks down the data into **Trend**, **Seasonal**, and **Residual** components.")
    decomp_fig = plot_decomposition(values, period_labels, int(seasonal_periods))
    if decomp_fig is not None:
        st.plotly_chart(decomp_fig, use_container_width=True)
        decomp_html = fig_to_html_bytes(decomp_fig)
        st.download_button(
            label="📥 Download Decomposition Chart (.html)",
            data=decomp_html,
            file_name="decomposition_chart.html",
            mime="text/html"
        )
    else:
        st.info(f"ℹ️ Not enough data for decomposition. Need at least {2 * int(seasonal_periods)} data points. Currently have {len(values)}.")


with tab_grafik:
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(x=period_labels, y=values, mode="lines+markers", name="Actual Values", line=dict(color="#A855F7", width=3)))
    history_fig.update_layout(title="Historical Time Series Visualization", xaxis_title="Period", yaxis_title="Value", template="plotly_white")
    st.plotly_chart(history_fig, use_container_width=True)

    if process_button:
        if mode == "Single Method":
            with st.spinner(f"Computing {selected_method}..."):
                forecast_test, error_table, metrics = evaluate_one_method(selected_method, train, test, test_periods, params)
                _, used_params = run_forecast_with_params(selected_method, train, len(test), params)
                future_forecast = run_forecast(selected_method, values, int(future_horizon), params)
                future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            residuals = test - forecast_test
            std_error = np.std(residuals)

            st.subheader(f"📊 Analysis Results: {selected_method}")
            with st.container(border=True):
                st.write("**🎯 Accuracy Validation Metrics**")
                m1, m2, m3 = st.columns(3)
                mape_val = f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A"
                m1.metric("Accuracy (MAPE)", mape_val)
                m2.metric("Error (MAD)", f"{metrics['MAD']:.4f}")
                m3.metric("Error (MSE)", f"{metrics['MSE']:.4f}")

            mape_label, mape_color, mape_desc = interpret_mape(metrics['MAPE'])
            st.markdown(
                f"""<div style="background:{mape_color}18;border-left:5px solid {mape_color};border-radius:8px;
                    padding:10px 18px;margin:8px 0;">
                    <span style="font-size:1rem;font-weight:700;color:{mape_color};">
                        📊 Forecast Accuracy Level: {mape_label}
                    </span><br>
                    <span style="color:#4A5568;font-size:0.9rem;">{mape_desc}</span>
                </div>""",
                unsafe_allow_html=True
            )

            if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"]:
                with st.container(border=True):
                    st.write("**⚙️ Optimal Alpha, Beta, Gamma Parameter Values**")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Alpha (Level)", format_param(used_params.get("Alpha")))
                    p2.metric("Beta (Trend)", format_param(used_params.get("Beta")))
                    p3.metric("Gamma (Seasonality)", format_param(used_params.get("Gamma")))

            st.write("") 
            tab1, tab2, tab3 = st.tabs(["📉 Model Evaluation Chart", "🔮 Future Projection Results", "📥 Download Reports"])

            with tab1:
                eval_fig = plot_actual_forecast(test_periods, test, forecast_test, "Validation Test: Actual Data vs Model Estimate")
                st.plotly_chart(eval_fig, use_container_width=True)

                st.download_button(
                    label="📥 Download Evaluation Chart (.html)",
                    data=fig_to_html_bytes(eval_fig),
                    file_name=f"eval_chart_{selected_method}.html",
                    mime="text/html"
                )

                with st.expander("View Error Computation Table Details"):
                    error_table_view = error_table.copy()
                    error_table_view.insert(0, "No", range(1, len(error_table_view) + 1))
                    st.dataframe(error_table_view, use_container_width=True, hide_index=True)

                st.write("#### 📉 Residual Analysis")
                res_time_fig, res_hist_fig = plot_residual_analysis(test_periods, residuals)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.plotly_chart(res_time_fig, use_container_width=True)
                with rc2:
                    st.plotly_chart(res_hist_fig, use_container_width=True)
                st.caption("💡 Ideal residuals: randomly scattered around zero with no clear pattern, and normally distributed in the histogram.")

            with tab2:
                st.write("### 🔮 Future Value Projection")
                proj_fig = plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, std_error)
                st.plotly_chart(proj_fig, use_container_width=True)

                st.download_button(
                    label="📥 Download Projection Chart (.html)",
                    data=fig_to_html_bytes(proj_fig),
                    file_name=f"projection_chart_{selected_method}.html",
                    mime="text/html"
                )
                st.divider()

                col_tabel, col_download = st.columns([2, 1])
                with col_tabel:
                    st.write("**Forecast Result Table**")
                    f_df = pd.DataFrame({"Period": future_labels, "Forecast Result": future_forecast})
                    f_df_view = f_df.copy()
                    f_df_view.insert(0, "No", range(1, len(f_df_view) + 1))
                    st.dataframe(f_df_view, use_container_width=True, hide_index=True)
                
                with col_download:
                    st.write("**Download Results**")
                    excel_data = convert_df_to_excel(f_df)
                    st.download_button(
                        label="📥 Download Results (.xlsx)",
                        data=excel_data,
                        file_name=f"Projection_Results_{selected_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # ─── NEW: Indonesian Recommendation ──────────────────────────────
                rec_text = get_production_recommendation_id(future_labels, future_forecast, values)
                if rec_text:
                    st.divider()
                    st.markdown("**🇮🇩 Rekomendasi Produksi (Bahasa Indonesia)**")
                    st.markdown(
                        f"""<div style="background:#F0FDF4;border-left:4px solid #10B981;border-radius:8px;
                            padding:12px 18px;margin:6px 0;">
                            {rec_text}
                        </div>""",
                        unsafe_allow_html=True
                    )

            with tab3:
                st.write("**Download all results in various formats:**")
                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    # Excel
                    excel_data2 = convert_df_to_excel(f_df)
                    st.download_button(
                        label="📥 Download Projection (.xlsx)",
                        data=excel_data2,
                        file_name=f"Projection_{selected_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with dl_c2:
                    # PDF Report
                    if REPORTLAB_AVAILABLE:
                        pdf_bytes = generate_pdf_report(
                            selected_method, metrics, future_labels, future_forecast,
                            values, period_labels, value_col, mape_label, mape_color, mape_desc,
                            used_params=used_params if selected_method in ["Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"] else None
                        )
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Download PDF Summary Report",
                                data=pdf_bytes,
                                file_name=f"Report_{selected_method}.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.info("💡 Install `reportlab` to enable PDF export: `pip install reportlab`")

        else:
            # Mode: Compare all methods
            with st.spinner("Comparing all 10 methods... please wait"):
                comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
            best_method = comparison_df.iloc[0]["Method"]

            future_forecast = run_forecast(best_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            residuals_best = test - details[best_method]["forecast"]
            std_error_best = np.std(residuals_best)

            st.subheader("🏆 Accuracy Comparison Table for All Methods")
            st.write("Automatically sorted from the model with the lowest error rate (MAPE).")

            comparison_display = comparison_df.copy()
            comparison_display["MAPE Interpretation"] = comparison_display["MAPE"].apply(
                lambda x: interpret_mape(x)[0]
            )
            st.dataframe(comparison_display, use_container_width=True, hide_index=True)

            best_mape = comparison_df.iloc[0]["MAPE"]
            mape_label, mape_color, mape_desc = interpret_mape(best_mape)
            st.markdown(
                f"""<div style="background:{mape_color}18;border-left:5px solid {mape_color};border-radius:8px;
                    padding:10px 18px;margin:8px 0;">
                    <b style="color:{mape_color};">💡 Best Model: {best_method}</b>
                    &nbsp;|&nbsp; MAPE: {best_mape:.2f}% &nbsp;|&nbsp;
                    <span style="color:{mape_color};font-weight:700;">{mape_label}</span><br>
                    <span style="color:#4A5568;font-size:0.9rem;">{mape_desc}</span>
                </div>""",
                unsafe_allow_html=True
            )

            ctab1, ctab2, ctab3, ctab4 = st.tabs(["📊 All Methods Chart", "🔮 Best Method Projection", "📉 Residual (Best)", "📥 Download Reports"])

            with ctab1:
                multi_fig = plot_all_methods_comparison(test_periods, test, details)
                st.plotly_chart(multi_fig, use_container_width=True)
                st.download_button(
                    label="📥 Download Comparison Chart (.html)",
                    data=fig_to_html_bytes(multi_fig),
                    file_name="all_methods_comparison.html",
                    mime="text/html"
                )

            with ctab2:
                proj_fig_cmp = plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, std_error_best)
                st.plotly_chart(proj_fig_cmp, use_container_width=True)
                st.download_button(
                    label="📥 Download Projection Chart (.html)",
                    data=fig_to_html_bytes(proj_fig_cmp),
                    file_name=f"projection_{best_method}.html",
                    mime="text/html"
                )
                proj_df = pd.DataFrame({"Period": future_labels, "Forecast Result": future_forecast})
                proj_df_view = proj_df.copy()
                proj_df_view.insert(0, "No", range(1, len(proj_df_view) + 1))
                st.dataframe(proj_df_view, use_container_width=True, hide_index=True)

                # Indonesian recommendation
                rec_text_cmp = get_production_recommendation_id(future_labels, future_forecast, values)
                if rec_text_cmp:
                    st.divider()
                    st.markdown("**🇮🇩 Rekomendasi Produksi (Bahasa Indonesia)**")
                    st.markdown(
                        f"""<div style="background:#F0FDF4;border-left:4px solid #10B981;border-radius:8px;
                            padding:12px 18px;margin:6px 0;">
                            {rec_text_cmp}
                        </div>""",
                        unsafe_allow_html=True
                    )

            with ctab3:
                st.write(f"#### 📉 Residual Analysis — {best_method}")
                res_t, res_h = plot_residual_analysis(test_periods, residuals_best)
                rcc1, rcc2 = st.columns(2)
                with rcc1:
                    st.plotly_chart(res_t, use_container_width=True)
                with rcc2:
                    st.plotly_chart(res_h, use_container_width=True)
                st.caption("💡 Ideal residuals: randomly scattered around zero with no clear pattern.")

            with ctab4:
                st.write("**Download all results in one Excel file with separate sheets per method:**")
                excel_full = convert_full_comparison_excel(comparison_df, details, future_labels, future_forecast, best_method)
                dl4_c1, dl4_c2 = st.columns(2)
                with dl4_c1:
                    st.download_button(
                        label="📥 Download Full Report — All Error Tables (.xlsx)",
                        data=excel_full,
                        file_name="Full_Forecasting_Comparison_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.caption("📋 Sheets: Method Comparison, Best Projection, + individual error table per method.")
                with dl4_c2:
                    if REPORTLAB_AVAILABLE:
                        best_metrics = details[best_method]["metrics"]
                        pdf_bytes_cmp = generate_pdf_report(
                            best_method, best_metrics, future_labels, future_forecast,
                            values, period_labels, value_col, mape_label, mape_color, mape_desc
                        )
                        if pdf_bytes_cmp:
                            st.download_button(
                                label="📄 Download PDF Summary Report",
                                data=pdf_bytes_cmp,
                                file_name=f"Report_{best_method}.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.info("💡 Install `reportlab` to enable PDF export: `pip install reportlab`")
