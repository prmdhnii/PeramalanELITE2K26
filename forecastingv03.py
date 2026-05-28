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


# --- PLOTLY CHART FUNCTIONS (COLORFUL & VIBRANT THESIS STYLE) ---

def plot_actual_forecast(periods, actual, forecast, title):
    fig = go.Figure()
    # Actual Color: Bright Indigo
    fig.add_trace(go.Scatter(x=periods, y=actual, mode="lines+markers", name="Actual", line=dict(color='#4F46E5', width=3)))
    # Forecast Color: Bright Pink/Neon Red
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode="lines+markers", name="Forecast", line=dict(color='#EC4899', width=3)))
    fig.update_layout(title=title, xaxis_title="Period", yaxis_title="Value", hovermode="x unified", template="plotly_white")
    return fig


def plot_future_forecast_with_ci(all_periods, actual_values, future_periods, future_forecast, residual_std=0):
    fig = go.Figure()
    
    # Historical (Vibrant Indigo)
    fig.add_trace(go.Scatter(x=all_periods, y=actual_values, mode="lines+markers", name="Historical Actual", line=dict(color='#4F46E5', width=3)))
    
    # Confidence Interval (Soft Pink Transparent)
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

    # Future Forecast Line (Vibrant Dashed Pink)
    fig.add_trace(go.Scatter(x=future_periods, y=future_forecast, mode="lines+markers", name="Main Projection", line=dict(color='#EC4899', width=3, dash='dash')))

    fig.update_layout(
        title="Future Value Projection Chart",
        xaxis_title="Period",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white"
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


def convert_all_to_excel(comparison_df, best_method_name, future_labels, future_forecast):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, index=False, sheet_name='Method_Comparison')
        
        best_df = pd.DataFrame({"Period": future_labels, "Main Forecast": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name='Best_Method_Projection')
        
        workbook  = writer.book
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4F46E5', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
        num_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        # Format Sheet 1
        ws1 = writer.sheets['Method_Comparison']
        ws1.set_row(0, 24)
        for col_num, value in enumerate(comparison_df.columns.values):
            ws1.write(0, col_num, value, header_format)
            
        for i, col in enumerate(comparison_df.columns):
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            if col == "Method":
                ws1.set_column(i, i, max_len, text_format)
            else:
                ws1.set_column(i, i, max_len, num_format)
                
        # Format Sheet 2
        ws2 = writer.sheets['Best_Method_Projection']
        ws2.set_row(0, 24)
        for col_num, value in enumerate(best_df.columns.values):
            ws2.write(0, col_num, value, header_format)
            
        for i, col in enumerate(best_df.columns):
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            if col == "Period":
                ws2.set_column(i, i, max_len, text_format)
            else:
                ws2.set_column(i, i, max_len, num_format)
            
    return output.getvalue()


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Projection_Results')
        
        workbook  = writer.book
        worksheet = writer.sheets['Projection_Results']
        worksheet.set_row(0, 24)
        
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4F46E5', 'font_color': '#FFFFFF', 
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })
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



# ─── FEATURE 4: MAPE Interpretation ───────────────────────────────────────────
def interpret_mape(mape_value):
    """Return (label, color, description) based on Lewis scale."""
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

    # Missing values
    missing = df[value_col].isna().sum()
    if missing > 0:
        issues.append({"type": "⚠️ Missing Values", "detail": f"{missing} missing value(s) in '{value_col}'", "severity": "warning"})

    # Negative values
    neg = (pd.to_numeric(df[value_col], errors="coerce") < 0).sum()
    if neg > 0:
        issues.append({"type": "⚠️ Negative Values", "detail": f"{neg} negative value(s) detected — may affect forecast accuracy", "severity": "warning"})

    # Outlier detection (Z-score)
    numeric_vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(numeric_vals) >= 4:
        z_scores = np.abs((numeric_vals - numeric_vals.mean()) / numeric_vals.std())
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            issues.append({"type": "🔴 Outliers Detected", "detail": f"{outliers} extreme outlier(s) (Z-score > 3) found in value column", "severity": "error"})

    # Duplicate dates
    if period_col is not None:
        dupes = df[period_col].duplicated().sum()
        if dupes > 0:
            issues.append({"type": "⚠️ Duplicate Periods", "detail": f"{dupes} duplicate period/date value(s) found", "severity": "warning"})

    return issues


# ─── FEATURE 1 & 3: Multi-method comparison chart ──────────────────────────────
def plot_all_methods_comparison(test_periods, test, details_dict):
    """Plot actual vs all method forecasts on one chart."""
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
    """Return two figures: residual over time + histogram."""
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
    """Simple additive decomposition: trend + seasonal + residual."""
    n = len(values)
    if n < 2 * seasonal_periods:
        return None

    # Trend via centered moving average
    half = seasonal_periods // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.mean(values[i - half: i + half + 1])

    # Seasonal component
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


# ─── FEATURE 8: Export chart as HTML (downloadable) ───────────────────────────
def fig_to_html_bytes(fig):
    import io as _io
    buf = _io.StringIO()
    fig.write_html(buf)
    return buf.getvalue().encode("utf-8")


# ─── FEATURE 3: Full comparison Excel export with all error tables ─────────────
def convert_full_comparison_excel(comparison_df, details_dict, future_labels, future_forecast, best_method):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#4F46E5", "font_color": "#FFFFFF", "border": 1, "align": "center", "valign": "vcenter"})
        num_fmt    = workbook.add_format({"num_format": "#,##0.00", "border": 1, "align": "right"})
        text_fmt   = workbook.add_format({"border": 1, "align": "left"})
        good_fmt   = workbook.add_format({"bg_color": "#D1FAE5", "border": 1, "align": "left", "bold": True})

        # Sheet 1 – Method Comparison
        comparison_df.to_excel(writer, index=False, sheet_name="Method_Comparison")
        ws1 = writer.sheets["Method_Comparison"]
        ws1.set_row(0, 24)
        for ci, col in enumerate(comparison_df.columns):
            ws1.write(0, ci, col, header_fmt)
            max_len = max(comparison_df[col].astype(str).map(len).max(), len(col)) + 4
            ws1.set_column(ci, ci, max_len, text_fmt if col == "Method" else num_fmt)
        # Highlight best row
        best_row_idx = comparison_df[comparison_df["Method"] == best_method].index
        if len(best_row_idx):
            excel_row = int(best_row_idx[0]) + 1
            for ci in range(len(comparison_df.columns)):
                ws1.write(excel_row, ci, comparison_df.iloc[int(best_row_idx[0]), ci], good_fmt)

        # Sheet 2 – Best Method Projection
        best_df = pd.DataFrame({"Period": future_labels, "Forecast": future_forecast})
        best_df.to_excel(writer, index=False, sheet_name="Best_Projection")
        ws2 = writer.sheets["Best_Projection"]
        ws2.set_row(0, 24)
        for ci, col in enumerate(best_df.columns):
            ws2.write(0, ci, col, header_fmt)
            max_len = max(best_df[col].astype(str).map(len).max(), len(col)) + 5
            ws2.set_column(ci, ci, max_len, text_fmt if col == "Period" else num_fmt)

        # Sheets 3+ – Error detail per method
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

# --- MAIN DASHBOARD INTERFACE ---

# --- Logo + Title Header ---
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:18px;margin-bottom:4px;">
        <img src="data:image/png;base64,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"
             style="width:72px;height:72px;object-fit:contain;flex-shrink:0;" />
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


if uploaded_file is None:
    st.info("💡 Instruction: Please upload your Excel or CSV file in the left panel to start the analysis.")
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

# ── FEATURE 2: Data Quality Check ──────────────────────────────────────
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


# --- TAB GRID & MAIN PROCESS IMPLEMENTATION AREA ---

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

            # ── FEATURE 4: MAPE Interpretation ──────────────────────────────
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
            tab1, tab2 = st.tabs(["📉 Model Evaluation Chart", "🔮 Future Projection Results"])

            with tab1:
                eval_fig = plot_actual_forecast(test_periods, test, forecast_test, "Validation Test: Actual Data vs Model Estimate")
                st.plotly_chart(eval_fig, use_container_width=True)

                # ── FEATURE 8: Download chart ─────────────────────────────
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

                # ── FEATURE 5: Residual Analysis ──────────────────────────
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

                # ── FEATURE 8: Download projection chart ──────────────────
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

        else:
            # Mode: Compare all methods
            comparison_df, details = evaluate_all_methods(train, test, test_periods, params)
            best_method = comparison_df.iloc[0]["Method"]

            future_forecast = run_forecast(best_method, values, int(future_horizon), params)
            future_labels = make_future_labels(period_dates, period_labels, int(future_horizon))

            # ── FEATURE 1: Multi-method comparison chart ──────────────────
            st.subheader("🏆 Accuracy Comparison Table for All Methods")
            st.write("Automatically sorted from the model with the lowest error rate (MAPE).")

            # Add MAPE interpretation column to table
            comparison_display = comparison_df.copy()
            comparison_display["MAPE Interpretation"] = comparison_display["MAPE"].apply(
                lambda x: interpret_mape(x)[0]
            )
            st.dataframe(comparison_display, use_container_width=True, hide_index=True)

            # ── FEATURE 4: Best method MAPE interpretation ─────────────────
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

            # Tabs for compare mode
            ctab1, ctab2, ctab3 = st.tabs(["📊 All Methods Chart", "🔮 Best Method Projection", "📥 Download Reports"])

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
                proj_fig_cmp = plot_future_forecast_with_ci(period_labels, values, future_labels, future_forecast, 0)
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

            with ctab3:
                st.write("**Download all results in one Excel file with separate sheets per method:**")
                # ── FEATURE 3: Full comparison export ──────────────────────
                excel_full = convert_full_comparison_excel(comparison_df, details, future_labels, future_forecast, best_method)
                st.download_button(
                    label="📥 Download Full Report — All Error Tables (.xlsx)",
                    data=excel_full,
                    file_name="Full_Forecasting_Comparison_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.caption("📋 Sheets included: Method Comparison summary, Best Method Projection, + individual error table for each of the 10 methods.")
