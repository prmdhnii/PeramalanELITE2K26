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
    /* 1. Global Background & Fonts */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* 2. Top Title Banner Styling */
    .title-banner {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .title-banner h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }
    .title-banner p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* 3. Section Headers */
    .section-header {
        color: #1E3A8A;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }

    /* 4. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
        padding: 2rem 1rem;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* 5. Custom Metric Cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
    }

    /* 6. Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E3A8A;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1E3A8A !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    /* 7. Footer Styling */
    .footer-container {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid #E2E8F0;
        background-color: #FFFFFF;
        border-radius: 12px 12px 0 0;
    }
    .footer-text {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .footer-highlight {
        color: #3B82F6;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


def render_title_banner():
    st.markdown("""
        <div class="title-banner">
            <h1>📈 Forecasting Dashboard Elementary Laboratory</h1>
            <p>Sistem Komparasi Otomatis & Analisis Mendalam 10 Metode Peramalan Time Series</p>
        </div>
    """, unsafe_allow_html=True)


def render_section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_custom_metric(title, value, color_hex="#1E293B"):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {color_hex};">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_credit_footer():
    st.markdown("""
        <div class="footer-container">
            <p class="footer-text">
                © 2026 Dashboard Peramalan Multi-Metode | Dikembangkan untuk 
                <span class="footer-highlight">Elementary Laboratory</span>
            </p>
            <p style="font-size: 0.8rem; color: #94A3B8; margin-top: -0.5rem;">
                Mendukung ekspor multi-format (Excel, CSV, PDF) dengan visualisasi interaktif tingkat tinggi.
            </p>
        </div>
    """, unsafe_allow_html=True)


# --- DICTIONARY INFORMASI METODE ---
METHOD_INFO = {
    "Naive Forecast": "Metode paling sederhana yang menggunakan nilai aktual periode terakhir sebagai hasil ramalan untuk periode berikutnya tanpa penyesuaian atau tren.",
    "Moving Average": "Menghitung rata-rata dari sejumlah n data historis terakhir secara konstan untuk memperhalus fluktuasi jangka pendek.",
    "Weighted Moving Average": "Mirip dengan Moving Average, namun memberikan bobot matematis yang lebih besar pada data yang lebih baru/terkini.",
    "Single Exponential Smoothing": "Metode penghalusan eksponensial untuk data stasioner (tanpa tren/musiman) menggunakan konstanta perataan alpha (α).",
    "Double Exponential Smoothing": "Metode Holt Linear untuk data yang mengandung komponen tren linier tanpa pola musiman, dikontrol parameter alpha (α) dan beta (β).",
    "Triple Exponential Smoothing": "Metode Holt-Winters untuk data kompleks yang mengandung komponen tren sekaligus musiman berkelanjutan (alpha, beta, gamma).",
    "Linear Trend Line": "Pendekatan kuadrat terkecil untuk memetakan tren linear garis lurus jangka panjang melintasi waktu historis.",
    "Least Square Quadratic": "Pendekatan regresi polinomial derajat dua untuk memetakan pola pergerakan data yang mengalami kurvatura melengkung.",
    "Seasonal Naive": "Metode peramalan musiman sederhana yang mendasarkan nilai ramalan pada nilai aktual di siklus musiman periode yang sama tahun sebelumnya.",
    "ARIMA": "Model Autoregressive Integrated Moving Average (p,d,q) yang mengombinasikan hubungan lag dependen dan sisa galat untuk pola data kompleks."
}


# --- KODE PERBAIKAN OUTPUT ALGORITMA PERAMALAN (FIXED) ---

def forecast_naive(history, horizon, **kwargs):
    if len(history) == 0: return np.zeros(horizon)
    return np.repeat(history[-1], horizon)

def forecast_moving_average(history, horizon, window=3, **kwargs):
    if len(history) == 0: return np.zeros(horizon)
    usable_window = min(window, len(history))
    pred = np.mean(history[-usable_window:])
    return np.repeat(pred, horizon)

def forecast_weighted_moving_average(history, horizon, weights=None, **kwargs):
    if weights is None: weights = [0.2, 0.3, 0.5]
    if len(history) == 0: return np.zeros(horizon)
    usable_window = min(len(weights), len(history))
    recent_values = np.array(history[-usable_window:], dtype=float)
    recent_weights = np.array(weights[-usable_window:], dtype=float)
    recent_weights = recent_weights / recent_weights.sum()
    pred = np.sum(recent_values * recent_weights)
    return np.repeat(pred, horizon)

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
        else:
            alpha_used = limit_smoothing_param(alpha)
        fitted = model.fit(smoothing_level=alpha_used)
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
            beta_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_trend"]))
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
        fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used)
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
            beta_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_trend"]))
            gamma_used = limit_smoothing_param(get_fitted_param(fitted_auto, ["smoothing_seasonal"]))
        else:
            alpha_used = limit_smoothing_param(alpha)
            beta_used = limit_smoothing_param(beta)
            gamma_used = limit_smoothing_param(gamma)
        fitted = model.fit(smoothing_level=alpha_used, smoothing_trend=beta_used, smoothing_seasonal=gamma_used)
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
    forecasts = []
    n = len(history)
    for i in range(horizon):
        idx = (n - seasonal_periods + (i % seasonal_periods)) % n
        forecasts.append(history[idx])
    return np.array(forecasts)

def forecast_arima(history, horizon, arima_order=(1, 1, 1), **kwargs):
    if not STATSMODELS_AVAILABLE or len(history) < 8: return forecast_naive(history, horizon)
    try:
        model = ARIMA(history, order=arima_order)
        fitted_model = model.fit()
        return np.array(fitted_model.forecast(steps=horizon))
    except Exception:
        return forecast_naive(history, horizon)


# --- ERROR CALCULATION & EVALUATION TABLE ---

def safe_mape(actual, predicted):
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = actual != 0
    if not np.any(mask): return 0.0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def calculate_error_table(actuals, fitted_values):
    actuals, fitted_values = np.array(actuals, dtype=float), np.array(fitted_values, dtype=float)
    df_err = pd.DataFrame({
        "Periode": np.arange(1, len(actuals) + 1),
        "Aktual (Yt)": actuals,
        "Prediksi (Ft)": fitted_values
    })
    df_err["Error (et)"] = df_err["Aktual (Yt)"] - df_err["Prediksi (Ft)"]
    df_err["|et| (Absolute)"] = df_err["Error (et)"].abs()
    df_err["et² (Squared)"] = df_err["Error (et)"] ** 2
    
    mapes = []
    for a, p in zip(actuals, fitted_values):
        if a == 0: mapes.append(0.0)
        else: mapes.append((abs(a - p) / abs(a)) * 100)
    df_err["|et/Yt| % (MAPE)"] = mapes
    
    mad = df_err["|et| (Absolute)"].mean()
    mse = df_err["et² (Squared)"].mean()
    mape = safe_mape(actuals, fitted_values)
    
    return df_err, {"MAD": mad, "MSE": mse, "MAPE": mape}

def run_backtest_fitted(history, method_name, params_dict):
    n = len(history)
    fitted = np.zeros(n)
    fitted[0] = history[0]
    
    for t in range(1, n):
        past = history[:t]
        if method_name == "Naive Forecast":
            f_val = forecast_naive(past, 1)[0]
        elif method_name == "Moving Average":
            f_val = forecast_moving_average(past, 1, window=params_dict['ma_window'])[0]
        elif method_name == "Weighted Moving Average":
            f_val = forecast_weighted_moving_average(past, 1, weights=params_dict['wma_weights'])[0]
        elif method_name == "Single Exponential Smoothing":
            f_val, _ = forecast_single_exponential_smoothing(past, 1, optimized=params_dict['ses_opt'], alpha=params_dict['ses_alpha'])
            f_val = f_val[0] if len(f_val) > 0 else past[-1]
        elif method_name == "Double Exponential Smoothing":
            f_val, _ = forecast_double_exponential_smoothing(past, 1, optimized=params_dict['des_opt'], alpha=params_dict['des_alpha'], beta=params_dict['des_beta'])
            f_val = f_val[0] if len(f_val) > 0 else past[-1]
        elif method_name == "Triple Exponential Smoothing":
            f_val, _ = forecast_triple_exponential_smoothing(past, 1, seasonal_periods=params_dict['tes_period'], optimized=params_dict['tes_opt'], alpha=params_dict['tes_alpha'], beta=params_dict['tes_beta'], gamma=params_dict['tes_gamma'])
            f_val = f_val[0] if len(f_val) > 0 else past[-1]
        elif method_name == "Linear Trend Line":
            f_val = forecast_linear_trend(past, 1)[0]
        elif method_name == "Least Square Quadratic":
            f_val = forecast_least_square_quadratic(past, 1)[0]
        elif method_name == "Seasonal Naive":
            f_val = forecast_seasonal_naive(past, 1, seasonal_periods=params_dict['snaive_period'])[0]
        elif method_name == "ARIMA":
            f_val = forecast_arima(past, 1, arima_order=params_dict['arima_order'])[0]
        else:
            f_val = past[-1]
        fitted[t] = f_val
    return fitted


# --- EXCEL & PDF GENERATORS ---

def generate_excel_report(details, best_method, future_labels, future_forecast, actuals, labels):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_data = []
        for m, d in details.items():
            summary_data.append({
                "Metode": m,
                "MAD": d["metrics"]["MAD"],
                "MSE": d["metrics"]["MSE"],
                "MAPE (%)": d["metrics"]["MAPE"],
                "Status": "🏆 TERBAIK (Key)" if m == best_method else "-"
            })
        df_sum = pd.DataFrame(summary_data).sort_values(by="MAPE (%)")
        df_sum.to_excel(writer, sheet_name="Ringkasan Komparasi", index=False)
        
        df_proj = pd.DataFrame({
            "Label Periode Depan": future_labels,
            f"Proyeksi ({best_method})": future_forecast
        })
        df_proj.to_excel(writer, sheet_name="Proyeksi Terbaik Jangka Panjang", index=False)
        
        for m, d in details.items():
            sheet_title = m.replace(" ", "_")[:31]
            d["error_table"].to_excel(writer, sheet_name=sheet_title, index=False)
            
    return output.getvalue()

def generate_pdf_report(method_name, metrics, future_labels, future_forecast, actuals, labels, val_col, mape_label, mape_color, mape_desc):
    if not REPORTLAB_AVAILABLE: return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=22, leading=26, textColor=colors.HexColor("#1E3A8A"), alignment=TA_CENTER)
        h2_style = ParagraphStyle('SectionH2', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor("#1D4ED8"), spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor("#334155"))
        center_text = ParagraphStyle('CenterTxt', parent=body_style, alignment=TA_CENTER)
        bold_center = ParagraphStyle('BoldCenterTxt', parent=body_style, fontName='Helvetica-Bold', alignment=TA_CENTER)
        
        elements = []
        elements.append(Paragraph("<b>LAPORAN RINGKAS ANALISIS PERAMALAN</b>", title_style))
        elements.append(Paragraph("Dashboard Analisis Otomatis — Elementary Laboratory", center_text))
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#CBD5E1"), spaceBefore=1, spaceAfter=15))
        
        intro_p = f"Dokumen ini memuat ringkasan hasil analisis komparasi otomatis. Metode peramalan terpilih yang menghasilkan nilai galat terkecil untuk kolom data <b>{val_col}</b> adalah <b>{method_name}</b>."
        elements.append(Paragraph(intro_p, body_style))
        elements.append(Spacer(1, 0.3 * cm))
        
        elements.append(Paragraph("Evaluasi Akurasi Model (Histori Kebelakang)", h2_style))
        met_data = [
            [Paragraph("<b>Metrik Evaluasi Galat</b>", bold_center), Paragraph("<b>Nilai Hitung (Skor)</b>", bold_center), Paragraph("<b>Kategori Akurasi</b>", bold_center)],
            [Paragraph("MAD (Mean Absolute Deviation)", body_style), Paragraph(f"{metrics['MAD']:.4f}", center_text), Paragraph("-", center_text)],
            [Paragraph("MSE (Mean Squared Error)", body_style), Paragraph(f"{metrics['MSE']:.4f}", center_text), Paragraph("-", center_text)],
            [Paragraph("MAPE (Mean Absolute Percentage Error)", body_style), Paragraph(f"<b>{metrics['MAPE']:.2f}%</b>", center_text), Paragraph(f"<font color='{mape_color}'><b>{mape_label}</b></font>", center_text)]
        ]
        t_met = Table(met_data, colWidths=[6.5*cm, 4.5*cm, 5.5*cm])
        t_met.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(t_met)
        elements.append(Spacer(1, 0.2 * cm))
        elements.append(Paragraph(f"<i>Keterangan Kriteria MAPE: {mape_desc}</i>", ParagraphStyle('Sub', parent=body_style, fontSize=8, textColor=colors.HexColor("#64748B"))))
        
        elements.append(Paragraph("Tabel Proyeksi Nilai Masa Depan (Horizon)", h2_style))
        proj_rows = [[Paragraph("<b>Periode Depan (Label)</b>", bold_center), Paragraph("<b>Nilai Hasil Proyeksi</b>", bold_center)]]
        for lbl, val in zip(future_labels, future_forecast):
            proj_rows.append([Paragraph(str(lbl), center_text), Paragraph(f"<b>{val:.4f}</b>", center_text)])
            
        t_proj = Table(proj_rows, colWidths=[8*cm, 8.5*cm])
        t_proj.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EFF6FF")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DBEAFE")),
            ('PADDING', (0,0), (-1,-1), 5),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        elements.append(t_proj)
        
        elements.append(Spacer(1, 0.8 * cm))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceBefore=1, spaceAfter=10))
        elements.append(Paragraph("Laporan ini dibuat otomatis oleh sistem cerdas berbasis Streamlit Web Engine.", ParagraphStyle('Foot', parent=body_style, fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor("#94A3B8"))))
        
        doc.build(elements)
        return buffer.getvalue()
    except Exception:
        return None


# --- APPLICATION LAYOUT ---

render_title_banner()

st.sidebar.markdown("## 📊 Pengaturan Data & File")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset (Excel atau CSV)", type=["csv", "xlsx"])

# DATASET DEFAULT INTERNAL (Bila user belum unggah file)
if uploaded_file is None:
    st.sidebar.info("💡 Menampilkan dataset bawaan (Contoh Pola Tren & Musiman Bulanan)")
    months = ["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "Mei 2024", "Jun 2024",
              "Jul 2024", "Agu 2024", "Sep 2024", "Okt 2024", "Nov 2024", "Des 2024",
              "Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "Mei 2025", "Jun 2025",
              "Jul 2025", "Agu 2025", "Sep 2025", "Okt 2025", "Nov 2025", "Des 2025"]
    simulated_values = [120, 135, 160, 155, 190, 220, 250, 240, 210, 185, 150, 175,
                        160, 185, 210, 200, 245, 290, 330, 315, 280, 245, 200, 230]
    df = pd.DataFrame({"Bulan": months, "Jumlah Permintaan": simulated_values})
    label_col = "Bulan"
    value_col = "Jumlah Permintaan"
else:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ File berhasil dimuat!")
        
        all_cols = list(df.columns)
        label_col = st.sidebar.selectbox("Pilih Kolom Label / Waktu (X):", all_cols, index=0)
        value_col = st.sidebar.selectbox("Pilih Kolom Nilai Aktual (Y):", all_cols, index=min(1, len(all_cols)-1))
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

# VALIDASI DATA
df = df.dropna(subset=[label_col, value_col])
values = df[value_col].astype(float).tolist()
period_labels = df[label_col].astype(str).tolist()

if len(values) < 5:
    st.error("❌ Jumlah baris data terlalu sedikit (minimal butuh 5 baris data aktual untuk analisis multi-metode).")
    st.stop()

st.sidebar.markdown("## ⚙️ Parameter Peramalan")
horizon = st.sidebar.number_input("Horizon Peramalan (Periode ke depan):", min_value=1, max_value=48, value=6)

with st.sidebar.expander("🛠️ Set Detail Parameter Manual"):
    ma_window = st.number_input("Moving Average Window:", min_value=2, max_value=len(values)-1, value=3)
    wma_w_str = st.text_input("WMA Weights (pisahkan koma):", value="0.2,0.3,0.5")
    try: wma_weights = [float(x.strip()) for x in wma_w_str.split(",")]
    except Exception: wma_weights = [0.2, 0.3, 0.5]
    
    st.markdown("---")
    ses_opt = st.checkbox("Optimasi Otomatis SES", value=True)
    ses_alpha = st.slider("Alpha Manual SES:", 0.01, 0.99, 0.20, disabled=ses_opt)
    
    st.markdown("---")
    des_opt = st.checkbox("Optimasi Otomatis DES", value=True)
    des_alpha = st.slider("Alpha Manual DES:", 0.01, 0.99, 0.20, disabled=des_opt)
    des_beta = st.slider("Beta Manual DES:", 0.01, 0.99, 0.10, disabled=des_opt)
    
    st.markdown("---")
    tes_period = st.number_input("Periode Musiman TES:", min_value=2, max_value=len(values)//2, value=12)
    tes_opt = st.checkbox("Optimasi Otomatis TES", value=True)
    tes_alpha = st.slider("Alpha Manual TES:", 0.01, 0.99, 0.20, disabled=tes_opt)
    tes_beta = st.slider("Beta Manual TES:", 0.01, 0.99, 0.10, disabled=tes_opt)
    tes_gamma = st.slider("Gamma Manual TES:", 0.01, 0.99, 0.20, disabled=tes_opt)
    
    st.markdown("---")
    snaive_period = st.number_input("Periode Musiman Seasonal Naive:", min_value=2, max_value=len(values)-1, value=12)
    
    st.markdown("---")
    p_arima = st.number_input("ARIMA p (Autoregressive):", 0, 5, 1)
    d_arima = st.number_input("ARIMA d (Differencing):", 0, 2, 1)
    q_arima = st.number_input("ARIMA q (Moving Average):", 0, 5, 1)
    arima_order = (p_arima, d_arima, q_arima)

params_dict = {
    'ma_window': ma_window, 'wma_weights': wma_weights,
    'ses_opt': ses_opt, 'ses_alpha': ses_alpha,
    'des_opt': des_opt, 'des_alpha': des_alpha, 'des_beta': des_beta,
    'tes_period': tes_period, 'tes_opt': tes_opt, 'tes_alpha': tes_alpha, 'tes_beta': tes_beta, 'tes_gamma': tes_gamma,
    'snaive_period': snaive_period, 'arima_order': arima_order
}


# --- PROSES PERHITUNGAN UTAMA (ALL 10 METHODS) ---
methods_list = [
    "Naive Forecast", "Moving Average", "Weighted Moving Average",
    "Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing",
    "Linear Trend Line", "Least Square Quadratic", "Seasonal Naive", "ARIMA"
]

details = {}
best_method = None
min_mape = float('inf')

for method in methods_list:
    fitted = run_backtest_fitted(values, method, params_dict)
    err_df, metrics = calculate_error_table(values, fitted)
    
    # Kalkulasi nilai proyeksi ke depan jangka panjang
    if method == "Naive Forecast":
        fut = forecast_naive(values, horizon)
        p_info = {}
    elif method == "Moving Average":
        fut = forecast_moving_average(values, horizon, window=ma_window)
        p_info = {}
    elif method == "Weighted Moving Average":
        fut = forecast_weighted_moving_average(values, horizon, weights=wma_weights)
        p_info = {}
    elif method == "Single Exponential Smoothing":
        fut, p_info = forecast_single_exponential_smoothing(values, horizon, optimized=ses_opt, alpha=ses_alpha)
    elif method == "Double Exponential Smoothing":
        fut, p_info = forecast_double_exponential_smoothing(values, horizon, optimized=des_opt, alpha=des_alpha, beta=des_beta)
    elif method == "Triple Exponential Smoothing":
        fut, p_info = forecast_triple_exponential_smoothing(values, horizon, seasonal_periods=tes_period, optimized=tes_opt, alpha=tes_alpha, beta=tes_beta, gamma=tes_gamma)
    elif method == "Linear Trend Line":
        fut = forecast_linear_trend(values, horizon)
        p_info = {}
    elif method == "Least Square Quadratic":
        fut = forecast_least_square_quadratic(values, horizon)
        p_info = {}
    elif method == "Seasonal Naive":
        fut = forecast_seasonal_naive(values, horizon, seasonal_periods=snaive_period)
        p_info = {}
    elif method == "ARIMA":
        fut = forecast_arima(values, horizon, arima_order=arima_order)
        p_info = {}
        
    details[method] = {
        "fitted": fitted,
        "error_table": err_df,
        "metrics": metrics,
        "future": fut,
        "params": p_info
    }
    
    if metrics["MAPE"] < min_mape:
        min_mape = metrics["MAPE"]
        best_method = method


# --- GENERASI LABEL MASA DEPAN (HORIZON) ---
future_labels = []
last_label = period_labels[-1]
try:
    # Upayakan deteksi jika bertipe angka tahun/integer biasa
    last_num = int(last_label)
    for i in range(1, horizon + 1):
        future_labels.append(str(last_num + i))
except Exception:
    try:
        # Upayakan jika berbasis parse penanggalan
        last_date = pd.to_datetime(last_label)
        for i in range(1, horizon + 1):
            # Asumsi bulanan jika string mengandung spasi / nama bulan
            if " " in last_label or "-" in last_label:
                next_d = last_date + pd.DateOffset(months=i)
                future_labels.append(next_d.strftime("%b %Y"))
            else:
                next_d = last_date + pd.DateOffset(days=i)
                future_labels.append(next_d.strftime("%Y-%m-%d"))
    except Exception:
        # Fallback teks biasa + indeks increment h1, h2...
        for i in range(1, horizon + 1):
            future_labels.append(f"H+{i} ({last_label})")


# --- RINGKASAN METODE TERBAIK UTAMA ---
best_metrics = details[best_method]["metrics"]
m_mape = best_metrics["MAPE"]

if m_mape < 10:
    mape_label, mape_color, mape_desc = "SANGAT AKURAT", "#10B981", "Model memiliki performa luar biasa, galat di bawah 10%."
elif m_mape < 20:
    mape_label, mape_color, mape_desc = "BAIK / AKURAT", "#3B82F6", "Model andal dan layak digunakan untuk keputusan operasional."
elif m_mape < 50:
    mape_label, mape_color, mape_desc = "WASPADA / CUKUP", "#F59E0B", "Akurasi moderat. Pertimbangkan penyesuaian parameter atau metode lain."
else:
    mape_label, mape_color, mape_desc = "TIDAK AKURAT", "#EF4444", "Tingkat galat ekstrem tingginya (di atas 50%). Hasil proyeksi sangat berisiko."

render_section_header("🎯 Ringkasan Analisis Metode Terbaik Otomatis")
c_m1, c_m2, c_m3, c_m4 = st.columns(4)
with c_m1: render_custom_metric("Metode Terpilih (Kunci Akurasi)", best_method, "#1D4ED8")
with c_m2: render_custom_metric("MAD Terkecil", f"{best_metrics['MAD']:.4f}", "#1E293B")
with c_m3: render_custom_metric("MSE Terkecil", f"{best_metrics['MSE']:.4f}", "#1E293B")
with c_m4: render_custom_metric("MAPE Optimum", f"{best_metrics['MAPE']:.2f}%", mape_color)

st.markdown(f"""
<div style='background-color: white; padding: 1rem 1.5rem; border-radius: 12px; border-left: 5px solid {mape_color}; margin-top: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
    <span style='font-weight: 800; color: {mape_color};'>STATUS EVALUASI KINERJA: {mape_label}</span> — {mape_desc}
</div>
""", unsafe_allow_html=True)


# --- MAIN INTERACTIVE TABS SYSTEM ---
st.write("")
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Ringkasan Komparasi Semua Metode",
    "🔮 Proyeksi Masa Depan (Horizon Forecast)",
    "📑 Detail Hitung Galat & Nilai Alpha/Beta",
    "📊 Dataset Riwayat (Input)"
])

# ─── TAB 1: RINGKASAN KOMPARASI KESELURUHAN ──────────────────────────────────
with tab1:
    st.markdown("### 📊 Peringkat Performa Akurasi Metode Peramalan")
    st.write("Semua metode dijalankan secara *backtesting* satu langkah ke depan (*one-step-ahead fitted values*) untuk menghitung akurasi riwayat nyata secara adil.")
    
    table_rows = []
    for m, d in details.items():
        table_rows.append({
            "Metode Peramalan": m,
            "MAD": d["metrics"]["MAD"],
            "MSE": d["metrics"]["MSE"],
            "MAPE (%)": d["metrics"]["MAPE"],
            "Status Kelayakan": "🏆 TERBAIK" if m == best_method else "Normal"
        })
    df_rank = pd.DataFrame(table_rows).sort_values(by="MAPE (%)")
    
    st.dataframe(
        df_rank.style.format({"MAD": "{:.4f}", "MSE": "{:.4f}", "MAPE (%)": "{:.2f}%"})
        .highlight_min(subset=["MAD", "MSE", "MAPE (%)"], color="#DCFCE7")
        .highlight_max(subset=["MAPE (%)"], color="#FEE2E2"),
        use_container_width=True
    )
    
    # Visualisasi Komparasi Fit Semua Metode vs Aktual
    st.markdown("### 📈 Grafik Komparasi Nilai Fit Histori")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=period_labels, y=values, name="🔴 AKTUAL NYATA", line=dict(color="#EF4444", width=3.5)))
    
    # Selalu munculkan yang terbaik dengan garis tebal, sisanya tipis
    for m, d in details.items():
        is_best = (m == best_method)
        fig_comp.add_trace(go.Scatter(
            x=period_labels,
            y=d["fitted"],
            name=f"✨ {m}" if is_best else m,
            line=dict(width=3 if is_best else 1.2, dash="dash" if not is_best else "solid"),
            opacity=1.0 if is_best else 0.5
        ))
    fig_comp.update_layout(
        title="Perbandingan Kecocokan Model Kurva Historis (Fitted Values)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig_comp.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig_comp.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig_comp, use_container_width=True)


# ─── TAB 2: PROYEKSI MASA DEPAN (HORIZON FORECAST) ───────────────────────────
with tab2:
    st.markdown("### 🔮 Proyeksi Nilai Masa Depan (Hasil Jangka Panjang)")
    st.write(f"Berikut merupakan visualisasi dan tabel hasil proyeksi ke depan menggunakan metode pilihan Anda atau metode terbaik otomatis (**{best_method}**).")
    
    # Pilihan metode khusus untuk visualisasi masa depan
    selected_view_method = st.selectbox("Ganti Metode Tampilan Proyeksi Gambar:", methods_list, index=methods_list.index(best_method))
    future_forecast = details[selected_view_method]["future"]
    
    # Gabungkan data untuk visualisasi berkesinambungan
    extended_labels = period_labels + future_labels
    
    fig_f = go.Figure()
    # Garis Aktual Historis
    fig_f.add_trace(go.Scatter(x=period_labels, y=values, name="Aktual Data Histori", line=dict(color="#1E293B", width=3)))
    # Garis Fit Historis Metode Terpilih
    fig_f.add_trace(go.Scatter(x=period_labels, y=details[selected_view_method]["fitted"], name=f"Fit Histori ({selected_view_method})", line=dict(color="#94A3B8", width=1.5, dash="dot")))
    # Garis Forecast Jangka Panjang Depan
    # Agar menyambung dengan data akhir riwayat:
    conn_labels = [period_labels[-1]] + future_labels
    conn_values = [values[-1]] + list(future_forecast)
    fig_f.add_trace(go.Scatter(x=conn_labels, y=conn_values, name=f"🔮 Proyeksi Depan ({selected_view_method})", line=dict(color="#2563EB", width=3.5)))
    
    fig_f.update_layout(
        title=f"Tren & Proyeksi Jangka Panjang Menggunakan Metode: {selected_view_method}",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    fig_f.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig_f.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig_f, use_container_width=True)
    
    # Tampilkan Tabel Hasil Forecast Masa Depan & Fitur Download Hasil Forecast
    st.markdown("#### 📋 Tabel Hasil Nilai Ramalan Masa Depan")
    df_future_table = pd.DataFrame({
        "Periode Masa Depan (Horizon)": future_labels,
        "Nilai Hasil Peramalan Forecast": future_forecast
    })
    
    col_tbl, col_dl = st.columns([2, 1])
    with col_tbl:
        st.dataframe(df_future_table.style.format({"Nilai Hasil Peramalan Forecast": "{:.4f}"}), use_container_width=True)
        
    with col_dl:
        st.markdown("<div style='background-color:#EFF6FF; padding:1.2rem; border-radius:10px; border:1px solid #BFDBFE;'>", unsafe_allow_html=True)
        st.markdown(f"##### 📥 Download Hasil Forecast ({selected_view_method})")
        st.write("Unduh hasil ramalan proyeksi masa depan ini langsung untuk kebutuhan laporan eksternal.")
        
        # 1. Excel Download
        buffer_fc_xlsx = io.BytesIO()
        with pd.ExcelWriter(buffer_fc_xlsx, engine='xlsxwriter') as wr_fc:
            df_future_table.to_excel(wr_fc, sheet_name="Hasil_Forecast", index=False)
        st.download_button(
            label="📥 Download Forecast (.xlsx)",
            data=buffer_fc_xlsx.getvalue(),
            file_name=f"Hasil_Forecast_{selected_view_method.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # 2. CSV Download
        csv_fc_data = df_future_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Forecast (.csv)",
            data=csv_fc_data,
            file_name=f"Hasil_Forecast_{selected_view_method.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ─── TAB 3: DETAIL PERHITUNGAN GALAT PER METODE & NILAI ALPHA/BETA ───────────
with tab3:
    st.markdown("### 📑 Telusuri Logika Hitung Galat & Nilai Alpha/Beta/Gamma Model")
    st.write("Gunakan menu dropdown di bawah untuk menginspeksi tabel rincian deviasi sisa error ($e_t$) dari masing-masing 10 metode peramalan.")
    
    inspect_method = st.selectbox("Pilih Metode yang Ingin Dibongkar Perhitungannya:", methods_list, index=methods_list.index(best_method))
    
    # Tampilkan Informasi Definisi Metode
    st.info(f"ℹ️ **Informasi Metode ({inspect_method}):** {METHOD_INFO[inspect_method]}")
    
    target_data = details[inspect_method]
    
    # Tampilkan Nilai Parameter Eksponensial / Model jika ada
    if target_data["params"]:
        st.markdown("#### ⚙️ Nilai Parameter Koefisien Hasil Fitting")
        p_cols = st.columns(3)
        with p_cols[0]: st.metric("Alpha (α) Perataan Level", format_param(target_data["params"].get("Alpha")))
        with p_cols[1]: st.metric("Beta (β) Perataan Tren", format_param(target_data["params"].get("Beta")))
        with p_cols[2]: st.metric("Gamma (γ) Perataan Musiman", format_param(target_data["params"].get("Gamma")))
        st.write("")
        
    st.markdown(f"#### 📐 Tabel Rincian Deviasi Error per Periode ({inspect_method})")
    st.dataframe(
        target_data["error_table"].style.format({
            "Aktual (Yt)": "{:.2f}",
            "Prediksi (Ft)": "{:.4f}",
            "Error (et)": "{:.4f}",
            "|et| (Absolute)": "{:.4f}",
            "et² (Squared)": "{:.4f}",
            "|et/Yt| % (MAPE)": "{:.2f}%"
        }),
        use_container_width=True
    )


# ─── TAB 4: DATASET RIWAYAT (INPUT) ──────────────────────────────────────────
with tab4:
    st.markdown("### 📊 Ringkasan Deskriptif Data Aktual Riwayat")
    st.write("Struktur tabel data mentah masukan yang dibaca aktif oleh mesin peramalan cerdas saat ini.")
    
    c_d1, c_d2 = st.columns([1, 2])
    with c_d1:
        st.markdown("#### 📑 Ringkasan Statistik Dasar")
        st.dataframe(df[value_col].describe().to_frame(), use_container_width=True)
    with c_d2:
        st.markdown("#### 📥 Tabel Data Masukan")
        st.dataframe(df, use_container_width=True)


# --- DOWNLOAD AREA FULL REPORT (BOTTOM ACCORDION) ---
st.write("")
st.write("")
render_section_header("📥 Download Full Comprehensive Report (Semua Metode)")
with st.expander("🚀 Ekspor Hasil Komparasi Lengkap (Excel / PDF Summary)"):
    st.write("Gunakan fitur di bawah untuk mencetak buku laporan komparasi menyeluruh yang berisi rangkasan statistik dan seluruh lembar kerja perhitungan metode sekaligus.")
    
    dl4_c1, dl4_c2 = st.columns(2)
    with dl4_c1:
        excel_full = generate_excel_report(details, best_method, future_labels, details[best_method]["future"], values, period_labels)
        if excel_full:
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
                best_method, best_metrics, future_labels, details[best_method]["future"],
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

# ─── CREDIT FOOTER (Results Page) ───────────────────────────────────────────
render_credit_footer()
