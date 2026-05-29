# ─── CREDIT FOOTER ────────────────────────────────────────────────────────────
# Tambahkan kode ini di bagian PALING BAWAH file utama kamu (setelah semua konten)

st.divider()

st.markdown("""
    <style>
    .credit-container {
        background: linear-gradient(135deg, #F0F4FF 0%, #EEF2FF 100%);
        border: 1px solid #E0E4FF;
        border-radius: 16px;
        padding: 24px 32px;
        margin-top: 12px;
        text-align: center;
    }
    .credit-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4F46E5 0%, #A855F7 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .credit-lab {
        background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.05rem;
        font-weight: 800;
        margin: 6px 0 14px 0;
        line-height: 1.4;
    }
    .credit-author-label {
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .credit-authors {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 4px;
    }
    .credit-chip {
        background: #FFFFFF;
        border: 1.5px solid #C7D2FE;
        border-radius: 999px;
        padding: 5px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #4F46E5;
        box-shadow: 0 2px 6px rgba(99,102,241,0.08);
    }
    .credit-year {
        color: #CBD5E1;
        font-size: 0.75rem;
        margin-top: 14px;
        font-weight: 500;
    }
    </style>

    <div class="credit-container">
        <div class="credit-badge">🏭 Developed by</div>
        <div class="credit-lab">
            Assistant Elementary Laboratory<br>Industrial Engineering
        </div>
        <div class="credit-author-label">✦ Authors ✦</div>
        <div class="credit-authors">
            <span class="credit-chip">👤 Primadhani Syah Putera</span>
            <span class="credit-chip">👤 Shafa Khansa Nabila</span>
            <span class="credit-chip">👤 Rafazella Alwan</span>
        </div>
        <div class="credit-year">© 2025 · Forecasting Dashboard · Elementary Laboratory IE</div>
    </div>
""", unsafe_allow_html=True)
