import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="AI Equity Valuation Pro", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] { color: #00ff00 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0e1117; }
    .stTabs [data-baseweb="tab"] { color: white; }
    table { color: white !important; background-color: #161b22; }
    .diag-box {
        background-color: #1c2128;
        padding: 15px;
        border-left: 5px solid #00ff00;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGO Y SIDEBAR ---
# Puedes cambiar esta URL por la de tu logo (hosting en GitHub o Imgur)
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/3310/3310111.png" 

st.sidebar.image(LOGO_URL, width=80)
st.sidebar.title("Valuation Engine")
st.sidebar.divider()

# --- 3. LÓGICA DE DATOS ---
PIB_DATA = {
    "United States": 2.2, "Spain": 1.5, "Germany": 1.1, "France": 1.3,
    "United Kingdom": 1.4, "Switzerland": 1.6, "Netherlands": 1.5,
    "China": 4.0, "Japan": 0.5, "Canada": 1.8, "Default": 2.0
}

@st.cache_data
def get_stock_full_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        fcf = stock.cashflow.loc['Free Cash Flow'].iloc[0]
    except:
        fcf = info.get('freeCashflow', 0)
    return info, fcf

# --- 4. CONFIGURACIÓN SIDEBAR ---
ticker = st.sidebar.text_input("Ticker", "NVDA").upper()

try:
    info, fcf_actual = get_stock_full_data(ticker)
    pais = info.get('country', 'Default')
    g_pib = PIB_DATA.get(pais, PIB_DATA["Default"])

    st.sidebar.subheader("⚙️ Parámetros")
    
    # Sliders para el modelo de 2 etapas
    g_fase1 = st.sidebar.slider("Crecimiento Fase 1 (5 años) %", 0.0, 60.0, 25.0) / 100
    g_perpetuo = st.sidebar.slider(f"g Perpetuo % (PIB {pais})", 0.0, 5.0, g_pib) / 100
    k = st.sidebar.slider("Retorno Exigido (k) %", 5.0, 18.0, 9.0) / 100

    # --- 5. PANEL PRINCIPAL ---
    st.title(f"📊 {info.get('longName', ticker)}")
    st.caption(f"📍 {pais} | {info.get('sector', 'N/A')} | {info.get('currency', 'USD')}")

    m1, m2, m3, m4 = st.columns(4)
    price = info.get('currentPrice', 1)
    currency = info.get('currency', 'USD')
    
    m1.metric("Precio Actual", f"{price} {currency}")
    m2.metric("PER", f"{info.get('trailingPE', 'N/A')}")
    m3.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    m4.metric("FCF Anual", f"{fcf_actual/1e9:.2f}B")

    st.divider()

    # --- 6. VALORACIÓN ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        tab_dcf, tab_diag = st.tabs(["💰 Modelo DCF (2 Etapas)", "🛡️ Diagnóstico de Calidad"])
        
        with tab_dcf:
            # Fase 1
            fcf_list = []
            fcf_descontado = 0
            for i in range(1, 6):
                fcf_i = fcf_actual * (1 + g_fase1)**i
                fcf_list.append(fcf_i)
                fcf_descontado += fcf_i / (1 + k)**i
            
            # Fase 2
            tv = (fcf_list[-1] * (1 + g_perpetuo)) / (k - g_perpetuo)
            tv_descontado = tv / (1 + k)**5
            
            debt, cash = info.get('totalDebt', 0), info.get('totalCash', 0)
            shares = info.get('sharesOutstanding', 1)
            val_dcf = (fcf_descontado + tv_descontado - debt + cash) / shares
            
            st.metric("Valor Objetivo DCF", f"{val_dcf:.2f} {currency}")
            st.latex(r"Value = \sum_{t=1}^{5} \frac{FCF_0(1+g_1)^t}{(1+k)^t} + \frac{TV}{(1+k)^5}")

        with tab_diag:
            roe = info.get('returnOnEquity', 0)
            net_income = info.get('netIncomeToCommon', 1)
            conversion = fcf_actual / net_income if net_income != 0 else 0
            
            if roe > k:
                st.markdown(f"<div class='diag-box'>✅ <b>Ventaja Competitiva:</b> ROE del {roe*100:.1f}% supera el coste de capital.</div>", unsafe_allow_html=True)
            if conversion > 1:
                st.markdown(f"<div class='diag-box'>💎 <b>Caja Real:</b> Excelente calidad de beneficios.</div>", unsafe_allow_html=True)

    with col_right:
        st.subheader("🛡️ Resumen Calidad")
        st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.1f}%")
        st.write(f"**FCF Conv.:** {conversion:.2f}x")
        st.write(f"**Liquidez:** {info.get('currentRatio', 0):.2f}x")

    # --- 7. MATRIZ DE SENSIBILIDAD ---
    st.subheader("📉 Sensibilidad G1 vs k")
    sens_k = [k-0.01, k, k+0.01]
    sens_g = [g_fase1-0.05, g_fase1, g_fase1+0.05]
    
    matrix = []
    for ki in sens_k:
        row = []
        for gi in sens_g:
            f1 = sum([(fcf_actual * (1 + gi)**i) / (1 + ki)**i for i in range(1, 6)])
            tv_i = ((fcf_actual * (1 + gi)**5) * (1 + g_perpetuo)) / (ki - g_perpetuo)
            row.append((f1 + (tv_i / (1 + ki)**5) - debt + cash) / shares)
        matrix.append(row)

    df_sens = pd.DataFrame(matrix, columns=[f"G1:{gi*100:.0f}%" for gi in sens_g], index=[f"k:{ki*100:.1f}%" for ki in sens_k])
    st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    # --- 8. VEREDICTO FINAL ---
    st.divider()
    potencial = ((val_dcf / price) - 1) * 100
    st.subheader(f"🎯 Precio Objetivo: {val_dcf:.2f} {currency}")
    
    if potencial > 20:
        st.success(f"🟢 **COMPRA** | Potencial: {potencial:.1f}%")
    elif potencial > 0:
        st.warning(f"🟡 **VALOR JUSTO** | Potencial: {potencial:.1f}%")
    else:
        st.error(f"🔴 **SOBREVALORADA** | Potencial: {potencial:.1f}%")

except Exception as e:
    st.sidebar.error("Esperando Ticker válido...")
