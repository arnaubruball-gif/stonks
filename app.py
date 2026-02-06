import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Equity Dashboard", layout="wide")

# --- ESTILO CSS PARA CAJAS NEGRAS Y TEXTO CLARO ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    /* Estilo para las métricas */
    [data-testid="stMetricValue"] { color: #00ff00 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 1.1rem; }
    div[data-testid="stMetric"] {
        background-color: #1a1c23;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
    }
    /* Estilo para los tabs y textos */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c23;
        border-radius: 4px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CACHÉ ---
@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        cashflow = stock.cashflow
        fcf = cashflow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cashflow.index else 0
    except:
        fcf = 0
    return info, fcf

# --- SIDEBAR ---
st.sidebar.header("🔍 Configuración de Análisis")

# Lista rápida de ejemplo (puedes ampliarla a 500)
tickers_populares = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "MC.PA", "ASML", "ITX.MC", "SAN.MC", "SAP"]
ticker_choice = st.sidebar.selectbox("Selecciona un Ticker popular:", tickers_populares)
ticker_manual = st.sidebar.text_input("O escribe otro Ticker:", ticker_choice).upper()

st.sidebar.divider()
st.sidebar.subheader("⚙️ Parámetros del Modelo")
k = st.sidebar.slider("Tasa de Descuento (k) %", 5.0, 15.0, 9.5) / 100
g = st.sidebar.slider("Crecimiento Perpetuo (g) %", 0.0, 5.0, 2.5) / 100

# --- CUERPO PRINCIPAL ---
st.title(f"🚀 Análisis de Valoración: {ticker_manual}")

try:
    info, fcf_actual = get_stock_data(ticker_manual)
    
    # MÉTRICAS PRINCIPALES EN CAJAS OSCURAS
    price = info.get('currentPrice', 1)
    currency = info.get('currency', 'USD')
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Actual", f"{price} {currency}")
    m2.metric("PER (Trailing)", f"{info.get('trailingPE', 'N/A')}")
    m3.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    m4.metric("FCF (Último)", f"{fcf_actual/1e9:.2f}B {currency}")

    st.write("---")

    # MODELOS DE VALORACIÓN
    tab_ddm, tab_dcf = st.tabs(["📊 Modelo de Dividendos (DDM)", "💸 Modelo de Flujos (DCF)"])

    with tab_ddm:
        st.subheader("Fórmula de Gordon Growth")
        st.latex(r"V_0 = \frac{D_0 \times (1 + g)}{k - g}")
        
        div_rate = info.get('trailingAnnualDividendRate', 0)
        if div_rate > 0:
            val_ddm = (div_rate * (1 + g)) / (k - g)
            st.info(f"Basado en un dividendo anual de {div_rate} {currency}")
            st.metric("Valor Intrínseco DDM", f"{val_ddm:.2f} {currency}")
        else:
            val_ddm = 0
            st.warning("Esta empresa no tiene un historial de dividendos compatible con DDM.")

    with tab_dcf:
        st.subheader("Discounted Cash Flow (5 años + Terminal)")
        st.latex(r"Value = \sum_{t=1}^{5} \frac{FCF_t}{(1+k)^t} + \frac{TV}{(1+k)^5}")
        
        # Proyecciones
        fcf_list = [fcf_actual * (1 + g)**i for i in range(1, 6)]
        fcf_descontado = sum([f / (1 + k)**(i+1) for i, f in enumerate(fcf_list)])
        
        # Valor Terminal
        tv = (fcf_list[-1] * (1 + g)) / (k - g)
        tv_descontado = tv / (1 + k)**5
        
        # Equity Value
        total_ev = fcf_descontado + tv_descontado
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)
        shares = info.get('sharesOutstanding', 1)
        
        val_dcf = (total_ev - net_debt) / shares
        
        st.metric("Valor Intrínseco DCF", f"{val_dcf:.2f} {currency}")
        st.write(f"*Nota: Se ha ajustado por una deuda neta de {net_debt/1e9:.2f}B.*")

    # VEREDICTO FINAL
    st.write("---")
    st.subheader("🎯 Resumen de Valoración y Objetivo")
    
    # Lógica de Objetivo
    if val_ddm > 0 and val_dcf > 0:
        target = (val_ddm + val_dcf) / 2
    else:
        target = max(val_ddm, val_dcf)

    if target > 0:
        potencial = ((target / price) - 1) * 100
        
        col_res, col_gauge = st.columns([1, 1])
        
        with col_res:
            st.write(f"### Valor Objetivo: **{target:.2f} {currency}**")
            if price < target * 0.75:
                st.success(f"💪 **FUERTE COMPRA** (Potencial: {potencial:.1f}%)")
            elif price < target:
                st.success(f"✅ **INFRAVALORADA** (Potencial: {potencial:.1f}%)")
            elif price < target * 1.15:
                st.warning(f"⚖️ **PRECIO JUSTO** (Margen estrecho)")
            else:
                st.error(f"❌ **SOBREVALORADA** (Riesgo de caída: {abs(potencial):.1f}%)")
    else:
        st.error("No hay datos suficientes para calcular un valor objetivo.")

except Exception as e:
    st.error(f"Error cargando datos de {ticker_manual}: {e}")
