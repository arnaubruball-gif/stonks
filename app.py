import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Equity Research Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 8px solid; }
    .good { background-color: #052111; border-color: #3fb950; }
    .bad { background-color: #210505; border-color: #f85149; }
    .neutral { background-color: #1c1c1c; border-color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LÓGICA DE DATOS ---
@st.cache_data
def get_extended_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="5y")
    financials = stock.financials
    balance = stock.balance_sheet
    cashflow = stock.cashflow
    return stock, info, hist, financials, balance, cashflow

# --- 3. SIDEBAR ---
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/3310/3310111.png"
st.sidebar.image(LOGO_URL, width=60)
st.sidebar.title("Pro Analyzer v3.0")
ticker = st.sidebar.text_input("Ticker", "NVDA").upper()

try:
    stock, info, hist, fin, bs, cf = get_extended_data(ticker)
    
    # Parámetros Sidebar
    st.sidebar.divider()
    g_f1 = st.sidebar.slider("Crecimiento Etapa 1 (%)", 0.0, 60.0, 20.0) / 100
    k = st.sidebar.slider("Tasa Descuento (k) %", 5.0, 15.0, 9.0) / 100

    # --- 4. HEADER ---
    st.title(f"{info.get('longName', ticker)}")
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    price = info.get('currentPrice', 1)
    col_h1.metric("Precio", f"{price} {info.get('currency')}")
    col_h2.metric("Market Cap", f"{info.get('marketCap', 0)/1e9:.2f}B")
    col_h3.metric("PER Actual", f"{info.get('trailingPE', 0):.2f}")
    col_h4.metric("Forward PER", f"{info.get('forwardPE', 0):.2f}")

    # --- 5. MODELOS DE ANÁLISIS ---
    t_val, t_quality, t_multiples = st.tabs(["💎 Valoración Intrínseca", "🛡️ Filtros de Calidad", "📊 Múltiplos e Histórico"])

    with t_val:
        # DCF 2 ETAPAS
        fcf_actual = info.get('freeCashflow', 0)
        g_p = 0.02 # Crecimiento perpetuo conservador
        fcf_list = [fcf_actual * (1 + g_f1)**i for i in range(1, 6)]
        fcf_d = sum([f / (1 + k)**i for i, f in enumerate(fcf_list, 1)])
        tv = (fcf_list[-1] * (1 + g_p)) / (k - g_p)
        tv_d = tv / (1 + k)**5
        val_dcf = (fcf_d + tv_d - info.get('totalDebt', 0) + info.get('totalCash', 0)) / info.get('sharesOutstanding', 1)
        
        # MODELO BENJAMIN GRAHAM (Modificado)
        eps = info.get('trailingEps', 0)
        val_graham = (eps * (8.5 + 2 * (g_f1 * 100)) * 4.4) / 4.5 # 4.5 es el yield de bonos AAA
        
        v1, v2 = st.columns(2)
        v1.metric("Valor DCF (Caja)", f"{val_dcf:.2f}")
        v2.metric("Valor Graham (EPS)", f"{val_graham:.2f}")
        
        st.write("### 📉 Matriz de Sensibilidad DCF")
        # Reutilizamos la lógica de matriz para dar robustez
        sk = [k-0.01, k, k+0.01]
        sg = [g_f1-0.05, g_f1, g_f1+0.05]
        m_data = [[((sum([(fcf_actual*(1+gi)**i)/(1+ki)**i for i in range(1,6)]) + (((fcf_actual*(1+gi)**5)*(1+g_p))/(ki-g_p))/(1+ki)**5) - info.get('totalDebt',0) + info.get('totalCash',0))/info.get('sharesOutstanding',1)) for gi in sg] for ki in sk]
        st.table(pd.DataFrame(m_data, columns=[f"G:{g*100:.0f}%" for g in sg], index=[f"k:{k*100:.1f}%" for k in sk]))

    with t_quality:
        st.subheader("Sistema de Puntuación F-Score de Piotroski")
        
        # Simplificación de puntos F-Score para el Dashboard
        points = 0
        checks = {
            "Rentabilidad Neta Positiva": info.get('returnOnAssets', 0) > 0,
            "Flujo de Caja Operativo Positivo": fcf_actual > 0,
            "ROE > Media Sector (Proxy 15%)": info.get('returnOnEquity', 0) > 0.15,
            "Margen Bruto Creciente (Proxy)": info.get('grossMargins', 0) > 0.40,
            "Liquidez > 1.5x": info.get('currentRatio', 0) > 1.5
        }
        
        for text, result in checks.items():
            col_a, col_b = st.columns([3, 1])
            col_a.write(text)
            if result:
                col_b.success("PASADO (+1)")
                points += 1
            else:
                col_b.error("FALLO (0)")
        
        st.metric("Puntuación Final de Calidad", f"{points} / 5")

    with t_multiples:
        st.subheader("Análisis de Márgenes y Múltiplos")
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.write("**Márgenes Operativos**")
            st.progress(info.get('grossMargins', 0), text=f"M. Bruto: {info.get('grossMargins', 0)*100:.1f}%")
            st.progress(info.get('ebitdaMargins', 0), text=f"M. EBITDA: {info.get('ebitdaMargins', 0)*100:.1f}%")
            st.progress(info.get('profitMargins', 0), text=f"M. Neto: {info.get('profitMargins', 0)*100:.1f}%")
        
        with m_col2:
            st.write("**Comparativa PER**")
            per_h = info.get('trailingPE', 0)
            avg_5y = 25 # Proxy de media sectorial
            st.metric("PER vs Media (Est.)", f"{per_h:.2f}", delta=f"{per_h - avg_5y:.2f}", delta_color="inverse")

    # --- 6. VEREDICTO FINAL ---
    st.divider()
    objetivo_final = (val_dcf * 0.7) + (val_graham * 0.3)
    margin = ((objetivo_final / price) - 1) * 100
    
    if margin > 20 and points >= 4:
        st.markdown(f"<div class='status-box good'><h3>🚀 COMPRA FUERTE</h3>Potencial: {margin:.1f}% | Calidad: {points}/5<br>La empresa tiene infravaloración y fundamentos sólidos.</div>", unsafe_allow_html=True)
    elif margin > 0:
        st.markdown(f"<div class='status-box neutral'><h3>⚖️ MANTENER / PRECIO JUSTO</h3>Potencial: {margin:.1f}%<br>El precio actual refleja bien el valor del negocio.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='status-box bad'><h3>⚠️ SOBREVALORADA</h3>Potencial: {margin:.1f}%<br>Riesgo de caída alto. No hay margen de seguridad.</div>", unsafe_allow_html=True)

except Exception as e:
    st.error("Error en el análisis. Verifica el Ticker.")
    st.write(e)
