import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="AI Equity Valuation Pro", layout="wide")

# Estilo para cajas negras y métricas de alta visibilidad
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] { color: #00ff00 !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0e1117; }
    .stTabs [data-baseweb="tab"] { color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LÓGICA DE DATOS Y PAÍSES ---
# Mapeo de crecimiento PIB estimado a largo plazo
PIB_DATA = {
    "United States": 2.2, "Spain": 1.5, "Germany": 1.1, "France": 1.3,
    "United Kingdom": 1.4, "Switzerland": 1.6, "Netherlands": 1.5,
    "China": 4.0, "Japan": 0.5, "Canada": 1.8, "Default": 2.0
}

@st.cache_data
def get_stock_full_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    # Intentar capturar Free Cash Flow
    try:
        fcf = stock.cashflow.loc['Free Cash Flow'].iloc[0]
    except:
        fcf = info.get('freeCashflow', 0)
    return info, fcf

# --- 3. SIDEBAR (ENTRADAS) ---
st.sidebar.header("🔍 Configuración")
ticker = st.sidebar.text_input("Ticker (ej: AAPL, MSFT, ITX.MC, MC.PA)", "AAPL").upper()

try:
    info, fcf_actual = get_stock_full_data(ticker)
    pais = info.get('country', 'Default')
    g_sugerida = PIB_DATA.get(pais, PIB_DATA["Default"])

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Parámetros de Valoración")
    
    # El slider de crecimiento se ajusta automáticamente al PIB del país detectado
    g = st.sidebar.slider(f"Crecimiento Perpetuo (g) % - Ref: PIB {pais}", 0.0, 5.0, g_sugerida) / 100
    k = st.sidebar.slider("Tasa de Descuento (k) % - Retorno Exigido", 5.0, 18.0, 9.0) / 100

    # --- 4. PANEL PRINCIPAL ---
    st.title(f"📊 Valoración Intrínseca: {info.get('longName', ticker)}")
    st.caption(f"Sede: {pais} | Sector: {info.get('sector', 'N/A')} | Industria: {info.get('industry', 'N/A')}")

    # MÉTRICAS CLAVE
    m1, m2, m3, m4 = st.columns(4)
    price = info.get('currentPrice', 1)
    currency = info.get('currency', 'USD')
    
    m1.metric("Precio Actual", f"{price} {currency}")
    m2.metric("PER (P/E Ratio)", info.get('trailingPE', 'N/A'))
    m3.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    m4.metric("FCF Anual", f"{fcf_actual/1e9:.2f}B {currency}")

    st.divider()

    # --- 5. CÁLCULOS DE LOS MODELOS ---
    tab_ddm, tab_dcf = st.tabs(["💡 Modelo Gordon (Dividendos)", "💰 Modelo DCF (Flujos de Caja)"])

    with tab_ddm:
        st.subheader("Valuación por Dividendos")
        st.latex(r"V = \frac{D_0 \times (1 + g)}{k - g}")
        
        d0 = info.get('trailingAnnualDividendRate', 0)
        if d0 > 0 and k > g:
            val_ddm = (d0 * (1 + g)) / (k - g)
            st.metric("Valor Objetivo DDM", f"{val_ddm:.2f} {currency}")
        else:
            val_ddm = 0
            st.warning("Modelo DDM no aplicable (No hay dividendos o k < g)")

    with tab_dcf:
        st.subheader("Valuación por Descuento de Flujos")
        st.latex(r"PV = \sum \frac{FCF_t}{(1+k)^t} + \frac{TV}{(1+k)^n}")
        
        # Proyección simplificada 5 años + Valor Terminal
        fcf_proyectado = [fcf_actual * (1 + g)**i for i in range(1, 6)]
        fcf_descontado = sum([f / (1 + k)**i for i, f in enumerate(fcf_proyectado, 1)])
        
        tv = (fcf_proyectado[-1] * (1 + g)) / (k - g)
        tv_descontado = tv / (1 + k)**5
        
        # Ajuste de Valor de Empresa a Valor de Capital (Equity Value)
        debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        shares = info.get('sharesOutstanding', 1)
        
        val_dcf = (fcf_descontado + tv_descontado - debt + cash) / shares
        
        st.metric("Valor Objetivo DCF", f"{val_dcf:.2f} {currency}")
        st.caption(f"Ajustado por Deuda Neta de {(debt-cash)/1e9:.2f}B {currency}")

    # --- 6. RESULTADO FINAL ---
    st.divider()
    st.subheader("🎯 Veredicto y Precio Objetivo")
    
    # Promediar modelos si ambos son válidos
    objetivo = (val_ddm + val_dcf) / 2 if (val_ddm > 0 and val_dcf > 0) else max(val_ddm, val_dcf)
    
    if objetivo > 0:
        potencial = ((objetivo / price) - 1) * 100
        c1, c2 = st.columns(2)
        
        c1.write(f"### Objetivo Promedio: **{objetivo:.2f} {currency}**")
        
        # Lógica de color según el margen de seguridad
        if price < objetivo * 0.8:
            st.success(f"🟢 **INFRAVALORADA** | Potencial: {potencial:.2f}%")
        elif price < objetivo:
            st.warning(f"🟡 **PRECIO JUSTO** | Potencial: {potencial:.2f}%")
        else:
            st.error(f"🔴 **SOBREVALORADA** | Potencial: {potencial:.2f}%")
            
        st.progress(min(max(potencial + 50, 0), 100) / 100) # Barra visual de potencial
    else:
        st.info("Introduce un Ticker válido o ajusta los parámetros para ver el resultado.")

except Exception as e:
    st.error(f"Error al cargar datos: {e}")
