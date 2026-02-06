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
    [data-testid="stMetricValue"] { color: #00ff00 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0e1117; }
    .stTabs [data-baseweb="tab"] { color: white; }
    table { color: white !important; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LÓGICA DE DATOS Y PAÍSES ---
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

# --- 3. SIDEBAR ---
st.sidebar.header("🔍 Configuración")
ticker = st.sidebar.text_input("Ticker (ej: AAPL, MSFT, ITX.MC, MC.PA)", "AAPL").upper()

try:
    info, fcf_actual = get_stock_full_data(ticker)
    pais = info.get('country', 'Default')
    g_pib = PIB_DATA.get(pais, PIB_DATA["Default"])

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Parámetros de Valoración")
    
    # Sliders maestros
    g = st.sidebar.slider(f"Crecimiento Perpetuo (g) % [PIB {pais}]", 0.0, 5.0, g_pib) / 100
    k = st.sidebar.slider("Tasa de Descuento (k) % [Retorno Exigido]", 5.0, 18.0, 9.0) / 100

    # --- 4. PANEL PRINCIPAL ---
    st.title(f"📊 {info.get('longName', ticker)}")
    st.caption(f"📍 {pais} | {info.get('sector', 'N/A')} | {info.get('currency', 'USD')}")

    # MÉTRICAS DE MERCADO
    m1, m2, m3, m4 = st.columns(4)
    price = info.get('currentPrice', 1)
    currency = info.get('currency', 'USD')
    
    m1.metric("Precio Actual", f"{price} {currency}")
    m2.metric("PER (P/E)", f"{info.get('trailingPE', 'N/A')}")
    m3.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    m4.metric("FCF Anual", f"{fcf_actual/1e9:.2f}B")

    st.divider()

    # --- 5. VALORACIÓN Y CALIDAD ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        tab_ddm, tab_dcf = st.tabs(["💡 Modelo Gordon", "💰 Modelo DCF"])
        
        # MODELO GORDON
        with tab_ddm:
            d0 = info.get('trailingAnnualDividendRate', 0)
            if d0 > 0 and k > g:
                val_ddm = (d0 * (1 + g)) / (k - g)
                st.metric("Valor DDM", f"{val_ddm:.2f} {currency}")
                st.latex(r"V = \frac{D_0(1+g)}{k-g}")
            else:
                val_ddm = 0
                st.warning("No disponible para este activo.")

        # MODELO DCF
        with tab_dcf:
            fcf_proyectado = [fcf_actual * (1 + g)**i for i in range(1, 6)]
            fcf_descontado = sum([f / (1 + k)**i for i, f in enumerate(fcf_proyectado, 1)])
            tv = (fcf_proyectado[-1] * (1 + g)) / (k - g)
            tv_descontado = tv / (1 + k)**5
            
            debt, cash = info.get('totalDebt', 0), info.get('totalCash', 0)
            shares = info.get('sharesOutstanding', 1)
            val_dcf = (fcf_descontado + tv_descontado - debt + cash) / shares
            
            st.metric("Valor DCF", f"{val_dcf:.2f} {currency}")
            st.caption(f"Ajuste Deuda Neta: {(debt-cash)/1e9:.2f}B")

    # MÉTRICAS DE CALIDAD (AFILADO)
    with col_right:
        st.subheader("🛡️ Calidad")
        roe = info.get('returnOnEquity', 0)
        conversion = fcf_actual / info.get('netIncomeToCommon', 1)
        cr = info.get('currentRatio', 0)
        
        st.write(f"**ROE:** {roe*100:.1f}% {'✅' if roe > k else '⚠️'}")
        st.write(f"**FCF/Net Income:** {conversion:.2f}x")
        st.write(f"**Liquidez (CR):** {cr:.2f}x")

    # --- 6. MATRIZ DE SENSIBILIDAD ---
    st.subheader("📉 Matriz de Sensibilidad (Margen de Error)")
    st.write("Cómo varía el precio objetivo si cambiamos las hipótesis de k y g:")
    
    sens_k = [k-0.01, k, k+0.01]
    sens_g = [g-0.005, g, g+0.005]
    
    # Usamos el modelo DDM para la matriz por simplicidad visual
    d0_val = d0 if d0 > 0 else (fcf_actual/shares) # Proxy si no hay div
    matrix = [[(d0_val * (1+gi))/(ki-gi) if ki > gi else 0 for gi in sens_g] for ki in sens_k]
    
    df_sens = pd.DataFrame(matrix, 
                           columns=[f"g: {gi*100:.1f}%" for gi in sens_g],
                           index=[f"k: {ki*100:.1f}%" for ki in sens_k])
    st.dataframe(df_sens.style.highlight_max(axis=None, color='#004400').format("{:.2f}"))

    # --- 7. VEREDICTO ---
    st.divider()
    objetivo = (val_ddm + val_dcf) / 2 if (val_ddm > 0 and val_dcf > 0) else max(val_ddm, val_dcf)
    
    if objetivo > 0:
        potencial = ((objetivo / price) - 1) * 100
        st.subheader(f"🎯 Veredicto Final: {objetivo:.2f} {currency}")
        
        if potencial > 20:
            st.success(f"🟢 **COMPRA CLARA** | Potencial: {potencial:.1f}%")
        elif potencial > 0:
            st.warning(f"🟡 **PRECIO JUSTO** | Potencial: {potencial:.1f}%")
        else:
            st.error(f"🔴 **SOBREVALORADA** | Potencial: {potencial:.1f}%")

except Exception as e:
    st.error(f"Error: Introduce un Ticker válido. Detalles: {e}")
