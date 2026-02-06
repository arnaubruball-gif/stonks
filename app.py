import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="AI Equity Valuation Pro", layout="wide")

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
    
    g = st.sidebar.slider(f"Crecimiento Perpetuo (g) % [PIB {pais}]", 0.0, 5.0, g_pib) / 100
    k = st.sidebar.slider("Tasa de Descuento (k) % [Retorno Exigido]", 5.0, 18.0, 9.0) / 100

    # --- 4. PANEL PRINCIPAL ---
    st.title(f"📊 {info.get('longName', ticker)}")
    st.caption(f"📍 {pais} | {info.get('sector', 'N/A')} | {info.get('currency', 'USD')}")

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
        tab_ddm, tab_dcf, tab_diag = st.tabs(["💡 Modelo Gordon", "💰 Modelo DCF", "🛡️ Diagnóstico de Calidad"])
        
        with tab_ddm:
            d0 = info.get('trailingAnnualDividendRate', 0)
            if d0 > 0 and k > g:
                val_ddm = (d0 * (1 + g)) / (k - g)
                st.metric("Valor DDM", f"{val_ddm:.2f} {currency}")
                st.latex(r"V = \frac{D_0(1+g)}{k-g}")
            else:
                val_ddm = 0
                st.warning("No disponible para este activo.")

        with tab_dcf:
            fcf_proyectado = [fcf_actual * (1 + g)**i for i in range(1, 6)]
            fcf_descontado = sum([f / (1 + k)**i for i, f in enumerate(fcf_proyectado, 1)])
            tv = (fcf_proyectado[-1] * (1 + g)) / (k - g)
            tv_descontado = tv / (1 + k)**5
            debt, cash = info.get('totalDebt', 0), info.get('totalCash', 0)
            shares = info.get('sharesOutstanding', 1)
            val_dcf = (fcf_descontado + tv_descontado - debt + cash) / shares
            
            st.metric("Valor DCF", f"{val_dcf:.2f} {currency}")
            if val_dcf < 0: st.error("Valor negativo: La deuda supera el valor operativo.")
            st.caption(f"Ajuste Deuda Neta: {(debt-cash)/1e9:.2f}B")

        with tab_diag:
            st.subheader("Interpretación de Métricas")
            roe = info.get('returnOnEquity', 0)
            net_income = info.get('netIncomeToCommon', 1)
            conversion = fcf_actual / net_income if net_income != 0 else 0
            cr = info.get('currentRatio', 0)

            # Bloque ROE
            if roe > k:
                st.markdown(f"<div class='diag-box'>✅ <b>ROE ({roe*100:.1f}%):</b> Supera tu exigencia del {k*100:.1f}%. La empresa es una <b>máquina de crear riqueza</b>.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ff4b4b;'>⚠️ <b>ROE ({roe*100:.1f}%):</b> Es menor que el coste de capital. El negocio podría estar <b>destruyendo valor</b>.</div>", unsafe_allow_html=True)
            
            # Bloque Caja
            if conversion > 1:
                st.markdown(f"<div class='diag-box'>💎 <b>Caja Real:</b> Convierte más del 100% de sus beneficios en efectivo. Alta calidad contable.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ffa500;'>🧐 <b>Caja vs Beneficio:</b> Solo convierte el {conversion*100:.1f}% del beneficio. Revisa si hay mucho gasto en mantenimiento.</div>", unsafe_allow_html=True)

            # Bloque Liquidez
            if cr > 1.5:
                st.markdown(f"<div class='diag-box'>🛡️ <b>Liquidez:</b> Muy sólida. Tiene {cr} veces más activos que deudas a corto plazo.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ff4b4b;'>🔥 <b>Liquidez Crítica:</b> Solo {cr}x. Podría tener problemas si el mercado se frena.</div>", unsafe_allow_html=True)

    with col_right:
        st.subheader("🛡️ Resumen Calidad")
        st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.1f}%")
        st.write(f"**FCF Conv.:** {conversion:.2f}x")
        st.write(f"**Liquidez:** {info.get('currentRatio', 0):.2f}x")

    st.subheader("📉 Matriz de Sensibilidad")
    sens_k = [k-0.01, k, k+0.01]
    sens_g = [g-0.005, g, g+0.005]
    d0_val = d0 if d0 > 0 else (fcf_actual/shares)
    matrix = [[(d0_val * (1+gi))/(ki-gi) if ki > gi else 0 for gi in sens_g] for ki in sens_k]
    df_sens = pd.DataFrame(matrix, columns=[f"g: {gi*100:.1f}%" for gi in sens_g], index=[f"k: {ki*100:.1f}%" for ki in sens_k])
    st.dataframe(df_sens.style.highlight_max(axis=None, color='#004400').format("{:.2f}"))

    st.divider()
    objetivo = (val_ddm + val_dcf) / 2 if (val_ddm > 0 and val_dcf > 0) else max(val_ddm, val_dcf)
    
    if objetivo > 0:
        potencial = ((objetivo / price) - 1) * 100
        st.subheader(f"🎯 Veredicto Final: {objetivo:.2f} {currency}")
        if potencial > 20: st.success(f"🟢 **COMPRA CLARA** | Potencial: {potencial:.1f}%")
        elif potencial > 0: st.warning(f"🟡 **PRECIO JUSTO** | Potencial: {potencial:.1f}%")
        else: st.error(f"🔴 **SOBREVALORADA** | Potencial: {potencial:.1f}%")

except Exception as e:
    st.error(f"Introduce un Ticker válido o ajusta los parámetros. Error: {e}")
