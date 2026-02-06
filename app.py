import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="AI Equity Analysis Pro", layout="wide")

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
    .diag-box {
        background-color: #1c2128;
        padding: 15px;
        border-left: 5px solid #30363d;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATOS PIB ---
PIB_DATA = {
    "United States": 2.2, "Spain": 1.5, "Germany": 1.1, "France": 1.3,
    "United Kingdom": 1.4, "Switzerland": 1.6, "Netherlands": 1.5,
    "China": 4.0, "Japan": 0.5, "Default": 2.0
}

@st.cache_data
def get_full_analysis(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        fcf = stock.cashflow.loc['Free Cash Flow'].iloc[0]
    except:
        fcf = info.get('freeCashflow', 0)
    return info, fcf

# --- 3. SIDEBAR ---
st.sidebar.header("🔍 Configuración")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()

try:
    info, fcf_actual = get_full_analysis(ticker)
    pais = info.get('country', 'Default')
    g_pib = PIB_DATA.get(pais, PIB_DATA["Default"])

    g = st.sidebar.slider(f"Crecimiento (g) % [PIB {pais}]", 0.0, 5.0, g_pib) / 100
    k = st.sidebar.slider("Descuento (k) % [Tu exigencia]", 5.0, 18.0, 9.0) / 100

    # --- 4. CABECERA ---
    st.title(f"📊 {info.get('longName', ticker)}")
    
    col_p, col_pe, col_div, col_fcf = st.columns(4)
    price = info.get('currentPrice', 1)
    currency = info.get('currency', 'USD')
    col_p.metric("Precio Actual", f"{price} {currency}")
    col_pe.metric("PER", f"{info.get('trailingPE', 'N/A')}")
    col_div.metric("Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    col_fcf.metric("FCF Anual", f"{fcf_actual/1e9:.2f}B")

    st.divider()

    # --- 5. SECCIÓN DE CALIDAD EXPLICADA ---
    st.subheader("🛡️ Análisis de Calidad (Business Health)")
    q1, q2, q3 = st.columns(3)

    # Lógica de ROE (Foso Competitivo)
    roe = info.get('returnOnEquity', 0)
    with q1:
        st.metric("ROE (Rentabilidad)", f"{roe*100:.1f}%")
        if roe > k:
            st.markdown(f"<div class='diag-box'>✅ <b>Foso Competitivo:</b> La empresa gana más de lo que le cuesta el capital. Está creando valor real.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='diag-box'>⚠️ <b>Destrucción de Valor:</b> El ROE es menor que tu exigencia ({k*100}%). El negocio no es eficiente.</div>", unsafe_allow_html=True)

    # Lógica de FCF Conversion (Calidad del Beneficio)
    net_income = info.get('netIncomeToCommon', 1)
    conversion = fcf_actual / net_income if net_income != 0 else 0
    with q2:
        st.metric("Conversión de Caja", f"{conversion:.2f}x")
        if conversion > 1:
            st.markdown(f"<div class='diag-box'>💎 <b>Caja de Calidad:</b> El flujo de caja supera al beneficio neto. Beneficios reales y tangibles.</div>", unsafe_allow_html=True)
        elif conversion > 0.7:
            st.markdown(f"<div class='diag-box'>🆗 <b>Normal:</b> La conversión es aceptable para una empresa industrial o madura.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='diag-box'>🚨 <b>Alarma Contable:</b> Mucho beneficio pero poca caja. Ojo con los inventarios o maquillaje contable.</div>", unsafe_allow_html=True)

    # Lógica de Solvencia (Current Ratio)
    cr = info.get('currentRatio', 0)
    with q3:
        st.metric("Liquidez (CR)", f"{cr:.2f}x")
        if cr > 1.5:
            st.markdown(f"<div class='diag-box'>🛡️ <b>Solvencia Alta:</b> Tiene activos líquidos de sobra para pagar sus deudas a corto plazo.</div>", unsafe_allow_html=True)
        elif cr > 1:
            st.markdown(f"<div class='diag-box'>⚖️ <b>Ajustada:</b> Liquidez justa. No hay peligro inmediato, pero no sobra nada.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='diag-box'>🔥 <b>Riesgo de Liquidez:</b> Debe más a corto plazo de lo que tiene en caja. Peligro de impago.</div>", unsafe_allow_html=True)

    st.divider()

    # --- 6. VALORACIÓN ---
    tab_dcf, tab_sens = st.tabs(["💰 Valoración DCF", "📉 Matriz de Sensibilidad"])

    with tab_dcf:
        fcf_list = [fcf_actual * (1 + g)**i for i in range(1, 6)]
        fcf_d = sum([f / (1 + k)**i for i, f in enumerate(fcf_list, 1)])
        tv = (fcf_list[-1] * (1 + g)) / (k - g)
        tv_d = tv / (1 + k)**5
        debt, cash = info.get('totalDebt', 0), info.get('totalCash', 0)
        shares = info.get('sharesOutstanding', 1)
        val_dcf = (fcf_d + tv_d - debt + cash) / shares
        
        c_v1, c_v2 = st.columns(2)
        c_v1.metric("Precio Objetivo DCF", f"{val_dcf:.2f} {currency}")
        
        # Explicación del DCF Negativo si ocurre
        if val_dcf < 0:
            st.error("⚠️ El valor es negativo porque la deuda es mayor que la capacidad de generar caja.")
        
        potencial = ((val_dcf / price) - 1) * 100
        if potencial > 20:
            c_v2.success(f"COMPRA: Potencial {potencial:.1f}%")
        elif potencial < 0:
            c_v2.error(f"CARA: Potencial {potencial:.1f}%")
        else:
            c_v2.warning(f"PRECIO JUSTO: Potencial {potencial:.1f}%")

    with tab_sens:
        sens_k = [k-0.01, k, k+0.01]
        sens_g = [g-0.005, g, g+0.005]
        d_proxy = (fcf_actual/shares)
        matrix = [[(d_proxy * (1+gi))/(ki-gi) if ki > gi else 0 for gi in sens_g] for ki in sens_k]
        df_sens = pd.DataFrame(matrix, columns=[f"g:{gi*100:.1f}%" for gi in sens_g], index=[f"k:{ki*100:.1f}%" for ki in sens_k])
        st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn'))
        st.caption("Esta matriz muestra cómo cambiaría el precio si tus estimaciones varían ligeramente.")

except Exception as e:
    st.error(f"Ticker no encontrado o datos insuficientes.")
