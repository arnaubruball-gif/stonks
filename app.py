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

# --- 2. LOGO Y LÓGICA DE PAÍSES ---
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/3310/3310111.png" 

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
st.sidebar.image(LOGO_URL, width=80)
st.sidebar.title("Valuation Engine")
ticker = st.sidebar.text_input("Ticker (ej: NVDA, MSFT, ITX.MC)", "NVDA").upper()

try:
    info, fcf_actual = get_stock_full_data(ticker)
    pais = info.get('country', 'Default')
    g_pib = PIB_DATA.get(pais, PIB_DATA["Default"])

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Parámetros de Valoración")
    
    # Sliders para el modelo de 2 etapas (Ajuste para Growth)
    st.sidebar.write("**Fase 1: Crecimiento Próximos 5 años**")
    g_fase1 = st.sidebar.slider("Crecimiento estimado (%)", 0.0, 60.0, 25.0) / 100
    
    st.sidebar.write("**Fase 2: Crecimiento Perpetuo**")
    g_perpetuo = st.sidebar.slider(f"g % (Ref: PIB {pais})", 0.0, 5.0, g_pib) / 100
    
    st.sidebar.write("**Retorno Exigido**")
    k = st.sidebar.slider("Tasa de Descuento (k) %", 5.0, 18.0, 9.0) / 100

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

    # --- 5. VALORACIÓN Y CALIDAD (DASHBOARD DIVIDIDO) ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        tab_dcf, tab_diag, tab_sens = st.tabs(["💰 Modelo DCF (2 Etapas)", "🛡️ Diagnóstico de Calidad", "📉 Matriz de Sensibilidad"])
        
        with tab_dcf:
            st.subheader("Análisis de Flujos Descontados")
            # Cálculo Fase 1
            fcf_list = []
            fcf_descontado = 0
            for i in range(1, 6):
                fcf_i = fcf_actual * (1 + g_fase1)**i
                fcf_list.append(fcf_i)
                fcf_descontado += fcf_i / (1 + k)**i
            
            # Cálculo Fase 2
            tv = (fcf_list[-1] * (1 + g_perpetuo)) / (k - g_perpetuo)
            tv_descontado = tv / (1 + k)**5
            
            # Ajustes de Deuda y Caja
            debt, cash = info.get('totalDebt', 0), info.get('totalCash', 0)
            shares = info.get('sharesOutstanding', 1)
            val_dcf = (fcf_descontado + tv_descontado - debt + cash) / shares
            
            st.metric("Valor Objetivo DCF", f"{val_dcf:.2f} {currency}")
            st.latex(r"Value = \sum_{t=1}^{5} \frac{FCF_0(1+g_1)^t}{(1+k)^t} + \frac{TV}{(1+k)^5}")
            if val_dcf < 0: st.error("⚠️ El valor es negativo: La deuda financiera supera la capacidad de generación de caja.")

        with tab_diag:
            st.subheader("Interpretación de Salud Financiera")
            roe = info.get('returnOnEquity', 0)
            net_income = info.get('netIncomeToCommon', 1)
            conversion = fcf_actual / net_income if net_income != 0 else 0
            cr = info.get('currentRatio', 0)

            # ROE Diagnosis
            if roe > k:
                st.markdown(f"<div class='diag-box'>✅ <b>ROE ({roe*100:.1f}%):</b> Supera tu exigencia. La empresa tiene un <b>Foso Competitivo</b> y crea valor.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ff4b4b;'>⚠️ <b>ROE ({roe*100:.1f}%):</b> Es bajo. La empresa no es eficiente transformando el capital en beneficio.</div>", unsafe_allow_html=True)
            
            # FCF Conversion Diagnosis
            if conversion > 1:
                st.markdown(f"<div class='diag-box'>💎 <b>Caja Real:</b> Convierte más del 100% del beneficio en FCF. Calidad de beneficios suprema.</div>", unsafe_allow_html=True)
            elif conversion > 0.7:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ffa500;'>🆗 <b>Normal:</b> Conversión saludable para el sector industrial/tecnológico.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ff4b4b;'>🚨 <b>Alarma:</b> Beneficio contable alto pero poca caja real. Ojo con la contabilidad.</div>", unsafe_allow_html=True)

            # Liquidity Diagnosis
            if cr > 1.5:
                st.markdown(f"<div class='diag-box'>🛡️ <b>Solvencia:</b> Liquidez muy sólida ({cr}x) para cubrir deudas a corto plazo.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diag-box' style='border-left-color: #ff4b4b;'>🔥 <b>Liquidez:</b> Solo {cr}x. Podría sufrir si hay un frenazo en las ventas.</div>", unsafe_allow_html=True)

        with tab_sens:
            st.write("Sensibilidad del precio según variaciones en k y Crecimiento Fase 1:")
            sens_k = [k-0.01, k, k+0.01]
            sens_g = [g_fase1-0.05, g_fase1, g_fase1+0.05]
            
            matrix = []
            for ki in sens_k:
                row = []
                for gi in sens_g:
                    f1_i = sum([(fcf_actual * (1 + gi)**i) / (1 + ki)**i for i in range(1, 6)])
                    tv_i = ((fcf_actual * (1 + gi)**5) * (1 + g_perpetuo)) / (ki - g_perpetuo)
                    row.append((f1_i + (tv_i / (1 + ki)**5) - debt + cash) / shares)
                matrix.append(row)

            df_sens = pd.DataFrame(matrix, columns=[f"G1:{gi*100:.0f}%" for gi in sens_g], index=[f"k:{ki*100:.1f}%" for ki in sens_k])
            st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    # PANEL LATERAL DERECHO (CALIDAD RESUMIDA)
    with col_right:
        st.subheader("🛡️ Resumen Calidad")
        st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.1f}%")
        st.write(f"**FCF Conv.:** {conversion:.2f}x")
        st.write(f"**Liquidez:** {info.get('currentRatio', 0):.2f}x")
        st.write(f"**Deuda/Patrimonio:** {info.get('debtToEquity', 0)/100:.2f}")

    # --- 6. VEREDICTO FINAL ---
    st.divider()
    potencial = ((val_dcf / price) - 1) * 100
    st.subheader(f"🎯 Precio Objetivo: {val_dcf:.2f} {currency}")
    
    if potencial > 20:
        st.success(f"🟢 **COMPRA CLARA** | Potencial de revalorización: {potencial:.1f}%")
    elif potencial > 0:
        st.warning(f"🟡 **PRECIO JUSTO** | Potencial limitado: {potencial:.1f}%")
    else:
        st.error(f"🔴 **SOBREVALORADA** | Cotiza un {abs(potencial):.1f}% por encima de su valor intrínseco.")

except Exception as e:
    st.sidebar.error("Introduce un Ticker válido para comenzar.")
