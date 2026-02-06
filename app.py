import streamlit as st
import yfinance as yf
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="AI Equity Valuation Dashboard", layout="wide")

st.title("📊 Dashboard de Valoración Intrínseca")
st.write("Análisis de las 500 principales empresas de USA y Europa.")

# 1. SIDEBAR - Selección de Empresa
ticker_input = st.sidebar.text_input("Introduce el Ticker (ej: AAPL, MSFT, ITX.MC)", "AAPL")

@st.cache_data
def get_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return stock, info

try:
    stock, info = get_data(ticker_input)
    
    # 2. MÉTRICAS CABECERA
    col1, col2, col3, col4 = st.columns(4)
    precio_actual = info.get('currentPrice', 0)
    col1.metric("Precio Actual", f"{precio_actual} {info.get('currency')}")
    col2.metric("PER (P/E Ratio)", info.get('trailingPE', 'N/A'))
    col3.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    col4.metric("Market Cap", f"{info.get('marketCap', 0):,.0f}")

    # 3. INPUTS PARA VALORACIÓN (Editables por el usuario)
    st.subheader("⚙️ Parámetros de Valoración")
    c1, c2 = st.columns(2)
    g = c1.slider("Tasa de crecimiento perpetuo (g) %", 0.0, 5.0, 2.5) / 100
    k = c2.slider("Tasa de descuento (k) %", 5.0, 15.0, 8.0) / 100

    # 4. CÁLCULO DDM (GORDON GROWTH)
    st.divider()
    div_last = info.get('trailingAnnualDividendRate', 0)
    if div_last > 0 and k > g:
        valor_gordon = (div_last * (1 + g)) / (k - g)
    else:
        valor_gordon = 0

    # 5. RESULTADO Y VEREDICTO
    st.subheader("🎯 Veredicto de Valoración")
    
    if valor_gordon > 0:
        upside = ((valor_gordon / precio_actual) - 1) * 100
        
        c_v1, c_v2 = st.columns(2)
        c_v1.metric("Valor Intrínseco (DDM)", f"{valor_gordon:.2f}")
        
        if precio_actual < valor_gordon * 0.8: # Margen de seguridad del 20%
            st.success(f"🟢 INFRAVALORADA: Potencial del {upside:.2f}%")
        elif precio_actual < valor_gordon:
            st.warning(f"🟡 PRECIO JUSTO: Potencial del {upside:.2f}%")
        else:
            st.error(f"🔴 SOBREVALORADA: Potencial negativo del {upside:.2f}%")
    else:
        st.info("Esta empresa no paga dividendos o los datos no permiten el modelo DDM.")

except Exception as e:
    st.error(f"Ticker no encontrado o error en la descarga: {e}")
