import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Equity Valuation Dashboard", layout="wide")

# --- ESTILO CSS PERSONALIZADO ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CACHÉ ---
@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    # Intentamos obtener el flujo de caja libre
    try:
        cashflow = stock.cashflow
        fcf = cashflow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cashflow.index else 0
    except:
        fcf = 0
    return info, fcf

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("🔍 Selección de Activo")
ticker_input = st.sidebar.text_input("Introduce Ticker (Ej: AAPL, MSFT, MC.PA, ITX.MC)", "AAPL").upper()

st.sidebar.header("⚙️ Parámetros de Valoración")
k = st.sidebar.slider("Tasa de Descuento (k) %", 5.0, 15.0, 9.0) / 100
g = st.sidebar.slider("Crecimiento Perpetuo (g) %", 0.0, 5.0, 2.5) / 100

# --- CUERPO PRINCIPAL ---
st.title(f"📊 Dashboard de Valoración: {ticker_input}")

try:
    info, fcf_actual = get_stock_data(ticker_input)
    
    # 1. MÉTRICAS DE MERCADO
    col1, col2, col3, col4 = st.columns(4)
    price = info.get('currentPrice', 1)
    currency = info.get('currency', 'USD')
    
    col1.metric("Precio Actual", f"{price} {currency}")
    col2.metric("PER (P/E)", f"{info.get('trailingPE', 'N/A')}")
    col3.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    col4.metric("FCF Actual", f"{fcf_actual/1e9:.2f}B {currency}")

    st.divider()

    # 2. MODELOS DE VALORACIÓN
    tab1, tab2 = st.tabs(["💡 Modelo de Dividendo (DDM)", "💰 Modelo de Flujo de Caja (DCF)"])

    with tab1:
        st.subheader("Modelo de Crecimiento de Gordon")
        st.latex(r"V = \frac{D_1}{k - g}")
        
        div_rate = info.get('trailingAnnualDividendRate', 0)
        if div_rate > 0:
            d1 = div_rate * (1 + g)
            val_ddm = d1 / (k - g)
            st.write(f"**Cálculo:** ({div_rate:.2f} * (1 + {g})) / ({k} - {g})")
            st.metric("Valor Intrínseco DDM", f"{val_ddm:.2f} {currency}")
        else:
            val_ddm = 0
            st.warning("Esta empresa no reparte dividendos suficientes para el modelo DDM.")

    with tab2:
        st.subheader("Discounted Cash Flow (Simplified)")
        st.latex(r"TV = \frac{FCF_n \times (1+g)}{k - g}")
        
        # Simulación simplificada a 5 años
        proyeccion_fcf = [fcf_actual * (1 + g)**i for i in range(1, 6)]
        v_presente_fcf = sum([fcf / (1 + k)**i for i, fcf in enumerate(proyeccion_fcf, 1)])
        
        terminal_value = (proyeccion_fcf[-1] * (1 + g)) / (k - g)
        v_presente_tv = terminal_value / (1 + k)**5
        
        enterprise_value = v_presente_fcf + v_presente_tv
        # Ajuste por deuda y caja para llegar al Equity Value
        cash = info.get('totalCash', 0)
        debt = info.get('totalDebt', 0)
        shares = info.get('sharesOutstanding', 1)
        
        equity_value = enterprise_value + cash - debt
        val_dcf = equity_value / shares
        
        st.write(f"**Valor Terminal Descontado:** {v_presente_tv/1e9:.2f}B")
        st.metric("Valor Intrínseco DCF", f"{val_dcf:.2f} {currency}")

    # 3. VEREDICTO FINAL
    st.divider()
    st.subheader("🎯 Resultado del Análisis")
    
    # Promediamos si ambos modelos existen, si no, usamos el disponible
    if val_ddm > 0 and val_dcf > 0:
        target = (val_ddm + val_dcf) / 2
    else:
        target = val_dcf if val_dcf > 0 else val_ddm

    if target > 0:
        upside = ((target / price) - 1) * 100
        
        col_res1, col_res2 = st.columns(2)
        col_res1.write(f"### Objetivo Promedio: **{target:.2f} {currency}**")
        
        if price < target * 0.8:
            st.success(f"🚀 **INFRAVALORADA** - Potencial del {upside:.2f}% (Margen de seguridad > 20%)")
        elif price < target:
            st.warning(f"⚖️ **PRECIO JUSTO** - Potencial del {upside:.2f}%")
        else:
            st.error(f"⚠️ **SOBREVALORADA** - Cotiza un {abs(upside):.2f}% por encima de su valor")
    else:
        st.info("No hay datos suficientes para un veredicto.")

except Exception as e:
    st.error(f"Error al procesar el Ticker: {ticker_input}. Asegúrate de que existe en Yahoo Finance.")
    st.info("Nota: Para empresas españolas usa el sufijo .MC (ej: ITX.MC, SAN.MC)")
