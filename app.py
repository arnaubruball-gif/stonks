import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="High-Speed Quants", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    .trade-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 8px solid; }
    .buy { background-color: #052111; border-color: #3fb950; }
    .wait { background-color: #211d05; border-color: #d29922; }
    .danger { background-color: #210505; border-color: #f85149; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LÓGICA CUANTITATIVA ---
def get_indicators(df):
    # ATR (Volatilidad para Stop Loss)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # RSI (Fuerza Relativa)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Z-Score de Precio (Distancia a la media)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Z_Score'] = (df['Close'] - df['MA20']) / df['STD20']
    
    return df

@st.cache_data(ttl=600)
def fetch_trading_data(ticker_symbol):
    try:
        dat = yf.Ticker(ticker_symbol)
        hist = dat.history(period="6mo") # 6 meses es ideal para swing
        if hist.empty: return None, None
        return dat.info, hist
    except: return None, None

# --- 3. SIDEBAR: GESTIÓN DE CAPITAL ---
st.sidebar.title("💰 Risk Management")
capital = st.sidebar.number_input("Capital Total ($)", value=10000)
risk_per_trade = st.sidebar.slider("Riesgo por Operación (%)", 0.5, 5.0, 1.0) / 100
leverage = st.sidebar.number_input("Apalancamiento (X)", value=5, min_value=1)

ticker = st.sidebar.text_input("Ticker para Swing", "TSLA").upper()
info, hist = fetch_trading_data(ticker)

# --- 4. DASHBOARD ---
if info and not hist.empty:
    hist = get_indicators(hist)
    last_price = hist['Close'].iloc[-1]
    last_atr = hist['ATR'].iloc[-1]
    last_rsi = hist['RSI'].iloc[-1]
    last_z = hist['Z_Score'].iloc[-1]
    
    st.title(f"⚡ Trading Analysis: {ticker}")
    
    # MÉTRICAS DE ENTRADA
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"{last_price:.2f}")
    c2.metric("RSI (14d)", f"{last_rsi:.1f}")
    c3.metric("ATR (Volatilidad)", f"{last_atr:.2f}")
    c4.metric("Z-Score Precio", f"{last_z:.2f}")

    st.divider()

    # --- PESTAÑAS DE OPERATIVA ---
    t_signal, t_risk, t_chart = st.tabs(["🎯 SEÑAL QUANTO", "🛡️ CALCULADORA RIESGO", "📈 GRÁFICO TÉCNICO"])

    with t_signal:
        st.subheader("Estado de la Tendencia")
        
        
        # Lógica de señales
        score = 0
        if last_rsi > 50: score += 1
        if last_price > hist['MA20'].iloc[-1]: score += 1
        if last_z < 1.5: score += 1 # No está excesivamente caro

        if score == 3:
            st.markdown(f"<div class='trade-box buy'><h2>🚀 SEÑAL: LONG (COMPRA)</h2>Fuerza confirmada. Z-Score saludable.</div>", unsafe_allow_html=True)
        elif score == 2:
            st.markdown(f"<div class='trade-box wait'><h2>⚖️ SEÑAL: NEUTRAL</h2>Esperar confirmación de RSI o Media Móvil.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='trade-box danger'><h2>⚠️ SEÑAL: EVITAR</h2>Debilidad técnica o sobrecompra extrema.</div>", unsafe_allow_html=True)

    with t_risk:
        st.subheader("Planificación de la Operación")
        # Cálculo de Stop Loss dinámico (2.5x ATR)
        stop_loss = last_price - (2.5 * last_atr)
        distancia_stop = last_price - stop_loss
        
        # Tamaño de la posición (Cuanto comprar para arriesgar solo el % del capital)
        dinero_en_riesgo = capital * risk_per_trade
        num_acciones = dinero_en_riesgo / distancia_stop
        posicion_nominal = num_acciones * last_price
        margen_requerido = posicion_nominal / leverage
        
        r1, r2 = st.columns(2)
        with r1:
            st.write(f"**Niveles Técnicos:**")
            st.info(f"🛑 Stop-Loss Sugerido: **{stop_loss:.2f}**")
            st.success(f"🎯 Take-Profit Sugerido (Risk 1:2): **{last_price + (distancia_stop * 2):.2f}**")
        
        with r2:
            st.write(f"**Gestión de Lotes:**")
            st.write(f"Acciones a comprar: `{int(num_acciones)}` unidades")
            st.write(f"Valor nominal de la posición: `${posicion_nominal:.2f}`")
            st.write(f"Margen requerido (con apalancamiento): `${margen_requerido:.2f}`")
            

    with t_chart:
        st.subheader("Movimiento de los últimos 6 meses")
        st.line_chart(hist[['Close', 'MA20']])
        st.caption("La línea azul es el precio, la roja es la media móvil de 20 periodos (tu soporte dinámico).")

else:
    st.error("No se pudo conectar con Yahoo Finance. Revisa el Ticker.")

# --- 5. RECORDATORIO DE RIESGO ---
st.sidebar.warning("🚨 El apalancamiento aumenta el riesgo. Nunca arriesgues más del 2% de tu capital por operación.")
