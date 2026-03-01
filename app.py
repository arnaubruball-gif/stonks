import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURACIÓN DE TERMINAL ---
st.set_page_config(page_title="Alpha Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #11151c; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- DICCIONARIO DE ACTIVOS ---
assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI", "Nikkei 225": "^N225"},
    "Commodities": {"Oro": "GC=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F", "Gas Natural": "NG=F"},
    "Bonds": {"Bono 10Y USA": "^TNX", "Bono 2Y USA": "^ZT=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

# --- SIDEBAR ---
st.sidebar.title("💎 Quant Selector")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=3)

# --- MOTOR DE DATOS Y CÁLCULOS MATEMÁTICOS ---
@st.cache_data(ttl=60)
def get_quant_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}
    df = yf.download(ticker_id, period=p_map[t], interval=t)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 1. Indicadores Básicos
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    
    # 3. Volatilidad Anualizada
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    
    # 4. ATR (Average True Range)
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(14).mean()
    
    return df

def get_camarilla_levels(df):
    # Basado en el cierre del día anterior (o vela anterior)
    H = df['High'].iloc[-2]
    L = df['Low'].iloc[-2]
    C = df['Close'].iloc[-2]
    diff = H - L
    return {
        'H4': C + diff * (1.1 / 2), # Resistencia fuerte (Breakout)
        'H3': C + diff * (1.1 / 4), # Resistencia técnica
        'L3': C - diff * (1.1 / 4), # Soporte técnico
        'L4': C - diff * (1.1 / 2)  # Soporte fuerte (Breakout)
    }

data = get_quant_data(ticker, temp)

if data is not None:
    # --- PANELES DE MÉTRICAS ---
    last_price = data['Close'].iloc[-1]
    camarilla = get_camarilla_levels(data)
    
    st.title(f"📈 {nombre_activo} | Terminal Intelligence")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"{last_price:.4f}", f"{data['Returns'].iloc[-1]*100:.2f}%")
    m2.metric("Volatilidad Anual", f"{data['Vol_Anual'].iloc[-1]:.1f}%")
    m3.metric("Z-Score", f"{data['Z-Score'].iloc[-1]:.2f}")
    m4.metric("ATR (Fluctuación)", f"{data['ATR'].iloc[-1]:.4f}")

    tab1, tab2, tab3 = st.tabs(["📊 Análisis Técnico", "🧬 Probabilidad & Volatilidad", "🎯 Pivotes Camarilla"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candles"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='orange', width=1), name="SMA 20"))
        # Bandas
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'] + 2*data['Std'], line=dict(color='rgba(255,0,0,0.2)'), name="+2σ"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'] - 2*data['Std'], line=dict(color='rgba(0,255,0,0.2)'), name="-2σ"))
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Análisis Probabilístico Institucional")
        c_left, c_right = st.columns(2)
        
        with c_left:
            # Histograma de Probabilidades
            fig_dist = px.histogram(data, x="Returns", nbins=50, title="Distribución de Retornos (Campana Gauss)", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with c_right:
            # Gauge de Probabilidad de éxito alcista
            score = 0
            if data['Z-Score'].iloc[-1] < -1.5: score += 35
            if data['RSI'].iloc[-1] < 35: score += 35
            if data['Close'].iloc[-1] > data['SMA20'].iloc[-1]: score += 30
            
            fig_score = go.Figure(go.Indicator(
                mode = "gauge+number", value = score,
                title = {'text': "Probabilidad de Reversión Alcista (%)"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "white"},
                         'steps': [{'range': [0, 40], 'color': "red"}, {'range': [70, 100], 'color': "green"}]}))
            fig_score.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig_score, use_container_width=True)

    with tab3:
        st.subheader("Niveles Camarilla (Puntos de Reacción)")
        st.write("Estos niveles son calculados matemáticamente. H3/L3 son zonas de reversión. H4/L4 son zonas de ruptura.")
        
        # Tabla de Niveles
        piv_cols = st.columns(4)
        piv_cols[0].metric("H4 (Short Breakout)", f"{camarilla['H4']:.4f}")
        piv_cols[1].metric("H3 (Venta Técnica)", f"{camarilla['H3']:.4f}")
        piv_cols[2].metric("L3 (Compra Técnica)", f"{camarilla['L3']:.4f}")
        piv_cols[3].metric("L4 (Long Breakout)", f"{camarilla['L4']:.4f}")

        # Gráfico de Niveles para hoy
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Scatter(x=data.index[-20:], y=data['Close'][-20:], name="Precio"))
        for level, val in camarilla.items():
            fig_cam.add_hline(y=val, line_dash="dash", line_color="yellow" if "3" in level else "red", annotation_text=level)
        fig_cam.update_layout(height=400, template="plotly_dark", title="Niveles Operativos Actuales")
        st.plotly_chart(fig_cam, use_container_width=True)
        
        

else:
    st.error("Error en la conexión con el servidor de datos.")
