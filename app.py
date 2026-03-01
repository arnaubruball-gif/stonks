import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURACIÓN DE TERMINAL ---
st.set_page_config(page_title="Alpha Quant Terminal Pro", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #11151c; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #11151c; border-radius: 5px; color: white; }
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

# --- FUNCIONES MATEMÁTICAS AVANZADAS ---
def calculate_hurst(series):
    """Calcula el Exponente de Hurst (H > 0.5 Tendencia, H < 0.5 Rango)"""
    if len(series) < 50: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=60)
def get_full_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}
    df = yf.download(ticker_id, period=p_map[t], interval=t)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores Técnicos
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    
    # Volatilidad y Retornos
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    
    # RVOL (Volumen Relativo)
    if 'Volume' in df.columns:
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df

def get_camarilla(df):
    H, L, C = df['High'].iloc[-2], df['Low'].iloc[-2], df['Close'].iloc[-2]
    d = H - L
    return {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}

# --- SIDEBAR ---
st.sidebar.title("💎 Quant Selector")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=3)

data = get_full_data(ticker, temp)

if data is not None:
    # --- HEADER ---
    last_price = data['Close'].iloc[-1]
    st.title(f"📈 {nombre_activo} | Terminal Intelligence")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"{last_price:.4f}", f"{data['Returns'].iloc[-1]*100:.2f}%")
    m2.metric("Z-Score", f"{data['Z-Score'].iloc[-1]:.2f} σ")
    m3.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
    m4.metric("Vol. Anual", f"{data['Vol_Anual'].iloc[-1]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis Técnico", "🧬 Probabilidad", "🎯 Camarilla", "🚀 Alpha Quant Metrics"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='orange', width=1.5), name="SMA 20"))
        fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c_l, c_r = st.columns(2)
        with c_l:
            fig_dist = px.histogram(data, x="Returns", nbins=50, title="Distribución Gaussiana de Retornos", color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_dist, use_container_width=True)
        with c_r:
            score = 0
            if data['Z-Score'].iloc[-1] < -1.5: score += 50
            if data['RSI'].iloc[-1] < 35: score += 50
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score, title={'text': "Probabilidad Reversión Alcista %"},
                gauge={'axis':{'range':[0,100]}, 'bar':{'color':"white"}, 'steps':[{'range':[0,40],'color':"red"},{'range':[70,100],'color':"green"}]}))
            fig_gauge.update_layout(height=350, template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab3:
        cam = get_camarilla(data)
        pc = st.columns(4)
        pc[0].metric("H4 (Breakout)", f"{cam['H4']:.4f}")
        pc[1].metric("H3 (Reversión)", f"{cam['H3']:.4f}")
        pc[2].metric("L3 (Reversión)", f"{cam['L3']:.4f}")
        pc[3].metric("L4 (Breakout)", f"{cam['L4']:.4f}")
        
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Scatter(x=data.index[-30:], y=data['Close'][-30:], name="Precio"))
        for l, v in cam.items():
            fig_cam.add_hline(y=v, line_dash="dash", line_color="red" if "4" in l else "yellow", annotation_text=l)
        fig_cam.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(fig_cam, use_container_width=True)

    with tab4:
        st.subheader("Métricas de Estructura Matemática")
        h_val = calculate_hurst(data['Close'].values)
        r_val = data['RVOL'].iloc[-1] if 'RVOL' in data.columns else 0
        
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Hurst Exponent", f"{h_val:.2f}")
        ac2.metric("Rel. Volume (RVOL)", f"{r_val:.2f}x")
        ac3.metric("Momentum (10p)", f"{((data['Close'].iloc[-1]/data['Close'].shift(10).iloc[-1])-1)*100:.2f}%")
        
        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.write("**Desviación Z-Score Temporal**")
            st.plotly_chart(px.line(data, x=data.index, y='Z-Score', title="Z-Score").add_hline(y=2, line_color="red").add_hline(y=-2, line_color="green"), use_container_width=True)
        with g2:
            st.write("**Anomalías de Volumen (RVOL)**")
            st.plotly_chart(px.bar(data, x=data.index, y='RVOL', title="RVOL").add_hline(y=1.5, line_color="red", line_dash="dash"), use_container_width=True)

else:
    st.error("No se pudieron cargar los datos. Verifica el Ticker.")
