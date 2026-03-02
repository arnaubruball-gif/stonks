import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant Terminal v4", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #0d1117; color: #8b949e; border-radius: 5px; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR MATEMÁTICO (SIN TOCAR LOGICA) ---
def calculate_hurst(series):
    if len(series) < 50: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=60)
def get_full_market_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}
    df = yf.download(ticker_id, period=p_map[t], interval=t)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    
    up = df['Returns'].clip(lower=0)
    down = -1 * df['Returns'].clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up/ema_down)))
    
    if 'Volume' in df.columns:
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    return df

# --- INTERFAZ ---
assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI"},
    "Commodities": {"Oro": "GC=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Asset Selection")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=2)

data = get_full_market_data(ticker, temp)

if data is not None:
    # Cálculo de Camarilla para usar en el gráfico
    H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
    d = H - L
    cam = {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}

    st.title(f"🚀 {nombre_activo} | Quantitative Terminal")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráfico Técnico + Camarilla", "🧬 Score de Reversión Pro", "🎯 Niveles Detalle", "🕵️ Dirección Estadística"])

    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio")])
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='#58a6ff', width=1), name="SMA 20"))
        
        # Añadir niveles Camarilla al gráfico principal
        colors = {'H4': 'rgba(255, 82, 82, 0.8)', 'H3': 'rgba(255, 160, 0, 0.8)', 'L3': 'rgba(0, 230, 118, 0.8)', 'L4': 'rgba(255, 23, 68, 0.8)'}
        for k, v in cam.items():
            fig.add_hline(y=v, line_dash="dash", line_color=colors[k], annotation_text=f" {k}", annotation_position="top right")
            
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Análisis de Probabilidad de Giro")
        c_l, c_r = st.columns([1, 1])
        
        z = data['Z-Score'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        score = 50
        if z < -1.5: score += 25
        if rsi < 35: score += 25
        if z > 1.5: score -= 25
        if rsi > 65: score -= 25
        score = max(0, min(100, score))

        with c_l:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                gauge={'axis':{'range':[0,100], 'tickwidth': 1}, 'bar':{'color':"#58a6ff"},
                       'steps':[{'range':[0,35],'color':'#ff7b72'},{'range':[35,65],'color':'#d29922'},{'range':[65,100],'color':'#3fb950'}]}))
            fig_g.update_layout(height=400, template="plotly_dark", margin=dict(t=50, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

        with c_r:
            st.write("### Veredicto del Algoritmo")
            label = "🔴 SOBRECOMPRA EXTREMA (BUSCAR VENTA)" if score < 35 else "🟢 SOBREVENTA EXTREMA (BUSCAR COMPRA)" if score > 65 else "🟡 ZONA NEUTRAL / EQUILIBRIO"
            st.info(f"**Estado Actual:** {label}")
            
            st.write("---")
            st.write(f"**Métricas en Tiempo Real:**")
            st.write(f"- **Z-Score:** {z:.2f} σ (Desviación de la media)")
            st.write(f"- **RSI:** {rsi:.1f} (Agotamiento del precio)")
            st.write(f"- **Confianza Estadística:** {abs(50-score)*2}%")

    with tab3:
        st.write("### Niveles de Referencia Camarilla")
        c_cam = st.columns(4)
        c_cam[0].metric("H4 (Short Breakout)", f"{cam['H4']:.4f}")
        c_cam[1].metric("H3 (Sell Pivot)", f"{cam['H3']:.4f}")
        c_cam[2].metric("L3 (Buy Pivot)", f"{cam['L3']:.4f}")
        c_cam[3].metric("L4 (Long Breakout)", f"{cam['L4']:.4f}")
        st.info("Los niveles H3/L3 son para reversión (rebote). Los niveles H4/L4 indican ruptura fuerte.")

    with tab4:
        st.subheader("🕵️ Motor de Inferencia Direccional")
        rets = data['Returns'].tail(60).dropna()
        skew_v = stats.skew(rets)
        kurt_v = stats.kurtosis(rets)
        hurst_v = calculate_hurst(data['Close'].values)
        
        ca, cb, cc = st.columns(3)
        ca.metric("Skewness (Sesgo)", f"{skew_v:.2f}")
        cb.metric("Kurtosis (Inercia)", f"{kurt_v:.2f}")
        cc.metric("Hurst Exponent", f"{hurst_v:.2f}")

        data['Expected'] = data['Returns'].rolling(20).mean()
        data['Residuals'] = data['Returns'] - data['Expected']
        st.write("**Aceleración Direccional (Residuos)**")
        st.plotly_chart(px.area(data.tail(100), y='Residuals', color_discrete_sequence=['#00d4ff']).update_layout(template="plotly_dark", height=350), use_container_width=True)

else:
    st.error("Error al cargar datos.")
