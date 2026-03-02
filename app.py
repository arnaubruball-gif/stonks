import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v7 - Master Confluence", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
    .confluencia-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR MATEMÁTICO DUAL ---
def calculate_hurst(series):
    if len(series) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def get_master_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "100d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 1. Z-PRICE (Elástico)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / df['Std']
    
    # 2. Z-DIFF RMF (Gasolina/Flujo)
    df['Ret'] = df['Close'].pct_change()
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    periodo_z = 20
    diff = df['Ret'].rolling(periodo_z).sum() - df['RMF'].pct_change().rolling(periodo_z).sum()
    df['Z_Diff'] = (diff - diff.rolling(periodo_z).mean()) / (diff.rolling(periodo_z).std() + 1e-10)
    
    # 3. OTROS INDICADORES
    up = df['Ret'].clip(lower=0)
    down = -1 * df['Ret'].clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up/ema_down)))
    
    if 'Volume' in df.columns:
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    return df

# --- ACTIVOS RESTAURADOS ---
assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI"},
    "Commodities": {"Oro": "GC=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Smart Money Terminal")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=2)

data = get_master_data(ticker, temp)

if data is not None:
    # Niveles Camarilla
    H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
    d = H - L
    cam = {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}

    st.title(f"🚀 {nombre_activo} | Master Engine")
    
    tab1, tab2, tab3 = st.tabs(["📊 Gráfico + Camarilla", "🧬 Score de Confluencia Dual", "🕵️ Dirección Estadística"])

    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio")])
        colors = {'H4': 'rgba(255, 82, 82, 0.6)', 'H3': 'rgba(255, 160, 0, 0.6)', 'L3': 'rgba(0, 230, 118, 0.6)', 'L4': 'rgba(255, 23, 68, 0.6)'}
        for k, v in cam.items():
            fig.add_hline(y=v, line_dash="dash", line_color=colors[k], annotation_text=f" {k}")
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        z_p = data['Z_Price'].iloc[-1]
        z_d = data['Z_Diff'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        
        # Veredicto de Confluencia Dual
        if z_p > 2.0 and z_d > 1.4:
            color, msg = "#ff4b4b", "🚨 VENTA DE ALTA PROBABILIDAD (Precio Agotado + Sin Flujo)"
        elif z_p < -2.0 and z_d < -1.4:
            color, msg = "#00cc96", "🟢 COMPRA DE ALTA PROBABILIDAD (Suelo Institucional + Acumulación RMF)"
        elif abs(z_p) > 2.0 and abs(z_d) < 1.0:
            color, msg = "#ffa500", "⚠️ TENDENCIA SANA: Precio extendido pero con gasolina real."
        else:
            color, msg = "#3d4463", "⚪ ESTADO: NEUTRAL / EQUILIBRIO"

        st.markdown(f'<div class="confluencia-box" style="background-color:{color}; color:white;">{msg}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Price (Elástico)", f"{z_p:.2f} σ")
        c2.metric("Z-Diff (Flujo RMF)", f"{z_d:.2f}")
        c3.metric("RSI Actual", f"{rsi:.1f}")

        # Gauge de Probabilidad
        score = 50
        if z_p < -2 or z_d < -1.4: score += 25
        if rsi < 30: score += 15
        if z_p > 2 or z_d > 1.4: score -= 25
        if rsi > 70: score -= 15
        
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=score, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#00d4ff"}, 'steps':[{'range':[0,35],'color':'#ff4b4b'},{'range':[35,65],'color':'#ffa500'},{'range':[65,100],'color':'#00cc96'}]}))
        fig_g.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig_g, use_container_width=True)

    with tab3:
        st.subheader("🕵️ Motor de Dirección e Inferencia")
        rets = data['Ret'].tail(60).dropna()
        skew_v = stats.skew(rets)
        kurt_v = stats.kurtosis(rets)
        hurst_v = calculate_hurst(data['Close'].values)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Skewness (Sesgo)", f"{skew_v:.2f}")
        m2.metric("Kurtosis", f"{kurt_v:.2f}")
        m3.metric("Hurst Exponent", f"{hurst_v:.2f}")

        st.divider()
        st.write("**Oscilador Dual (Z-Price vs Z-Diff)**")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff (RMF)", line=dict(color='#ffd700', dash='dot')))
        fig_dual.add_hline(y=2, line_dash="dash", line_color="red")
        fig_dual.add_hline(y=-2, line_dash="dash", line_color="green")
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=400), use_container_width=True)

        if 'Vol_ZScore' in data.columns:
            st.write("**Anomalías de Volumen (Smart Money Activity)**")
            st.plotly_chart(px.bar(data.tail(100), y='Vol_ZScore', color='Vol_ZScore', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=300), use_container_width=True)

else:
    st.error("Error al cargar datos.")
