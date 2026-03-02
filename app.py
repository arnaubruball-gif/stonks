import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant Terminal v5", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #0d1117; color: #8b949e; border-radius: 5px; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR MATEMÁTICO ---
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
    # Camarilla
    H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
    d = H - L
    cam = {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}

    st.title(f"🚀 {nombre_activo} | Terminal v5")
    
    tab1, tab2, tab4 = st.tabs(["📊 Gráfico + Camarilla", "🧬 Score de Reversión", "🕵️ Dirección Estadística"])

    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio")])
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='#00d4ff', width=1.5), name="SMA 20"))
        
        colors = {'H4': 'rgba(255, 82, 82, 0.6)', 'H3': 'rgba(255, 160, 0, 0.6)', 'L3': 'rgba(0, 230, 118, 0.6)', 'L4': 'rgba(255, 23, 68, 0.6)'}
        for k, v in cam.items():
            fig.add_hline(y=v, line_dash="dash", line_color=colors[k], annotation_text=f" {k}")
            
        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Panel de Probabilidad y Confluencia")
        z = data['Z-Score'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        
        # LOGICA DE SCORE
        score = 50
        if z < -1.5: score += 25
        if rsi < 35: score += 25
        if z > 1.5: score -= 25
        if rsi > 65: score -= 25
        score = max(5, min(95, score))

        c1, c2 = st.columns([1, 1.2])
        with c1:
            # Gauge Corregido (Barra azul funcional)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                gauge={'axis':{'range':[0,100]}, 
                       'bar':{'color':"#00d4ff", 'thickness': 0.6}, # BARRA AZUL
                       'steps':[{'range':[0,35],'color':'#ff4b4b'},{'range':[35,65],'color':'#ffa500'},{'range':[65,100],'color':'#00cc96'}],
                       'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.8, 'value': score}}))
            fig_g.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig_g, use_container_width=True)

        with c2:
            st.markdown(f"### Veredicto: {'🟢 COMPRA' if score > 65 else '🔴 VENTA' if score < 35 else '🟡 NEUTRAL'}")
            st.write("---")
            # Datos estéticos adicionales
            k1, k2 = st.columns(2)
            k1.metric("Distancia a Media", f"{z:.2f} σ")
            k1.write(f"**RSI (Agotamiento):** {rsi:.1f}")
            
            k2.metric("Volatilidad Realizada", f"{data['Vol_Anual'].iloc[-1]:.1f}%")
            k2.write(f"**Prob. de Giro:** {abs(50-score)*2}%")
            
            st.divider()
            st.write("**Niveles Críticos Camarilla:**")
            st.code(f"Resistencia (H3): {cam['H3']:.4f} | Soporte (L3): {cam['L3']:.4f}")

    with tab4:
        st.subheader("🕵️ Motor de Inferencia Direccional (Restaurado)")
        rets = data['Returns'].tail(60).dropna()
        skew_v = stats.skew(rets)
        kurt_v = stats.kurtosis(rets)
        hurst_v = calculate_hurst(data['Close'].values)
        
        ca, cb, cc = st.columns(3)
        ca.metric("Skewness (Sesgo)", f"{skew_v:.2f}", help="Positivo: Sesgo alcista latente")
        cb.metric("Kurtosis (Inercia)", f"{kurt_v:.2f}", help=">3: Probabilidad de cisne negro")
        cc.metric("Hurst Exponent", f"{hurst_v:.2f}", delta="Tendencia" if hurst_v > 0.5 else "Rango")

        st.divider()
        col_res, col_vol = st.columns(2)
        
        with col_res:
            data['Expected'] = data['Returns'].rolling(20).mean()
            data['Residuals'] = data['Returns'] - data['Expected']
            st.write("**Aceleración vs Deceleración (Residuos)**")
            fig_res = px.area(data.tail(100), y='Residuals', color_discrete_sequence=['#00d4ff'])
            fig_res.add_hline(y=0, line_dash="dash", line_color="white")
            fig_res.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_res, use_container_width=True)

        with col_vol:
            st.write("**Anomalía de Volumen (Z-Score)**")
            if 'Vol_ZScore' in data.columns:
                fig_v = px.bar(data.tail(100), y='Vol_ZScore', color='Vol_ZScore', color_continuous_scale='RdYlGn')
                fig_v.add_hline(y=2, line_dash="dot", line_color="red")
                fig_v.update_layout(template="plotly_dark", height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_v, use_container_width=True)
else:
    st.error("Error al cargar datos.")
