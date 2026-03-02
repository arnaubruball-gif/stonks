import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Quant Intelligence Terminal", layout="wide")

# Estilos CSS para un look de Terminal Bloomberg/Reuters
st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #0d1117; 
        border-radius: 8px 8px 0px 0px; 
        padding: 10px 20px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #161b22 !important; 
        color: #58a6ff !important; 
        border-bottom: 2px solid #58a6ff !important; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- ACTIVOS ---
assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI", "Nikkei 225": "^N225"},
    "Commodities": {"Oro": "GC=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F", "Gas Natural": "NG=F"},
    "Bonds": {"Bono 10Y USA": "^TNX", "Bono 2Y USA": "^ZT=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

# --- MOTOR CUANTITATIVO ---
@st.cache_data(ttl=60)
def get_full_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}
    df = yf.download(ticker_id, period=p_map[t], interval=t)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores Base
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    
    # RSI Avanzado
    up = df['Returns'].clip(lower=0)
    down = -1 * df['Returns'].clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up/ema_down)))
    
    if 'Volume' in df.columns:
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    return df

# --- SIDEBAR ---
st.sidebar.title("📑 Asset Intelligence")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=2)

data = get_full_data(ticker, temp)

if data is not None:
    last_price = data['Close'].iloc[-1]
    z_actual = data['Z-Score'].iloc[-1]
    rsi_actual = data['RSI'].iloc[-1]
    
    st.title(f"🚀 {nombre_activo} Intelligence Dashboard")
    
    # 1. HEADER MÉTRICAS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Price", f"{last_price:.4f}", f"{data['Returns'].iloc[-1]*100:.2f}%")
    m2.metric("Z-Score (σ)", f"{z_actual:.2f}")
    m3.metric("RSI (14)", f"{rsi_actual:.1f}")
    m4.metric("Vol. Anualizada", f"{data['Vol_Anual'].iloc[-1]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráfico", "🧬 Probabilidad & Pricing", "🎯 Camarilla", "🚀 Alpha Quant"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='#58a6ff', width=1.5), name="SMA 20"))
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Modelado de Reversión y Probabilidad")
        c_l, c_r = st.columns([1.5, 1])
        
        with c_l:
            # CURVA DE DENSIDAD PRO
            returns_clean = data['Returns'].dropna()
            fig_dist = px.histogram(returns_clean, nbins=100, histnorm='probability density', opacity=0.2, color_discrete_sequence=['#58a6ff'])
            x_range = np.linspace(returns_clean.min(), returns_clean.max(), 200)
            kde = stats.gaussian_kde(returns_clean)
            fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range), line=dict(color='#58a6ff', width=3), name="KDE"))
            fig_dist.add_vline(x=returns_clean.iloc[-1], line_dash="dash", line_color="#ff7b72", annotation_text="ACTUAL")
            fig_dist.update_layout(template="plotly_dark", height=400, showlegend=False, title="Distribución Probabilística de Retornos")
            st.plotly_chart(fig_dist, use_container_width=True)

        with c_r:
            # GAUGE DE PROBABILIDAD MEJORADO
            score = 0
            # Lógica de pesos: Z-score es el 60% del peso, RSI el 40%
            if z_actual < -1.5: score += 30
            if z_actual < -2.5: score += 30
            if rsi_actual < 35: score += 20
            if rsi_actual < 25: score += 20
            
            # Si estamos en corto
            if z_actual > 1.5: score = 100 - 30
            if z_actual > 2.5: score = 100 - 60
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score Reversión Alcista %", 'font': {'size': 20}},
                delta={'reference': 50, 'increasing': {'color': "#3fb950"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#58a6ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#30363d",
                    'steps': [
                        {'range': [0, 30], 'color': '#ff7b72'}, # Rojo (Baja prob)
                        {'range': [30, 70], 'color': '#d29922'}, # Ambar
                        {'range': [70, 100], 'color': '#3fb950'} # Verde (Alta prob)
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig_gauge.update_layout(height=350, template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            veredicto = "REVERSIÓN PROBABLE" if score > 70 else "CONTINUIDAD / RANGO" if score > 30 else "AGOTAMIENTO ALCISTA"
            st.markdown(f"<h3 style='text-align: center; color: #58a6ff;'>{veredicto}</h3>", unsafe_allow_html=True)

    with tab3:
        # CAMARILLA NIVEL PRO
        H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
        d = H - L
        cam = {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}
        
        cols_cam = st.columns(4)
        for i, (k, v) in enumerate(cam.items()):
            color = "inverse" if "4" in k else "normal"
            cols_cam[i].metric(k, f"{v:.4f}", delta="Punto de Reacción", delta_color=color)

        fig_cam = go.Figure()
        fig_cam.add_trace(go.Scatter(x=data.index[-40:], y=data['Close'][-40:], name="Price", line=dict(color="#58a6ff", width=2)))
        for k, v in cam.items():
            line_c = "#ff7b72" if "4" in k else "#d29922"
            fig_cam.add_hline(y=v, line_dash="dot", line_color=line_c, annotation_text=k)
        fig_cam.update_layout(height=450, template="plotly_dark", title="Mapa de Liquidez Camarilla (Static Levels)")
        st.plotly_chart(fig_cam, use_container_width=True)

    with tab4:
        # ALPHA METRICS
        h_val = calculate_hurst(data['Close'].values)
        st.subheader("Métricas de Estructura Fractal")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Hurst Exponent", f"{h_val:.2f}", "Tendencia" if h_val > 0.5 else "Rango")
        if 'RVOL' in data.columns:
            c2.metric("Relative Volume", f"{data['RVOL'].iloc[-1]:.2f}x", "Institutional Activity")
        c3.metric("Z-Score Distance", f"{z_actual:.2f} σ", "Standard Deviations")
        
        st.divider()
        st.plotly_chart(px.line(data, x=data.index, y='Z-Score', title="Historical Z-Score Oscillations").add_hline(y=2, line_color="red").add_hline(y=-2, line_color="green"), use_container_width=True)

else:
    st.error("Error cargando datos. Revisa la conexión o el Ticker.")
