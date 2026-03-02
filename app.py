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
    
    # Indicadores Técnicos Clave
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    
    # RSI
    up = df['Returns'].clip(lower=0)
    down = -1 * df['Returns'].clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up/ema_down)))
    
    # Volumen Metrics
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
    # --- HEADER ---
    st.title(f"🚀 {nombre_activo} | Quantitative Terminal")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"{data['Close'].iloc[-1]:.4f}", f"{data['Returns'].iloc[-1]*100:.2f}%")
    m2.metric("Z-Score Price", f"{data['Z-Score'].iloc[-1]:.2f} σ")
    m3.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")
    m4.metric("Volatility", f"{data['Vol_Anual'].iloc[-1]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráfico Técnico", "🧬 Curva de Probabilidad", "🎯 Niveles Camarilla", "🕵️ Dirección Estadística"])

    # Pestañas anteriores (Lógica Intacta)
    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='#58a6ff')))
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c_l, c_r = st.columns([1.5, 1])
        with c_l:
            rets = data['Returns'].dropna()
            fig_kde = px.histogram(rets, nbins=100, histnorm='probability density', opacity=0.3, color_discrete_sequence=['#58a6ff'])
            x = np.linspace(rets.min(), rets.max(), 200)
            kde = stats.gaussian_kde(rets)
            fig_kde.add_trace(go.Scatter(x=x, y=kde(x), line=dict(color='#58a6ff', width=3)))
            fig_kde.add_vline(x=rets.iloc[-1], line_dash="dash", line_color="red")
            fig_kde.update_layout(template="plotly_dark", title="Distribución de Retornos")
            st.plotly_chart(fig_kde, use_container_width=True)
        with c_r:
            score = 50
            if data['Z-Score'].iloc[-1] < -1.5: score += 25
            if data['RSI'].iloc[-1] < 35: score += 25
            if data['Z-Score'].iloc[-1] > 1.5: score -= 25
            if data['RSI'].iloc[-1] > 65: score -= 25
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=score, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#58a6ff"}, 'steps':[{'range':[0,35],'color':'#ff7b72'},{'range':[65,100],'color':'#3fb950'}]}))
            fig_g.update_layout(height=350, template="plotly_dark")
            st.plotly_chart(fig_g, use_container_width=True)

    with tab3:
        H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
        d = H - L
        cam = {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}
        st.write("### Niveles Institucionales")
        st.plotly_chart(px.line(data.tail(50), y='Close', title="Camarilla Levels").add_hline(y=cam['H3'], line_color="orange").add_hline(y=cam['L3'], line_color="orange"), use_container_width=True)

    # NUEVA PESTAÑA 4: DIRECCIÓN ESTADÍSTICA
    with tab4:
        st.subheader("🕵️ Motor de Inferencia Direccional")
        
        # 1. Cálculos Estadísticos Superiores
        recent_rets = data['Returns'].tail(60).dropna()
        skew_val = stats.skew(recent_rets)
        kurt_val = stats.kurtosis(recent_rets)
        hurst_val = calculate_hurst(data['Close'].values)
        
        ca, cb, cc = st.columns(3)
        with ca:
            st.metric("Skewness (Sesgo)", f"{skew_val:.2f}")
            st.caption("Si > 0: Fuerza alcista latente. Si < 0: Presión bajista.")
        with cb:
            st.metric("Kurtosis (Inercia)", f"{kurt_val:.2f}")
            st.caption("Si > 3: Riesgo de movimiento violento (Fat Tails).")
        with cc:
            st.metric("Hurst Exponent", f"{hurst_val:.2f}")
            st.caption("> 0.55: Tendencia sólida. < 0.45: Rango/Reversión.")

        st.divider()

        # 2. Análisis de Residuos y Aceleración
        # Detectamos si el precio se acelera más allá de su comportamiento promedio
        data['Expected'] = data['Returns'].rolling(20).mean()
        data['Residuals'] = data['Returns'] - data['Expected']
        
        col_res, col_vol = st.columns(2)
        
        with col_res:
            st.write("**Aceleración Direccional (Residuos)**")
            fig_res = px.area(data.tail(100), y='Residuals', color_discrete_sequence=['#00d4ff'])
            fig_res.add_hline(y=0, line_dash="dash", line_color="white")
            fig_res.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption("Residuos positivos + Skew positivo = Confirmación de subida con inercia.")

        with col_vol:
            st.write("**Anomalía de Volumen (Z-Score)**")
            if 'Vol_ZScore' in data.columns:
                fig_v = px.bar(data.tail(100), y='Vol_ZScore', color='Vol_ZScore', color_continuous_scale='RdYlGn')
                fig_v.add_hline(y=2, line_dash="dot", line_color="red")
                fig_v.update_layout(template="plotly_dark", height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_v, use_container_width=True)
                st.caption("Volumen Z > 2 indica entrada masiva institucional (posible techo o suelo).")

else:
    st.error("Error al cargar datos.")
