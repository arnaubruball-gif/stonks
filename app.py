import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v6.1 - Dual Z-Engine", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
    .confluencia-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR MATEMÁTICO DUAL ---
@st.cache_data(ttl=300)
def get_dual_engine_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "100d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 1. Z-SCORE ESTÁNDAR (Precio vs Media) - "El Elástico"
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / df['Std']
    
    # 2. Z-DIFF RMF (Flujo vs Retorno) - "La Gasolina"
    df['Ret'] = df['Close'].pct_change()
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    periodo_z = 20
    diff = df['Ret'].rolling(periodo_z).sum() - df['RMF'].pct_change().rolling(periodo_z).sum()
    df['Z_Diff'] = (diff - diff.rolling(periodo_z).mean()) / (diff.rolling(periodo_z).std() + 1e-10)
    
    # RSI para filtro extra
    up = df['Ret'].clip(lower=0)
    down = -1 * df['Ret'].clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up/ema_down)))
    
    return df

# --- INTERFAZ ---
assets = {"Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"], "Indices": ["^GSPC", "^IXIC"], "Crypto": ["BTC-USD", "ETH-USD"]}
ticker = st.sidebar.selectbox("Seleccionar Activo", [item for sublist in assets.values() for item in sublist])
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=2)

data = get_dual_engine_data(ticker, temp)

if data is not None:
    z_p = data['Z_Price'].iloc[-1]
    z_d = data['Z_Diff'].iloc[-1]
    rsi = data['RSI'].iloc[-1]

    tab1, tab2 = st.tabs(["📊 Gráfico Avanzado", "🧬 Master Score (Dual Engine)"])

    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Análisis de Confluencia: Precio + Flujo Monetario")
        
        # LÓGICA DE CONFLUENCIA
        # ¿El precio está lejos? (Z-Price > 2 o < -2)
        # ¿El flujo está agotado? (Z-Diff > 1.4 o < -1.4)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Price (Distancia)", f"{z_p:.2f} σ", help="Mide si el precio está lejos de su media.")
        c2.metric("Z-Diff (Flujo RMF)", f"{z_d:.2f}", help="Mide si el dinero institucional acompaña al movimiento.")
        c3.metric("RSI (Agotamiento)", f"{rsi:.1f}")

        st.divider()

        # SISTEMA DE VEREDICTO DUAL
        if z_p > 2.0 and z_d > 1.4:
            color, msg = "#ff4b4b", "🚨 ALERTA: VENTA DE ALTA PROBABILIDAD (Confluencia Dual)"
        elif z_p < -2.0 and z_d < -1.4:
            color, msg = "#00cc96", "🟢 ALERTA: COMPRA DE ALTA PROBABILIDAD (Confluencia Dual)"
        elif abs(z_p) > 2.0 and abs(z_d) < 1.0:
            color, msg = "#ffa500", "⚠️ TENDENCIA FUERTE: Precio caro pero con gasolina. No vendas."
        else:
            color, msg = "#3d4463", "⚪ ESTADO: NEUTRAL / ACUMULACIÓN"

        st.markdown(f'<div class="confluencia-box" style="background-color:{color}; color:white;">{msg}</div>', unsafe_allow_html=True)

        # Gráfico Comparativo
        st.write("**Visualización Dual (Normalizada)**")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=data.index[-50:], y=data['Z_Price'].tail(50), name="Z-Price (Elástico)", line=dict(color='#00d4ff')))
        fig_comp.add_trace(go.Scatter(x=data.index[-50:], y=data['Z_Diff'].tail(50), name="Z-Diff (Gasolina)", line=dict(color='#ffd700', dash='dot')))
        fig_comp.add_hline(y=2.0, line_dash="dash", line_color="red")
        fig_comp.add_hline(y=-2.0, line_dash="dash", line_color="green")
        st.plotly_chart(fig_comp.update_layout(template="plotly_dark", height=400), use_container_width=True)
        
        st.info("💡 **Tip de Oro:** Las mejores entradas ocurren cuando la línea azul (Precio) y la amarilla (Dinero) están AMBAS fuera de las líneas rojas/verdes.")

else:
    st.error("No se pudieron cargar los datos.")
