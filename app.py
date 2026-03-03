import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime, time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Alpha Quant v9.5 - Sniper Edition", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-bottom: 20px; }
    .insider-box { background: linear-gradient(90deg, #1e2130 0%, #0d1117 100%); padding: 15px; border-left: 5px solid #ffd700; border-radius: 8px; margin-bottom: 20px; }
    .sniper-box { background-color: #05080a; border: 1px dashed #ffd700; padding: 20px; border-radius: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE CÁLCULO ---
def calculate_hurst(series):
    if len(series) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def get_alpha_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    dxy = yf.download("DX-Y.NYB", period=p_map[t], interval=t, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
    
    df['Ret'] = df['Close'].pct_change()
    df['Volatilidad'] = df['Ret'].rolling(20).std()
    df['Z_Vol'] = (df['Volatilidad'] - df['Volatilidad'].rolling(20).mean()) / (df['Volatilidad'].rolling(20).std() + 1e-10)
    
    # JDetector
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std20'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std20'] + 1e-10)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(20).sum() - df['RMF'].pct_change().rolling(20).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(20).mean()) / (diff_val.rolling(20).std() + 1e-10)
    
    # R2 & Edge
    r2_series = []
    for i in range(len(df)):
        if i < 20: r2_series.append(0); continue
        subset = df.iloc[i-20:i].dropna()
        try:
            X = sm.add_constant(subset['RMF'])
            r2_series.append(sm.OLS(subset['Ret'], X).fit().rsquared)
        except: r2_series.append(0)
    df['R2_Dynamic'] = r2_series
    
    df['V_Eff'] = (df['Close'].diff().abs()) / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(20).mean()) / (df['V_Eff'].rolling(20).std() + 1e-10)
    df['DXY_Corr'] = df['Ret'].rolling(20).corr(dxy['Close'].pct_change())
    
    # Datos de Ayer (High/Low/Close)
    df['Prev_High'] = df['High'].shift(1).rolling(24).max() if t=='1h' else df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1).rolling(24).min() if t=='1h' else df['Low'].shift(1)
    
    return df

# --- INTERFAZ ---
assets_dict = {
    "Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"},
    "Indices": {"Nasdaq 100": "^IXIC", "S&P 500": "^GSPC"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Sniper Terminal v9.5")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=0)

data = get_alpha_data(ticker, temp)

if data is not None:
    # Variables de Control
    z_p, z_d, z_v, r2 = data['Z_Price'].iloc[-1], data['Z_Diff'].iloc[-1], data['Z_Vol'].iloc[-1], data['R2_Dynamic'].iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    eff = data['Z_Eff'].iloc[-1]
    
    # Niveles Sniper (05:00-06:00 AM)
    p_high, p_low = data['Prev_High'].iloc[-1], data['Prev_Low'].iloc[-1]
    last_price = data['Close'].iloc[-1]
    atr = data['Std20'].iloc[-1] * 1.5

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Probabilidad", "⚡ Sniper (05-06 AM)", "🧬 Flujo JDetector", "🕵️ Estructura", "🔗 Edge"])

    with tab1:
        prob_buy = 50.0 + ((-z_p * 15) if z_p < -1.5 else 0) + ((-z_d * 20) if z_d < -1.2 else 0)
        prob_sell = 50.0 + ((z_p * 15) if z_p > 1.5 else 0) + ((z_d * 20) if z_d > 1.2 else 0)
        p_buy, p_sell = (prob_buy/(prob_buy+prob_sell))*100, (prob_sell/(prob_buy+prob_sell))*100
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="prob-box" style="color:#00cc96; border-color:#00cc96;"><h4>COMPRA</h4><h2>{p_buy:.1f}%</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="prob-box" style="color:#ff4b4b; border-color:#ff4b4b;"><h4>VENTA</h4><h2>{p_sell:.1f}%</h2></div>', unsafe_allow_html=True)
        
        tipo = "TREND" if hurst > 0.55 else "REVERSION"
        st.markdown(f'<div class="strategy-card"><h3>📋 Playbook: {tipo}</h3><p>Estado de Volatilidad: {"EXPLOSIVA 🔥" if z_v > 1.2 else "CALMA ⚓"}</p></div>', unsafe_allow_html=True)
        
        fig_p = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig_p.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("⚡ Táctica de Apertura (Sniper 05:00 CET)")
        
        # Lógica de Sesgo
        in_range = p_low < last_price < p_high
        bias = "Neutral/Rango" if in_range else ("Ruptura Alcista" if last_price > p_high else "Ruptura Bajista")
        
        st.markdown(f"""<div class="sniper-box">
            <h4 style="color:#ffd700;">Estado de la Sesión: {bias}</h4>
            <p>Precio Actual: <b>{last_price:.4f}</b> | Rango Ayer: {p_low:.4f} - {p_high:.4f}</p>
        </div>""", unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        if z_d < -1.2:
            s1.metric("ACCIÓN", "ORDEN COMPRA", "Flujo Entrando")
            s2.metric("Nivel Sugerido (Limit)", f"{(last_price - (atr*0.5)):.4f}")
            s3.metric("Take Profit", f"{data['SMA20'].iloc[-1]:.4f}")
            st.warning(f"Stop Loss (3 días): {(last_price - (atr*2)):.4f}")
        elif z_d > 1.2:
            s1.metric("ACCIÓN", "ORDEN VENTA", "Flujo Saliendo")
            s2.metric("Nivel Sugerido (Limit)", f"{(last_price + (atr*0.5)):.4f}")
            s3.metric("Take Profit", f"{data['SMA20'].iloc[-1]:.4f}")
            st.warning(f"Stop Loss (3 días): {(last_price + (atr*2)):.4f}")
        else:
            st.info("⌛ Sin confluencia clara en Z-Diff. Esperar a la vela de las 06:00.")

    with tab3:
        st.subheader("🧬 Auditoría JDetector")
        m1, m2, m3 = st.columns(3)
        m1.metric("Z-Price (Distancia)", f"{z_p:.2f}")
        m2.metric("Z-Diff (Presión)", f"{z_d:.2f}")
        m3.metric("R2 Dinámico", f"{r2:.3f}")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=350), use_container_width=True)

    with tab4:
        st.subheader("🕵️ Inferencia de Estructura")
        c_a, c_b = st.columns(2)
        c_a.metric("Hurst (Memoria)", f"{hurst:.2f}")
        c_b.metric("Z-Volatilidad", f"{z_v:.2f}")
        st.plotly_chart(px.area(data.tail(100), y='Ret', title="Inercia de Precios").update_layout(template="plotly_dark", height=300), use_container_width=True)

    with tab5:
        st.subheader("🔗 Institutional Edge")
        st.markdown(f'<div class="insider-box"><b>Correlación DXY:</b> {data["DXY_Corr"].iloc[-1]:.2f}</div>', unsafe_allow_html=True)
        st.write("**Detector de Absorción (Z-Efficiency)**")
        st.plotly_chart(px.bar(data.tail(60), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=350), use_container_width=True)

else:
    st.error("Error al cargar datos institucionales.")
