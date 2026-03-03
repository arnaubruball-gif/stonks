import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v10.6 - Final Edition", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-bottom: 20px; }
    .gold-header { color: #ffd700; font-weight: bold; border-bottom: 1px solid #ffd700; padding-bottom: 5px; margin-bottom: 15px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
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
def get_master_data(ticker_id, t):
    p_map = {"1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    dxy = yf.download("DX-Y.NYB", period=p_map[t], interval=t, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
    
    W = 14 # Alta sensibilidad
    df['Ret'] = df['Close'].pct_change()
    df['SMA'] = df['Close'].rolling(W).mean()
    df['Std'] = df['Close'].rolling(W).std()
    df['Z_Price'] = (df['Close'] - df['SMA']) / (df['Std'] + 1e-10)
    
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(W).sum() - df['RMF'].pct_change().rolling(W).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(W).mean()) / (diff_val.rolling(W).std() + 1e-10)
    
    df['Skew'] = df['Ret'].rolling(30).skew()
    
    r2_s = []
    for i in range(len(df)):
        if i < W: r2_s.append(0); continue
        sub = df.iloc[i-W:i].dropna()
        try: r2_s.append(sm.OLS(sub['Ret'], sm.add_constant(sub['RMF'])).fit().rsquared)
        except: r2_s.append(0)
    df['R2'] = r2_s

    df['V_Eff'] = (df['Close'].diff().abs()) / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(W).mean()) / (df['V_Eff'].rolling(W).std() + 1e-10)
    df['DXY_Corr'] = df['Ret'].rolling(W).corr(dxy['Close'].pct_change())
    
    return df

# --- INTERFAZ ---
assets = {"Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X"}, "Indices": {"Nasdaq 100": "^IXIC"}, "Crypto": {"Bitcoin": "BTC-USD"}}
st.sidebar.title("📑 Master Sniper v10.6")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp_sel = st.sidebar.selectbox("Temp", ["1h", "4h", "1d"])
data = get_master_data(assets[cat][nombre], temp_sel)

if data is not None:
    row = data.iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Probabilidad & Playbook", "🕵️ Diagnóstico Sniper", "🧬 Historial de Flujo", "🔗 Absorción & DXY"])

    with tab1:
        # Lógica de Probabilidad (Tab 1 Original)
        prob_buy = 50.0 + ((-row['Z_Price'] * 15) if row['Z_Price'] < -1.5 else 0) + ((-row['Z_Diff'] * 20) if row['Z_Diff'] < -1.0 else 0)
        prob_sell = 50.0 + ((row['Z_Price'] * 15) if row['Z_Price'] > 1.5 else 0) + ((row['Z_Diff'] * 20) if row['Z_Diff'] > 1.0 else 0)
        p_buy, p_sell = (prob_buy/(prob_buy+prob_sell))*100, (prob_sell/(prob_buy+prob_sell))*100
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="prob-box" style="color:#00cc96; border-color:#00cc96;"><h4>Probabilidad COMPRA</h4><h2>{p_buy:.1f}%</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="prob-box" style="color:#ff4b4b; border-color:#ff4b4b;"><h4>Probabilidad VENTA</h4><h2>{p_sell:.1f}%</h2></div>', unsafe_allow_html=True)
        
        tipo = "TREND FOLLOWING" if hurst > 0.55 else "MEAN REVERSION"
        st.markdown(f'<div class="strategy-card"><h3>📋 Playbook: {tipo}</h3><p>Hurst: {hurst:.2f} | R2 Calidad: {row["R2"]:.3f}</p></div>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("Interpretación de Datos (05:00 - 06:00 AM)")
        diag_data = [
            {"Dato": "Z-Diff (Flujo)", "Estado": "🟢 COMPRA" if row['Z_Diff'] < -1.0 else "🔴 VENTA" if row['Z_Diff'] > 1.0 else "⚪ Neutral", "Significado": "Presión institucional de entrada/salida"},
            {"Dato": "Skewness", "Estado": "🚀 Alcista" if row['Skew'] > 0.15 else "📉 Bajista" if row['Skew'] < -0.15 else "⚖️ Simétrico", "Significado": "Asimetría en la velocidad de rebote"},
            {"Dato": "R2 Calidad", "Estado": "💎 ALTA" if row['R2'] > 0.1 else "💨 RUIDO", "Significado": "Fiabilidad del movimiento actual"},
            {"Dato": "Z-Price", "Estado": "⚠️ EXTREMO" if abs(row['Z_Price']) > 2 else "⚓ Normal", "Significado": "Distancia a la media (Fair Value)"}
        ]
        st.table(pd.DataFrame(diag_data))

    with tab3:
        st.markdown("<div class='gold-header'>🧬 HISTORIAL DE FLUJO INSTITUCIONAL</div>", unsafe_allow_html=True)
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(x=data.index, y=data['Z_Price'], name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_flow.add_trace(go.Scatter(x=data.index, y=data['Z_Diff'], name="Z-Diff (Dinero)", line=dict(color='#ffd700', dash='dot')))
        fig_flow.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig_flow.add_hline(y=-1.0, line_dash="dash", line_color="green")
        st.plotly_chart(fig_flow.update_layout(template="plotly_dark", height=450), use_container_width=True)

    with tab4:
        st.markdown("<div class='gold-header'>🔗 ABSORCIÓN & CORRELACIÓN DXY</div>", unsafe_allow_html=True)
        col_eff, col_dxy = st.columns(2)
        with col_eff:
            st.plotly_chart(px.bar(data.tail(50), y='Z_Eff', color='Z_Eff', title="Detector de Absorción", color_continuous_scale='RdYlGn').update_layout(template="plotly_dark"), use_container_width=True)
        with col_dxy:
            st.plotly_chart(px.line(data.tail(50), y='DXY_Corr', title="Dinámica DXY").update_layout(template="plotly_dark"), use_container_width=True)

else:
    st.error("Error al obtener datos.")
