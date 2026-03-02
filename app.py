import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v7.9 - Institutional Suite", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-bottom: 20px; }
    .insider-box { background: linear-gradient(90deg, #1e2130 0%, #0d1117 100%); padding: 15px; border-left: 5px solid #ffd700; border-radius: 8px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE CÁLCULO INTEGRADO ---
def calculate_hurst(series):
    if len(series) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def get_full_master_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    # Descarga principal
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    dxy = yf.download("DX-Y.NYB", period=p_map[t], interval=t, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
    
    # 1. Básicos y Volatilidad
    df['Ret'] = df['Close'].pct_change()
    df['Volatilidad'] = df['Ret'].rolling(20).std()
    df['Z_Vol'] = (df['Volatilidad'] - df['Volatilidad'].rolling(20).mean()) / (df['Volatilidad'].rolling(20).std() + 1e-10)
    
    # 2. JDetector (Z-Price & Z-Diff)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Close'].rolling(20).std() + 1e-10)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(20).sum() - df['RMF'].pct_change().rolling(20).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(20).mean()) / (diff_val.rolling(20).std() + 1e-10)
    
    # 3. R2 Dinámico
    r2_series = []
    for i in range(len(df)):
        if i < 20: r2_series.append(0); continue
        subset = df.iloc[i-20:i].dropna()
        try:
            X = sm.add_constant(subset['RMF'])
            r2_series.append(sm.OLS(subset['Ret'], X).fit().rsquared)
        except: r2_series.append(0)
    df['R2_Dynamic'] = r2_series
    
    # 4. EDGE: V-Efficiency & DXY Corr
    df['V_Eff'] = (df['Close'].diff().abs()) / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(20).mean()) / (df['V_Eff'].rolling(20).std() + 1e-10)
    df['DXY_Corr'] = df['Ret'].rolling(20).corr(dxy['Close'].pct_change())
    
    return df

# --- INTERFAZ ---
assets_dict = {
    "Currencies": {"EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"Nasdaq 100": "^IXIC", "S&P 500": "^GSPC", "DAX 40": "^GDAXI"},
    "Commodities": {"Oro": "GC=F", "Petróleo": "CL=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Master Quant Terminal")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=1)

data = get_full_master_data(ticker, temp)

if data is not None:
    # Variables de Control
    z_p, z_d, z_v, r2 = data['Z_Price'].iloc[-1], data['Z_Diff'].iloc[-1], data['Z_Vol'].iloc[-1], data['R2_Dynamic'].iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    skew = stats.skew(data['Ret'].tail(30).dropna())
    eff = data['Z_Eff'].iloc[-1]
    corr_dxy = data['DXY_Corr'].iloc[-1]

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Probabilidad", "🧬 Flujo JDetector", "🕵️ Estructura Pro", "🔗 Institutional Edge"])

    with tab1:
        prob_buy = 50.0 + ((-z_p * 10) if z_p < -1.5 else 0) + ((-z_d * 15) if z_d < -1.2 else 0)
        prob_sell = 50.0 + ((z_p * 10) if z_p > 1.5 else 0) + ((z_d * 15) if z_d > 1.2 else 0)
        p_buy, p_sell = (prob_buy/(prob_buy+prob_sell))*100, (prob_sell/(prob_buy+prob_sell))*100
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="prob-box" style="color:#00cc96; border-color:#00cc96;"><h4>COMPRA</h4><h2>{p_buy:.1f}%</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="prob-box" style="color:#ff4b4b; border-color:#ff4b4b;"><h4>VENTA</h4><h2>{p_sell:.1f}%</h2></div>', unsafe_allow_html=True)
        
        tipo = "TREND FOLLOWING" if hurst > 0.55 else "MEAN REVERSION" if hurst < 0.45 else "RANGO"
        st.markdown(f'<div class="strategy-card"><h3>📋 Playbook: {tipo}</h3><p>Riesgo: {"FUERTE 🔥" if z_v > 1.2 else "ESTABLE ⚓"}</p></div>', unsafe_allow_html=True)
        
        fig_p = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig_p.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("🧬 Panel JDetector & Auditoría")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Z-Price", f"{z_p:.2f}")
        col2.metric("Z-Diff", f"{z_d:.2f}")
        col3.metric("R2 Dynamic", f"{r2:.3f}")
        col4.metric("Z-Vol", f"{z_v:.2f}")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=350), use_container_width=True)
        
        audit_df = data.tail(7).copy()
        audit_df['Veredicto'] = audit_df.apply(lambda x: "🚨 VENTA" if x['Z_Diff'] > 1.4 else ("🟢 COMPRA" if x['Z_Diff'] < -1.4 else "⚪ Neutral"), axis=1)
        st.table(audit_df[['Close', 'Z_Price', 'Z_Diff', 'R2_Dynamic', 'Veredicto']])

    with tab3:
        st.subheader("🕵️ Inferencia de Régimen")
        m1, m2, m3 = st.columns(3)
        m1.metric("Hurst", f"{hurst:.2f}")
        m2.metric("Skewness", f"{skew:.2f}")
        m3.metric("Kurtosis", f"{stats.kurtosis(data['Ret'].tail(30).dropna()):.2f}")
        st.plotly_chart(px.area(data.tail(100), y='Ret', title="Aceleración de Retornos").update_layout(template="plotly_dark", height=300), use_container_width=True)

    with tab4:
        st.subheader("🔗 Correlaciones Ocultas y Absorción Institucional")
        st.markdown(f"""<div class="insider-box"><b>DXY Alignment:</b> {corr_dxy:.2f} <br>
        <small>Detección de flujo inverso respecto al Dólar Index.</small></div>""", unsafe_allow_html=True)
        
        c_eff, c_corr = st.columns(2)
        with c_eff:
            st.write("**Z-Efficiency (Detector de Absorción)**")
            st.caption("Valores < -1 indican que el volumen institucional está absorbiendo el movimiento sin desplazar el precio.")
            st.plotly_chart(px.bar(data.tail(60), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=300), use_container_width=True)
        with c_corr:
            st.write("**Correlación Dinámica vs DXY**")
            st.plotly_chart(px.line(data.tail(60), y='DXY_Corr').update_layout(template="plotly_dark", height=300), use_container_width=True)

else:
    st.error("Error en el motor de datos.")
