import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v7.1 - Full JDetector Engine", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
    .confluencia-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid #30363d; }
    .audit-table { font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR MATEMÁTICO COMPLETO ---
def calculate_hurst(series):
    if len(series) < 20: return 0.5
    lags = range(2, 15)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def get_full_engine_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 1. Z-PRICE (Elástico)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std'] + 1e-10)
    
    # 2. Z-DIFF RMF (JDetector Engine)
    df['Ret'] = df['Close'].pct_change()
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    periodo_z = 20
    diff = df['Ret'].rolling(periodo_z).sum() - df['RMF'].pct_change().rolling(periodo_z).sum()
    df['Z_Diff'] = (diff - diff.rolling(periodo_z).mean()) / (diff.rolling(periodo_z).std() + 1e-10)
    
    # 3. R2 DINÁMICO (Calidad de la regresión RMF)
    r2_series = []
    for i in range(len(df)):
        if i < 20: r2_series.append(0); continue
        subset = df.iloc[i-20:i].dropna()
        try:
            r2 = sm.OLS(subset['Ret'], sm.add_constant(subset['RMF'])).fit().rsquared
            r2_series.append(r2)
        except: r2_series.append(0)
    df['R2_Dynamic'] = r2_series
    
    # 4. OTROS
    df['RSI'] = 100 - (100 / (1 + (df['Ret'].clip(lower=0).ewm(13).mean() / (-1*df['Ret'].clip(upper=0)).ewm(13).mean())))
    if 'Volume' in df.columns:
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-10)
    
    return df

# --- ACTIVOS ---
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

data = get_full_engine_data(ticker, temp)

if data is not None:
    # Camarilla
    H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
    d_range = H - L
    cam = {'H4': C+d_range*(1.1/2), 'H3': C+d_range*(1.1/4), 'L3': C-d_range*(1.1/4), 'L4': C-d_range*(1.1/2)}

    tab1, tab2, tab3 = st.tabs(["📊 Gráfico + Camarilla", "🧬 Score Confluencia & Auditoría", "🕵️ Dirección Estadística"])

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
        r2 = data['R2_Dynamic'].iloc[-1]
        
        # 1. VEREDICTO DUAL (Logica JDetector)
        if z_p > 2.0 and z_d > 1.4:
            color, msg = "#ff4b4b", "🚨 VENTA DE ALTA PROBABILIDAD (Confluencia Dual)"
        elif z_p < -2.0 and z_d < -1.4:
            color, msg = "#00cc96", "🟢 COMPRA DE ALTA PROBABILIDAD (Confluencia Dual)"
        elif abs(z_p) > 1.8:
            color, msg = "#ffa500", "⚠️ PRECIO EXTENDIDO: Posible tendencia fuerte o agotamiento."
        else:
            color, msg = "#3d4463", "⚪ ESTADO: NEUTRAL"

        st.markdown(f'<div class="confluencia-box" style="background-color:{color}; color:white;">{msg}</div>', unsafe_allow_html=True)

        # 2. MÉTRICAS CLAVE
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z-Price (Elástico)", f"{z_p:.2f} σ")
        c2.metric("Z-Diff (RMF)", f"{z_d:.2f}")
        c3.metric("R2 Dynamic (Fuerza RMF)", f"{r2:.3f}")
        c4.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")

        st.divider()

        # 3. AUDITORÍA DE PRESIÓN (Lo que faltaba del historial)
        st.write("**🎯 Auditoría de Presión (Últimas 7 velas)**")
        audit_df = data.tail(7).copy()
        audit_df['Signal'] = audit_df.apply(lambda x: "🟢 COMPRA" if x['Z_Diff'] < -1.4 else ("🚨 VENTA" if x['Z_Diff'] > 1.4 else "⚪ Neutral"), axis=1)
        audit_df_display = audit_df[['Close', 'Z_Price', 'Z_Diff', 'R2_Dynamic', 'Signal']]
        st.table(audit_df_display.style.format({'Close': '{:.4f}', 'Z_Price': '{:.2f}', 'Z_Diff': '{:.2f}', 'R2_Dynamic': '{:.3f}'}))

    with tab3:
        st.subheader("🕵️ Motor de Inferencia & Estructura")
        rets = data['Ret'].tail(60).dropna()
        m1, m2, m3 = st.columns(3)
        m1.metric("Skewness (Sesgo)", f"{stats.skew(rets):.2f}")
        m2.metric("Kurtosis", f"{stats.kurtosis(rets):.2f}")
        m3.metric("Hurst Exponent", f"{calculate_hurst(data['Close'].values):.2f}")

        st.divider()
        st.write("**Oscilador Dual: Precio vs Flujo Institucional (RMF)**")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-100:], y=data['Z_Price'].tail(100), name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-100:], y=data['Z_Diff'].tail(100), name="Z-Diff (Gasolina)", line=dict(color='#ffd700', dash='dot')))
        fig_dual.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5)
        fig_dual.add_hline(y=-2, line_dash="dash", line_color="green", opacity=0.5)
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=450), use_container_width=True)
        
        if 'Vol_ZScore' in data.columns:
            st.write("**Anomalías de Volumen Z-Score**")
            st.plotly_chart(px.bar(data.tail(100), y='Vol_ZScore', color='Vol_ZScore', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=300), use_container_width=True)

else:
    st.error("Error al cargar datos.")
