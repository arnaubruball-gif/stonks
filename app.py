import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v7.5 - Decision Master", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .veredicto-banner { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid #ffffff33; color: white; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
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
def get_master_engine_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 1. Retornos y Volatilidad
    df['Ret'] = df['Close'].pct_change()
    df['Volatilidad'] = df['Ret'].rolling(20).std()
    df['Z_Vol'] = (df['Volatilidad'] - df['Volatilidad'].rolling(20).mean()) / (df['Volatilidad'].rolling(20).std() + 1e-10)
    
    # 2. Z-Price (Elástico)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std'] + 1e-10)
    
    # 3. Z-Diff RMF (Flujo)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff = df['Ret'].rolling(20).sum() - df['RMF'].pct_change().rolling(20).sum()
    df['Z_Diff'] = (diff - diff.rolling(20).mean()) / (diff.rolling(20).std() + 1e-10)
    
    # 4. R2 Dinámico
    r2_series = []
    for i in range(len(df)):
        if i < 20: r2_series.append(0); continue
        subset = df.iloc[i-20:i].dropna()
        try:
            X = sm.add_constant(subset['RMF'])
            r2 = sm.OLS(subset['Ret'], X).fit().rsquared
            r2_series.append(r2)
        except: r2_series.append(0)
    df['R2_Dynamic'] = r2_series
    
    # 5. RSI y Volumen
    df['RSI'] = 100 - (100 / (1 + (df['Ret'].clip(lower=0).ewm(13).mean() / (-1*df['Ret'].clip(upper=0)).ewm(13).mean())))
    if 'Volume' in df.columns:
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-10)
    
    return df

# --- ACTIVOS ---
assets_dict = {
    "Currencies": {"EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI"},
    "Commodities": {"Oro": "GC=F", "Petróleo": "CL=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Smart Money Terminal")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=2)

data = get_master_engine_data(ticker, temp)

if data is not None:
    # Camarilla Prep
    H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
    d_range = H - L
    cam = {'H4': C+d_range*(1.1/2), 'H3': C+d_range*(1.1/4), 'L3': C-d_range*(1.1/4), 'L4': C-d_range*(1.1/2)}
    
    # Estados Actuales
    z_p, z_d, z_v = data['Z_Price'].iloc[-1], data['Z_Diff'].iloc[-1], data['Z_Vol'].iloc[-1]
    hurst_v = calculate_hurst(data['Close'].values)
    vol_label = "FUERTE 🔥" if z_v > 1.2 else "NORMAL ⚓" if z_v > -1 else "BAJA ❄️"

    tab1, tab2, tab3 = st.tabs(["📊 Gráfico & Camarilla", "🧬 Confluencia & Auditoría", "🕵️ Estadística"])

    with tab1:
        # Veredicto Acción del Precio
        sig1 = "COMPRA (Zona L3/L4)" if data['Close'].iloc[-1] < cam['L3'] else "VENTA (Zona H3/H4)" if data['Close'].iloc[-1] > cam['H3'] else "NEUTRAL (Rango)"
        bg1 = "#00cc96" if "COMPRA" in sig1 else "#ff4b4b" if "VENTA" in sig1 else "#3d4463"
        st.markdown(f'<div class="veredicto-banner" style="background-color:{bg1};">VEREDICTO PRECIO: {sig1} | VOLATILIDAD: {vol_label}</div>', unsafe_allow_html=True)

        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        for k, v in cam.items():
            fig.add_hline(y=v, line_dash="dash", line_color="white", opacity=0.3, annotation_text=k)
        st.plotly_chart(fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        # Veredicto Smart Money
        sig2 = "COMPRA FUERTE (Confluencia)" if z_d < -1.4 and z_p < -1.8 else "VENTA FUERTE (Confluencia)" if z_d > 1.4 and z_p > 1.8 else "SIN CONFIRMACIÓN"
        bg2 = "#00cc96" if "COMPRA" in sig2 else "#ff4b4b" if "VENTA" in sig2 else "#3d4463"
        st.markdown(f'<div class="veredicto-banner" style="background-color:{bg2};">VEREDICTO FLUJO RMF: {sig2} | R2: {data["R2_Dynamic"].iloc[-1]:.2f}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Price", f"{z_p:.2f}")
        c2.metric("Z-Diff", f"{z_d:.2f}")
        c3.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")

        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff (Dinero)", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=300), use_container_width=True)
        
        st.write("**🎯 Auditoría de Presión (Historial)**")
        audit_df = data.tail(7).copy()
        audit_df['Signal'] = audit_df.apply(lambda x: "🟢 COMPRA" if x['Z_Diff'] < -1.4 else ("🚨 VENTA" if x['Z_Diff'] > 1.4 else "⚪ Neutral"), axis=1)
        st.table(audit_df[['Close', 'Z_Price', 'Z_Diff', 'R2_Dynamic', 'Signal']].style.format({'Close': '{:.4f}', 'Z_Price': '{:.2f}', 'Z_Diff': '{:.2f}', 'R2_Dynamic': '{:.3f}'}))

    with tab3:
        # Veredicto Estadístico
        skew_v = stats.skew(data['Ret'].tail(30).dropna())
        sig3 = "ALCISTA (Persistente)" if hurst_v > 0.55 and skew_v > 0 else "BAJISTA (Persistente)" if hurst_v > 0.55 and skew_v < 0 else "REVERSIÓN A LA MEDIA"
        st.markdown(f'<div class="veredicto-banner" style="background-color:#1e2130;">ESTRUCTURA: {sig3} | HURST: {hurst_v:.2f}</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Skewness", f"{skew_v:.2f}")
        m2.metric("Kurtosis", f"{stats.kurtosis(data['Ret'].tail(30).dropna()):.2f}")
        m3.metric("Z-Volatilidad", f"{z_v:.2f}")

        st.write("**Aceleración y Volumen Z-Score**")
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(px.area(data.tail(60), y='Ret', title="Aceleración (Retornos)").update_layout(template="plotly_dark", height=300), use_container_width=True)
        if 'Vol_ZScore' in data.columns:
            col_b.plotly_chart(px.bar(data.tail(60), y='Vol_ZScore', color='Vol_ZScore', color_continuous_scale='RdYlGn', title="Anomalía Volumen").update_layout(template="plotly_dark", height=300), use_container_width=True)

else:
    st.error("No se pudieron cargar los datos.")
