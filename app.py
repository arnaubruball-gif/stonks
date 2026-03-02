import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v7.7 - Full Flow Audit", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-top: 10px; margin-bottom: 20px; }
    .audit-header { background-color: #1e2130; padding: 10px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
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
def get_advanced_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df['Ret'] = df['Close'].pct_change()
    
    # Z-Price (Distancia a la media)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std'] + 1e-10)
    
    # Z-Diff RMF (Presión de Flujo)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(20).sum() - df['RMF'].pct_change().rolling(20).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(20).mean()) / (diff_val.rolling(20).std() + 1e-10)
    
    # R2 Dinámico (Calidad del flujo)
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
    
    # Volatilidad
    df['Volatilidad'] = df['Ret'].rolling(20).std()
    df['Z_Vol'] = (df['Volatilidad'] - df['Volatilidad'].rolling(20).mean()) / (df['Volatilidad'].rolling(20).std() + 1e-10)
    
    return df

# --- INTERFAZ ---
assets_dict = {
    "Currencies": {"EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Smart Money Terminal")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=1)

data = get_advanced_data(ticker, temp)

if data is not None:
    # Parámetros Actuales
    z_p = data['Z_Price'].iloc[-1]
    z_d = data['Z_Diff'].iloc[-1]
    r2 = data['R2_Dynamic'].iloc[-1]
    z_v = data['Z_Vol'].iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    skew = stats.skew(data['Ret'].tail(30).dropna())

    # --- LÓGICA PROBABILÍSTICA ---
    prob_buy = 50.0 + ((-z_p * 10) if z_p < -1.5 else 0) + ((-z_d * 15) if z_d < -1.2 else 0)
    prob_sell = 50.0 + ((z_p * 10) if z_p > 1.5 else 0) + ((z_d * 15) if z_d > 1.2 else 0)
    total = prob_buy + prob_sell
    p_buy, p_sell = (prob_buy/total)*100, (prob_sell/total)*100

    tab1, tab2, tab3 = st.tabs(["📊 Probabilidad & Playbook", "🧬 Análisis de Flujo (JDetector)", "🕵️ Estructura Pro"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="prob-box" style="background-color:#00cc9611; border-color:#00cc96; color:#00cc96;"><h4>Prob. COMPRA</h4><h2>{p_buy:.1f}%</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="prob-box" style="background-color:#ff4b4b11; border-color:#ff4b4b; color:#ff4b4b;"><h4>Prob. VENTA</h4><h2>{p_sell:.1f}%</h2></div>', unsafe_allow_html=True)
        
        # Playbook Dinámico
        tipo = "TREND FOLLOWING" if hurst > 0.55 else "MEAN REVERSION" if hurst < 0.45 else "RANGO/INDECISIÓN"
        st.markdown(f'''<div class="strategy-card"><h3>📋 Playbook: {tipo}</h3>
        <p>Estrategia: {"Buscar entradas en retrocesos (Buy the dip / Sell the rally)" if hurst > 0.55 else "Operar extremos de Z-Score hacia el target 0"}</p>
        <p>Riesgo: {"FUERTE 🔥" if z_v > 1.2 else "ESTABLE ⚓"}</p></div>''', unsafe_allow_html=True)
        
        fig_price = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig_price.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("🧬 Panel de Auditoría JDetector")
        
        # 1. Métricas de Flujo Críticas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Z-Score (Precio)", f"{z_p:.2f} σ", delta="Extremo" if abs(z_p)>2 else None)
        m2.metric("Z-Diff (RMF)", f"{z_d:.2f}", delta="Divergencia" if abs(z_d)>1.4 else None)
        m3.metric("R2 (Confianza)", f"{r2:.3f}")
        m4.metric("Z-Vol (Riesgo)", f"{z_v:.2f}")

        # 2. Oscilador Dual (La visualización que pedías)
        st.write("**Oscilador Dual: Precio vs Flujo Institucional**")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff (Dinero)", line=dict(color='#ffd700', dash='dot')))
        fig_dual.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.5)
        fig_dual.add_hline(y=-2, line_dash="dash", line_color="green", opacity=0.5)
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=20)), use_container_width=True)

        # 3. Auditoría de Presión Histórica
        st.markdown('<div class="audit-header">🎯 Auditoría de Presión (Últimas 8 Velas)</div>', unsafe_allow_html=True)
        audit_df = data.tail(8).copy()
        audit_df['Veredicto'] = audit_df.apply(lambda x: "🚨 VENTA" if x['Z_Diff'] > 1.4 else ("🟢 COMPRA" if x['Z_Diff'] < -1.4 else "⚪ Neutral"), axis=1)
        st.table(audit_df[['Close', 'Z_Price', 'Z_Diff', 'R2_Dynamic', 'Veredicto']].style.format({'Close': '{:.4f}', 'Z_Price': '{:.2f}', 'Z_Diff': '{:.2f}', 'R2_Dynamic': '{:.3f}'}))

    with tab3:
        st.subheader("🕵️ Análisis de Estructura y Sesgo")
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Hurst (Persistencia)", f"{hurst:.2f}")
        c_b.metric("Skewness (Sesgo)", f"{skew:.2f}")
        c_c.metric("Kurtosis (Riesgo Cola)", f"{stats.kurtosis(data['Ret'].tail(30).dropna()):.2f}")
        
        st.divider()
        st.write("**Aceleración de Residuos (Inercia)**")
        fig_res = px.area(data.tail(100), y='Ret', color_discrete_sequence=['#00d4ff'])
        st.plotly_chart(fig_res.update_layout(template="plotly_dark", height=300), use_container_width=True)

else:
    st.error("Error al cargar datos.")
