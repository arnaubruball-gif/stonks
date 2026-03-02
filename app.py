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
st.set_page_config(page_title="Alpha Quant v7.2 - Integrated Dual Engine", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }
    .confluencia-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR MATEMÁTICO ---
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
    
    # 1. Z-PRICE (Distancia a la Media)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std'] + 1e-10)
    
    # 2. Z-DIFF RMF (Flujo vs Retorno)
    df['Ret'] = df['Close'].pct_change()
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    periodo_z = 20
    diff = df['Ret'].rolling(periodo_z).sum() - df['RMF'].pct_change().rolling(periodo_z).sum()
    df['Z_Diff'] = (diff - diff.rolling(periodo_z).mean()) / (diff.rolling(periodo_z).std() + 1e-10)
    
    # 3. R2 DINÁMICO
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
    
    # 4. RSI
    df['RSI'] = 100 - (100 / (1 + (df['Ret'].clip(lower=0).ewm(13).mean() / (-1*df['Ret'].clip(upper=0)).ewm(13).mean())))
    
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
        
        # VEREDICTO
        if z_p > 2.0 and z_d > 1.4:
            color, msg = "#ff4b4b", "🚨 VENTA DE ALTA PROBABILIDAD (Confluencia Dual Detectada)"
        elif z_p < -2.0 and z_d < -1.4:
            color, msg = "#00cc96", "🟢 COMPRA DE ALTA PROBABILIDAD (Confluencia Dual Detectada)"
        elif abs(z_p) > 2.0:
            color, msg = "#ffa500", "⚠️ PRECIO SOBREEXTENDIDO: Precaución, tendencia fuerte sin giro de flujo."
        else:
            color, msg = "#3d4463", "⚪ ESTADO: MERCADO EN EQUILIBRIO"

        st.markdown(f'<div class="confluencia-box" style="background-color:{color}; color:white;">{msg}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z-Price (Elástico)", f"{z_p:.2f} σ")
        c2.metric("Z-Diff (RMF)", f"{z_d:.2f}")
        c3.metric("R2 Dynamic", f"{r2:.3f}")
        c4.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")

        # --- OSCILADOR DUAL REINCORPORADO ---
        st.write("**Visualización del Oscilador Dual (Precio vs Dinero)**")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price (Precio)", line=dict(color='#00d4ff', width=2)))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff (Smart Money)", line=dict(color='#ffd700', dash='dot', width=2)))
        fig_dual.add_hline(y=2, line_dash="dash", line_color="#ff4b4b", opacity=0.5)
        fig_dual.add_hline(y=-2, line_dash="dash", line_color="#00cc96", opacity=0.5)
        fig_dual.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_dual, use_container_width=True)

        st.divider()

        # AUDITORÍA DE PRESIÓN
        st.write("**🎯 Auditoría de Presión Histórica (Últimas 7 velas)**")
        audit_df = data.tail(7).copy()
        audit_df['Signal'] = audit_df.apply(lambda x: "🟢 COMPRA" if x['Z_Diff'] < -1.4 else ("🚨 VENTA" if x['Z_Diff'] > 1.4 else "⚪ Neutral"), axis=1)
        st.table(audit_df[['Close', 'Z_Price', 'Z_Diff', 'R2_Dynamic', 'Signal']].style.format({'Close': '{:.4f}', 'Z_Price': '{:.2f}', 'Z_Diff': '{:.2f}', 'R2_Dynamic': '{:.3f}'}))

    # NUEVA PESTAÑA 4: DIRECCIÓN ESTADÍSTICA
    with tab3:
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
