import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v7.6 - Probability Engine", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-top: 10px; }
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
    # Z-Scores
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std'] + 1e-10)
    
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff = df['Ret'].rolling(20).sum() - df['RMF'].pct_change().rolling(20).sum()
    df['Z_Diff'] = (diff - diff.rolling(20).mean()) / (diff.rolling(20).std() + 1e-10)
    
    # Volatilidad y Eficiencia
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
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=2)

data = get_advanced_data(ticker, temp)

if data is not None:
    # Parámetros para Probabilidad
    z_p = data['Z_Price'].iloc[-1]
    z_d = data['Z_Diff'].iloc[-1]
    z_v = data['Z_Vol'].iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    skew = stats.skew(data['Ret'].tail(30).dropna())
    
    # --- CÁLCULO DE PROBABILIDAD (BAYESIAN-LIKE) ---
    prob_buy = 50.0
    prob_sell = 50.0
    
    # Lógica de Precio
    if z_p < -2: prob_buy += 15
    if z_p > 2: prob_sell += 15
    # Lógica de Flujo (RMF)
    if z_d < -1.4: prob_buy += 20
    if z_d > 1.4: prob_sell += 20
    # Lógica de Inercia
    if skew > 0.3: prob_buy += 10
    if skew < -0.3: prob_sell += 10
    
    # Normalización
    total = prob_buy + prob_sell
    p_buy = (prob_buy / total) * 100
    p_sell = (prob_sell / total) * 100
    
    # --- DETERMINAR TIPO DE TRADE ---
    if hurst > 0.58:
        tipo_trade = "TREND FOLLOWING (Seguimiento de Tendencia)"
        desc_trade = "El mercado muestra persistencia. Buscar retrocesos a la media para entrar a favor del movimiento."
    elif hurst < 0.42:
        tipo_trade = "MEAN REVERSION (Retorno a la Media)"
        desc_trade = "El mercado está en rango o agotado. Buscar extremos (Z > 2) para apostar al regreso al centro."
    else:
        tipo_trade = "ZONA DE ACUMULACIÓN / INDECISIÓN"
        desc_trade = "Baja convicción estadística. Esperar a que el Hurst suba o el Z-Price llegue a extremos."

    tab1, tab2, tab3 = st.tabs(["📊 Probabilidad & Estrategia", "🧬 Análisis de Flujo", "🕵️ Estructura Pro"])

    with tab1:
        # Banners de Probabilidad
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'''<div class="prob-box" style="background-color: #00cc9622; border-color: #00cc96;">
                <h4 style="color: #00cc96; margin:0;">Probabilidad COMPRA</h4>
                <h2 style="margin:10px;">{p_buy:.1f}%</h2>
            </div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="prob-box" style="background-color: #ff4b4b22; border-color: #ff4b4b;">
                <h4 style="color: #ff4b4b; margin:0;">Probabilidad VENTA</h4>
                <h2 style="margin:10px;">{p_sell:.1f}%</h2>
            </div>''', unsafe_allow_html=True)

        st.markdown(f'''<div class="strategy-card">
            <h3 style="color: #00d4ff; margin-top:0;">📋 Playbook: {tipo_trade}</h3>
            <p style="font-size: 1.1em;">{desc_trade}</p>
            <p><b>Volatilidad:</b> {"🔥 ALTA - Reducir apalancamiento" if z_v > 1.2 else "⚓ ESTABLE - Operativa estándar"}</p>
        </div>''', unsafe_allow_html=True)

        # Gráfico Candlestick
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("🧬 Auditoría de Confluencia")
        st.write(f"**Anomalía de Flujo (Z-Diff):** {z_d:.2f} | **Fuerza del Precio (Z-Price):** {z_p:.2f}")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=350), use_container_width=True)
        
        st.info("💡 Si la Probabilidad de Venta es alta y el Hurst es > 0.58, no es una reversión, es un inicio de tendencia bajista.")

    with tab3:
        st.subheader("🕵️ Análisis Profundo de Estructura")
        # Tabla comparativa de Estructura
        est_data = {
            "Métrica": ["Persistencia (Hurst)", "Sesgo (Skew)", "Riesgo Cola (Kurtosis)", "Volatilidad (Z-Vol)"],
            "Valor": [f"{hurst:.2f}", f"{skew:.2f}", f"{stats.kurtosis(data['Ret'].tail(30).dropna()):.2f}", f"{z_v:.2f}"],
            "Interpretación": [
                "Tendencia" if hurst > 0.55 else "Rango",
                "Fuerza Alcista" if skew > 0 else "Fuerza Bajista",
                "Mov. Explosivo" if stats.kurtosis(data['Ret'].tail(30).dropna()) > 1 else "Normal",
                "Expansión" if z_v > 1 else "Contracción"
            ]
        }
        st.table(pd.DataFrame(est_data))
        
        st.write("**Aceleración de Retornos (Inercia)**")
        st.plotly_chart(px.area(data.tail(100), y='Ret', color_discrete_sequence=['#00d4ff']).update_layout(template="plotly_dark", height=300), use_container_width=True)

else:
    st.error("Error al cargar datos.")
