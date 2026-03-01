import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant Terminal v2", layout="wide")

assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI", "Nikkei 225": "^N225"},
    "Commodities": {"Oro": "GC=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F"},
    "Bonds": {"Bono 10Y USA": "^TNX", "Bono 2Y USA": "^ZT=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

# --- FUNCIONES MATEMÁTICAS ---
def calculate_hurst(series):
    """Calcula el Exponente de Hurst para determinar si hay tendencia o reversión."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=60)
def get_quant_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}
    df = yf.download(ticker_id, period=p_map[t], interval=t)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Cálculos existentes
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    
    # Volumen Relativo
    if 'Volume' in df.columns:
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df

# --- INTERFAZ ---
st.sidebar.title("💎 Quant Selector")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=3)

data = get_quant_data(ticker, temp)

if data is not None:
    st.title(f"📈 {nombre_activo} | Terminal Intelligence")
    
    # Pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análisis Técnico", 
        "🧬 Probabilidad & Volatilidad", 
        "🎯 Pivotes Camarilla", 
        "🚀 Alpha Quant Metrics"
    ])

    # [Mantener contenido de Tab 1, 2 y 3 igual que antes...]
    # (Omitido por brevedad, pero se mantiene en tu archivo real)

    with tab4:
        st.subheader("Métricas de Alta Frecuencia y Estructura")
        
        c1, c2, c3 = st.columns(3)
        
        # 1. Hurst Exponent
        hurst_val = calculate_hurst(data['Close'].values)
        with c1:
            st.metric("Hurst Exponent", f"{hurst_val:.2f}")
            if hurst_val > 0.55: st.success("Tendencial (Momentum)")
            elif hurst_val < 0.45: st.warning("Mean Reverting (Rango)")
            else: st.info("Caminata Aleatoria")

        # 2. RVOL Actual
        rvol_val = data['RVOL'].iloc[-1] if 'RVOL' in data.columns else 0
        with c2:
            st.metric("Relative Volume (RVOL)", f"{rvol_val:.2f}x")
            if rvol_val > 2: st.error("Alto Flujo Institucional")

        # 3. Z-Score Actual
        z_val = data['Z-Score'].iloc[-1]
        with c3:
            st.metric("Z-Score", f"{z_val:.2f} σ")
        
        st.divider()
        
        # GRÁFICOS DE ALPHA METRICS
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("**Gráfico Dinámico de Z-Score**")
            fig_z = px.line(data, x=data.index, y='Z-Score', title="Desviación respecto a SMA20")
            fig_z.add_hline(y=2, line_dash="dash", line_color="red")
            fig_z.add_hline(y=-2, line_dash="dash", line_color="green")
            fig_z.update_layout(template="plotly_dark")
            st.plotly_chart(fig_z, use_container_width=True)
            
        with col_g2:
            st.write("**Análisis de Volumen Relativo (RVOL)**")
            fig_rvol = px.bar(data, x=data.index, y='RVOL', title="RVOL > 1.0 = Interés Superior al Promedio")
            fig_rvol.add_hline(y=1, line_dash="dot", line_color="white")
            fig_rvol.update_layout(template="plotly_dark")
            st.plotly_chart(fig_rvol, use_container_width=True)

        # Matriz de Momentum (Cambios porcentuales multitemporales)
        st.write("**Momentum Multitemporal**")
        mom_cols = st.columns(4)
        mom_cols[0].metric("1 Vela", f"{data['Returns'].iloc[-1]*100:.2f}%")
        mom_cols[1].metric("5 Velas", f"{((data['Close'].iloc[-1]/data['Close'].shift(5).iloc[-1])-1)*100:.2f}%")
        mom_cols[2].metric("10 Velas", f"{((data['Close'].iloc[-1]/data['Close'].shift(10).iloc[-1])-1)*100:.2f}%")
        mom_cols[3].metric("20 Velas", f"{((data['Close'].iloc[-1]/data['Close'].shift(20).iloc[-1])-1)*100:.2f}%")

st.info("💡 El Exponente de Hurst ayuda a decidir la estrategia: si H > 0.5, opera rupturas; si H < 0.5, opera rebotes en soportes/resistencias.")
