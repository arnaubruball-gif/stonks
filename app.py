import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Intelligence Terminal", layout="wide")

# Lista extendida de activos (puedes añadir los que quieras)
assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI", "Nikkei 225": "^N225"},
    "Commodities": {"Oro": "GC=F", "Plata": "SI=F", "Petróleo WTI": "CL=F", "Gas Natural": "NG=F", "Cobre": "HG=F"},
    "Bonds": {"Bono 10Y USA": "^TNX", "Bono 2Y USA": "^ZT=F", "Bono 30Y USA": "^TYX"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

# --- SIDEBAR: SELECCIÓN ---
st.sidebar.title("🔍 Asset Selector")
categoria = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
activo_nombre = st.sidebar.selectbox("Activo", list(assets_dict[categoria].keys()))
ticker = assets_dict[categoria][activo_nombre]

temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=2)
periodo_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}

# --- MOTOR DE DATOS (Individual para evitar NaNs) ---
@st.cache_data(ttl=60)
def get_asset_data(ticker_id, p, t):
    df = yf.download(ticker_id, period=p, interval=t)
    if df.empty: return None
    # Limpieza de MultiIndex si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = get_asset_data(ticker, periodo_map[temp], temp)

if data is not None:
    # --- CÁLCULOS TÉCNICOS ---
    last_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = ((last_price - prev_price) / prev_price) * 100

    # Indicadores
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    data['Std'] = data['Close'].rolling(20).std()
    data['Upper'] = data['SMA20'] + (2 * data['Std'])
    data['Lower'] = data['SMA20'] - (2 * data['Std'])
    data['Z-Score'] = (data['Close'] - data['SMA20']) / data['Std']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # --- HEADER ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Actual", f"{last_price:.4f}")
    c2.metric("Cambio Período", f"{change:.2f}%", delta=f"{change:.2f}%")
    c3.metric("Z-Score (20)", f"{data['Z-Score'].iloc[-1]:.2f}")
    c4.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")

    # --- DISEÑO DE PESTAÑAS ---
    t1, t2, t3 = st.tabs(["📈 Análisis Técnico & Volatilidad", "📊 Flujos & Volumen", "🧠 Inteligencia de Mercado"])

    with t1:
        col_main, col_stats = st.columns([3, 1])
        with col_main:
            fig = go.Figure()
            # Velas
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio"))
            # Bandas de Bollinger
            fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], line=dict(color='rgba(173, 216, 230, 0.4)'), name="Banda Superior"))
            fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], fill='tonexty', line=dict(color='rgba(173, 216, 230, 0.4)'), name="Banda Inferior"))
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.write("### Niveles Clave")
            st.write(f"**Máximo Período:** {data['High'].max():.4f}")
            st.write(f"**Mínimo Período:** {data['Low'].max():.4f}")
            st.write(f"**Distancia SMA 50:** {((last_price/data['SMA50'].iloc[-1])-1)*100:.2f}%")
            
            # Gauge de Sentimiento Técnico
            rsi_val = data['RSI'].iloc[-1]
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = rsi_val,
                title = {'text': "RSI Sentiment"},
                gauge = {'axis': {'range': [0, 100]},
                         'steps': [{'range': [0, 30], 'color': "green"}, {'range': [70, 100], 'color': "red"}],
                         'bar': {'color': "white"}}))
            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20), template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

    with t2:
        st.subheader("Análisis de Volumen Relativo (RVOL)")
        if 'Volume' in data.columns and data['Volume'].sum() > 0:
            data['Vol_MA'] = data['Volume'].rolling(20).mean()
            rvol = data['Volume'].iloc[-1] / data['Vol_MA'].iloc[-1]
            
            c_v1, c_v2 = st.columns([2,1])
            with c_v1:
                fig_vol = px.bar(data, x=data.index, y='Volume', color='Volume', title="Volumen por Vela")
                st.plotly_chart(fig_vol, use_container_width=True)
            with c_v2:
                st.metric("RVOL (Volumen Relativo)", f"{rvol:.2f}x")
                st.info("Un RVOL > 2.0 indica entrada de instituciones o manos fuertes.")
        else:
            st.warning("Datos de volumen no disponibles para este activo.")

    with t3:
        st.subheader("Detección de Anomalías y Ciclos")
        # Gráfico de Desviación Z-Score
        fig_z = px.area(data, x=data.index, y='Z-Score', title="Z-Score (Normalidad Estadística)")
        fig_z.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Sobreventa")
        st.plotly_chart(fig_z, use_container_width=True)
        
        st.markdown("""
        ### Cómo leer esta Inteligencia:
        1. **Z-Score > 2:** El precio está agotado al alza. Probabilidad de reversión a la media.
        2. **RSI < 30 + Z-Score < -2:** Zona de alta probabilidad de compra (Capitulación).
        3. **RVOL Alto + Ruptura de Banda:** Confirmación de nueva tendencia institucional.
        """)

else:
    st.error("No se pudo obtener datos para este activo. Reintenta con otra temporalidad.")
