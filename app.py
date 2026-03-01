import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Alpha Terminal Pro", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CUSTOM (MODO OSCURO TERMINAL) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- DICCIONARIO DE ACTIVOS ---
assets = {
    "Dólar (DXY)": "DX-Y.NYB",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^IXIC",
    "Bono 10Y USA": "^TNX",
    "Oro": "GC=F",
    "Petróleo WTI": "CL=F",
    "Bitcoin": "BTC-USD",
    "Cobre": "HG=F",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X"
}

# --- SIDEBAR: CONFIGURACIÓN DE TRADING ---
st.sidebar.title("🎮 Control de Terminal")
st.sidebar.divider()

temporalidad = st.sidebar.selectbox(
    "Selecciona Temporalidad", 
    ["15m", "1h", "4h", "1d"], 
    index=3 # Por defecto Diario
)

# Ajuste de periodo según temporalidad para no saturar la API
periodo_map = {"15m": "5d", "1h": "15d", "4h": "60d", "1d": "1y"}
periodo = periodo_map[temporalidad]

# --- FUNCIONES DE CÁLCULO ---
@st.cache_data(ttl=60)
def get_data(tickers, _temp, _per):
    raw_data = yf.download(list(tickers.values()), period=_per, interval=_temp)
    return raw_data['Close'], raw_data['Volume']

def calculate_indicators(data, volume, window=20):
    # Z-Score (Desviación de la media)
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    z_score = (data - ma) / std
    
    # RVOL (Volumen Relativo)
    avg_vol = volume.rolling(window=window).mean()
    rvol = volume / avg_vol
    
    return z_score, rvol

# --- OBTENCIÓN DE DATOS ---
try:
    prices, volumes = get_data(assets, temporalidad, periodo)
    z_scores, rvol_scores = calculate_indicators(prices, volumes)
    
    # --- HEADER: MÉTRICAS DE DESVIACIÓN (Z-SCORE) ---
    st.subheader(f"📊 Market Health Monitor ({temporalidad})")
    cols = st.columns(5)
    
    for i, (name, ticker) in enumerate(list(assets.items())[:10]):
        current_p = prices[ticker].iloc[-1]
        z = z_scores[ticker].iloc[-1]
        r = rvol_scores[ticker].iloc[-1]
        
        # Color de alerta según Z-Score (Extremos estadísticos)
        status_color = "normal"
        if z > 2: status_color = "inverse" # Sobrecompra
        if z < -2: status_color = "inverse" # Sobreventa
        
        with cols[i % 5]:
            st.metric(
                label=name, 
                value=f"{current_p:.2f}", 
                delta=f"Z:{z:.2f} | RVOL:{r:.1f}x",
                delta_color=status_color
            )

    st.divider()

    # --- TABS DE ANÁLISIS ---
    tab_sentimiento, tab_tecnico, tab_correlacion = st.tabs([
        "🔥 Sentimiento & Flujos", 
        "📈 Charting & Volatilidad", 
        "🔗 Correlaciones Intermarket"
    ])

    with tab_sentimiento:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Indicador de Riesgo: Ratio Cobre/Oro")
            # Ratio clave para predecir movimientos en Bonos y Dólar
            ratio = prices[assets["Cobre"]] / prices[assets["Oro"]]
            fig_ratio = px.line(ratio, title="Cobre/Oro: Subida = Risk-On (Crecimiento) | Bajada = Risk-Off (Refugio)")
            fig_ratio.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_ratio, use_container_width=True)
            
        with col2:
            st.subheader("⚠️ Alertas de Volumen")
            # Filtrar activos con RVOL inusual (>1.5)
            high_vol = rvol_scores.iloc[-1][rvol_scores.iloc[-1] > 1.5]
            if not high_vol.empty:
                for t, v in high_vol.items():
                    asset_name = [k for k, val in assets.items() if val == t][0]
                    st.warning(f"**{asset_name}**: Volumen {v:.2f}x superior a la media.")
            else:
                st.success("No se detectan anomalías de volumen.")

    with tab_tecnico:
        selected_asset = st.selectbox("Seleccionar Activo para Análisis", list(assets.keys()))
        t_id = assets[selected_asset]
        
        # Gráfico con Bandas de Desviación
        fig_pro = go.Figure()
        
        # Precio y Media
        fig_pro.add_trace(go.Scatter(x=prices.index, y=prices[t_id], name="Precio", line=dict(color='#00ffcc')))
        
        # Bandas de Bollinger (Z-Score +2/-2)
        ma = prices[t_id].rolling(20).mean()
        std = prices[t_id].rolling(20).std()
        
        fig_pro.add_trace(go.Scatter(x=prices.index, y=ma + (2*std), name="Techo (+2σ)", line=dict(dash='dash', color='red')))
        fig_pro.add_trace(go.Scatter(x=prices.index, y=ma - (2*std), name="Suelo (-2σ)", line=dict(dash='dash', color='green')))
        
        fig_pro.update_layout(template="plotly_dark", height=500, title=f"Desviación Estándar de {selected_asset}")
        st.plotly_chart(fig_pro, use_container_width=True)

    with tab_correlacion:
        st.subheader("Matriz de Correlación (Últimas 30 velas)")
        corr_matrix = prices.tail(30).corr()
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig_corr.update_layout(template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.info("💡 **Tip de Trading:** Busca activos con correlación negativa (ej. DXY vs Oro) para coberturas.")

except Exception as e:
    st.error(f"Error cargando los mercados: {e}")
    st.info("Asegúrate de tener conexión a internet y que los símbolos de Yahoo Finance sean correctos.")
