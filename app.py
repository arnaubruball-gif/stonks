import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ─── CONFIGURACIÓN DE INTERFAZ ────────────────────────────────────────────────
st.set_page_config(page_title="QUANT TERMINAL v3.0", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #05080d; color: #cdd9e5; }
    .stMetric { background: #0a1019; border: 1px solid #1a2d40; padding: 15px; border-radius: 4px; }
    .status-box { padding: 15px; border-radius: 4px; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ─── MOTOR CUANTITATIVO ───────────────────────────────────────────────────────

class QuantEngine:
    @staticmethod
    def get_market_regime(df, period=14):
        """Calcula ADX y Estado de tendencia (Fase del mercado)"""
        df = df.copy()
        high, low, close = df['High'], df['Low'], df['Close']
        
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean().iloc[-1]
        
        return adx, plus_di.iloc[-1], minus_di.iloc[-1], atr.iloc[-1]

    @staticmethod
    def calculate_markov(df):
        """Modelo simplificado de persistencia de régimen (Markov)"""
        returns = df['Close'].pct_change().dropna()
        mu, std = returns.mean(), returns.std()
        
        # Clasificar estados: 1 (Bull), 0 (Neutral), -1 (Bear)
        states = returns.apply(lambda x: 1 if x > 0.5*std else (-1 if x < -0.5*std else 0))
        last_state = states.iloc[-1]
        persistence = (states == last_state).rolling(20).mean().iloc[-1]
        
        names = {1: "BULLISH TREND", 0: "MEAN REVERSION", -1: "BEARISH TREND"}
        colors = {1: "#00e676", 0: "#ffd600", -1: "#ff1744"}
        return names[last_state], colors[last_state], persistence

    @staticmethod
    def get_volatility_metrics(df):
        """Análisis de Volatilidad: Realizada vs Z-Score vs Percentil"""
        returns = df['Close'].pct_change().dropna()
        real_vol = returns.rolling(21).std() * np.sqrt(252) * 100
        vol_mean = real_vol.rolling(60).mean()
        vol_std = real_vol.rolling(60).std()
        
        vol_z = (real_vol - vol_mean) / vol_std
        # Varianza de retornos (Comprimida vs Expandida)
        var_status = "COMPRIMIDA (Squeeze)" if vol_z.iloc[-1] < -1.5 else "EXPANDIDA"
        
        return real_vol.iloc[-1], vol_z.iloc[-1], var_status

# ─── DASHBOARD PRINCIPAL ──────────────────────────────────────────────────────

def main():
    st.title("⚡ QUANT TERMINAL PRO v3.0")
    
    with st.sidebar:
        st.header("Configuración")
        ticker = st.text_input("Activo Principal", value="EURUSD=X")
        tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
        history = st.slider("Días Histórico", 30, 200, 100)
        st.divider()
        st.caption("Correlaciones Globales")
        refs = st.multiselect("Comparar con:", ["^GSPC", "DX-Y.NYB", "GC=F", "BTC-USD"], ["^GSPC", "DX-Y.NYB", "GC=F"])

    # 1. Carga de Datos
    data = yf.download(ticker, period=f"{history}d", interval=tf)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

    if data.empty:
        st.warning("Esperando datos de Yahoo Finance...")
        return

    # 2. Ejecución de Modelos
    engine = QuantEngine()
    adx, p_di, m_di, last_atr = engine.get_market_regime(data)
    m_name, m_color, m_pers = engine.calculate_markov(data)
    r_vol, v_z, v_status = engine.get_volatility_metrics(data)
    
    # 3. UI - BLOQUE 1: ESTADO DEL MERCADO
    st.subheader("🌐 ESTADO DEL MERCADO (REGIME & MARKOV)")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"<div class='status-box' style='color:{m_color}; border-color:{m_color}'>{m_name}</div>", unsafe_allow_html=True)
    with c2:
        st.metric("Persistencia Markov", f"{m_pers:.0%}", help="Probabilidad de mantener el régimen actual")
    with c3:
        st.metric("Fuerza ADX", f"{adx:.1f}", delta="Tendencia" if adx > 25 else "Rango", delta_color="normal" if adx > 25 else "off")
    with c4:
        # Z-Diff Order Flow (Proxy)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        mf = np.where(tp > tp.shift(1), tp * data['Volume'], -tp * data['Volume'])
        rmf = pd.Series(mf).rolling(14).sum()
        z_flow = ((rmf - rmf.rolling(14).mean()) / rmf.rolling(14).std()).iloc[-1]
        st.metric("Z-Order Flow", f"{z_flow:.2f}")

    # 4. UI - BLOQUE 2: VOLATILIDAD RELATIVA (VRP)
    st.markdown("---")
    st.subheader("📊 PERFIL DE VOLATILIDAD")
    v1, v2, v3, v4 = st.columns(4)
    
    with v1:
        st.metric("Vol Realizada (Ann)", f"{r_vol:.1f}%")
    with v2:
        st.metric("Z-Score Vol", f"{v_z:.2f}σ")
    with v3:
        atr_pct = (data['High'] - data['Low']).rolling(100).rank(pct=True).iloc[-1] * 100
        st.metric("Percentil ATR", f"{atr_pct:.1f}%", delta="Anomalía" if atr_pct > 90 else None)
    with v4:
        st.metric("Varianza Retornos", v_status)

    # 5. UI - BLOQUE 3: CORRELACIONES Y GRÁFICO
    st.markdown("---")
    g1, g2 = st.columns([2, 1])

    with g1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
        # Añadir Z-Flow en el segundo panel
        z_series = ((rmf - rmf.rolling(14).mean()) / rmf.rolling(14).std())
        fig.add_trace(go.Bar(x=data.index, y=z_series, name="Z-Flow", marker_color="#ffd600", opacity=0.5), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="#05080d", plot_bgcolor="#05080d", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("Matriz de Correlación")
        if refs:
            corr_data = yf.download([ticker] + refs, period="60d", interval="1d")['Close'].corr()[ticker].drop(ticker)
            for name, val in corr_data.items():
                st.write(f"**{name} vs {ticker}**")
                st.progress(abs(val))
                st.caption(f"Coeficiente: {val:.2f}")
        else:
            st.info("Selecciona activos en la barra lateral para ver correlaciones.")

if __name__ == "__main__":
    main()
