import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────
st.set_page_config(page_title="QUANT TERMINAL PRO", layout="wide")

# Estilos de Terminal
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #04070d; color: #cdd9e5; }
    .stMetric { background: #0a1019; border: 1px solid #1a2d40; padding: 15px; border-radius: 4px; }
    .status-box { padding: 15px; border-radius: 4px; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ─── LÓGICA CUANTITATIVA ─────────────────────────────────────────────────────

def get_quantitative_data(df, period=14):
    df = df.copy()
    close = df['Close']
    high, low = df['High'], df['Low']
    
    # --- 1. ESTADO DEL MERCADO (ADX & REGIME) ---
    # True Range
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # ADX (Simplificado)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean().iloc[-1]
    
    # --- 2. VOLATILIDAD RELATIVA (VRP & ATR PERCENTILE) ---
    returns = close.pct_change().dropna()
    realized_vol = returns.rolling(21).std() * np.sqrt(252) * 100
    vol_z = (realized_vol - realized_vol.rolling(60).mean()) / realized_vol.rolling(60).std()
    
    # Percentil de ATR (¿Es el movimiento actual extremo?)
    current_atr = atr.iloc[-1]
    atr_percentile = (atr.rolling(100).rank(pct=True).iloc[-1]) * 100
    
    # --- 3. ORDER FLOW (Z-DIFF) ---
    tp = (high + low + close) / 3
    # Si no hay volumen (Forex), usamos el rango como proxy de actividad
    vol_proxy = df['Volume'] if df['Volume'].nunique() > 5 else (high - low)
    mf = np.where(tp > tp.shift(1), tp * vol_proxy, -tp * vol_proxy)
    rmf = pd.Series(mf).rolling(period).sum()
    z_diff = ((rmf - rmf.rolling(period).mean()) / rmf.rolling(period).std()).iloc[-1]
    
    return {
        "adx": adx,
        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],
        "vol_realized": realized_vol.iloc[-1],
        "vol_z": vol_z.iloc[-1],
        "atr_pct": atr_percentile,
        "z_diff": z_diff
    }

# ─── INTERFAZ DASHBOARD ──────────────────────────────────────────────────────

st.title("⚡ QUANTUM TERMINAL v2.0")
st.caption("Market Regime • Volatility Squeeze • Correlation Matrix")

with st.sidebar:
    st.header("Configuración")
    ticker = st.text_input("Símbolo Principal", value="EURUSD=X")
    horizon = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
    st.divider()
    st.caption("Activos Correlacionados")
    c_ref1 = st.text_input("Ref 1", "^GSPC") # S&P500
    c_ref2 = st.text_input("Ref 2", "DX-Y.NYB") # DXY
    c_ref3 = st.text_input("Ref 3", "GC=F") # Oro

# Carga de datos
data = yf.download(ticker, period="100d", interval=horizon)
if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

if not data.empty:
    q = get_quantitative_data(data)

    # ─── SECCIÓN 1: ESTADO DEL MERCADO ───
    st.subheader("🌐 ESTADO DEL MERCADO")
    c1, c2, c3, c4 = st.columns(4)
    
    # Determinar Régimen
    if q['adx'] > 25:
        regime = "TENDENCIA"
        color = "#00e5ff"
        trend_dir = "ALCISTA" if q['plus_di'] > q['minus_di'] else "BAJISTA"
    else:
        regime = "RANGO / LATERAL"
        color = "#ffd600"
        trend_dir = "NEUTRAL"

    with c1:
        st.markdown(f"<div class='status-box' style='color:{color}'>{regime}</div>", unsafe_allow_html=True)
    with c2:
        st.metric("Fuerza ADX", f"{q['adx']:.1f}", help=">25 indica tendencia fuerte")
    with c3:
        st.metric("Dirección", trend_dir)
    with c4:
        st.metric("Z-Order Flow", f"{q['z_diff']:.2f}")

    # ─── SECCIÓN 2: PERFIL DE VOLATILIDAD ───
    st.markdown("---")
    st.subheader("📊 VOLATILIDAD Y RIESGO")
    v1, v2, v3, v4 = st.columns(4)
    
    with v1:
        # Varianza comprimida o expandida
        st.metric("Vol Realizada (Ann)", f"{q['vol_realized']:.1f}%")
    with v2:
        st.metric("Z-Score Vol", f"{q['vol_z']:.2f}σ", 
                  help=">2σ indica riesgo de reversión inminente")
    with v3:
        # Percentil de ATR
        status_atr = "ANOMALÍA" if q['atr_pct'] > 90 else "NORMAL"
        st.metric("Percentil ATR", f"{q['atr_pct']:.1f}%", delta=status_atr)
    with v4:
        squeeze = "SQUEEZE" if q['vol_z'] < -1.5 else "EXPANSIÓN"
        st.metric("Fase Vol", squeeze)

    # ─── SECCIÓN 3: CORRELACIONES Y GRÁFICO ───
    st.markdown("---")
    g1, g2 = st.columns([2, 1])

    with g1:
        st.subheader("Análisis de Precio")
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig.update_layout(template="plotly_dark", height=450, paper_bgcolor="#04070d", plot_bgcolor="#04070d", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("Matriz de Correlación")
        # Descarga de activos correlacionados
        corr_assets = [ticker, c_ref1, c_ref2, c_ref3]
        c_data = yf.download(corr_assets, period="60d", interval="1d")['Close'].corr()[ticker].drop(ticker)
        
        for name, val in c_data.items():
            st.write(f"**{name}**")
            color_bar = "#00e676" if val > 0 else "#ff1744"
            st.progress(abs(val))
            st.caption(f"Coeficiente de Pearson: {val:.2f}")

    # ─── MÓDULO EXTRA: DETECCIÓN DE DIVERGENCIAS ───
    st.divider()
    st.subheader("🔍 Smart Money Divergence")
    
    # Lógica simple de divergencia
    price_change = data['Close'].pct_change(10).iloc[-1]
    flow_change = q['z_diff']
    
    if price_change > 0 and flow_change < -1:
        st.warning("⚠️ DIVERGENCIA BAJISTA: El precio sube pero el flujo de dinero institucional cae (Distribución).")
    elif price_change < 0 and flow_change > 1:
        st.success("✅ DIVERGENCIA ALCISTA: El precio cae pero el flujo de dinero institucional sube (Acumulación).")
    else:
        st.write("No se detectan divergencias institucionales significativas en este momento.")

else:
    st.warning("Cargando datos...")    r_val = H - L
    df['H4'], df['H3'] = C + r_val * (1.1/2), C + r_val * (1.1/4)
    df['L3'], df['L4'] = C - r_val * (1.1/4), C - r_val * (1.1/2)
    
    return df

def get_dynamic_diagnosis(z_d, z_p, skew, r2):
    diag = []
    if z_d < -1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🟢 COMPRA", "Significado": "Entrada de dinero institucional"})
    elif z_d > 1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🔴 VENTA", "Significado": "Salida de dinero / Distribución"})
    else: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "⚪ Neutral", "Significado": "Sin presión clara"})
    
    if abs(z_p) > 2: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚠️ EXTREMO", "Significado": "Precio sobreextendido. Reversión probable."})
    else: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚓ Estable", "Significado": "Zona de Fair Value"})
    
    if skew > 0.2: diag.append({"Dato": "Skewness", "Estado": "🚀 Alcista", "Significado": "Sesgo de rebote rápido"})
    elif skew < -0.2: diag.append({"Dato": "Skewness", "Estado": "📉 Bajista", "Significado": "Riesgo de caídas bruscas"})
    else: diag.append({"Dato": "Skewness", "Estado": "⚖️ Simétrico", "Significado": "Equilibrio de riesgo"})
    
    if r2 > 0.15: diag.append({"Dato": "R2 (Calidad)", "Estado": "💎 ALTA", "Significado": "Movimiento institucional confirmado"})
    else: diag.append({"Dato": "R2 (Calidad)", "Estado": "💨 RUIDO", "Significado": "Cuidado con trampas de bajo volumen"})
    return pd.DataFrame(diag)

# --- LISTA DE ACTIVOS ACTUALIZADA (24/5 CONTINUOS) ---
assets = {
    "Índices (24/5 Continuos)": {
        "Nasdaq 100 E-Mini": "NQ=F",      # Futuro continuo
        "S&P 500 E-Mini": "ES=F",        # Futuro continuo
        "Dow Jones Mini": "YM=F",       # Futuro continuo
        "DAX 40 (GER)": "FDAX.EX",       # Futuro continuo Europa
        "Nikkei 225": "NKD=F"            # Futuro continuo Asia/USA
    },
    "Currencies (24/5)": {
        "EUR/USD": "EURUSD=X", 
        "GBP/USD": "GBPUSD=X", 
        "USD/JPY": "JPY=X", 
        "AUD/USD": "AUDUSD=X"
    },
    "Commodities (24/5)": {
        "Oro": "GC=F", 
        "Plata": "SI=F", 
        "Petróleo WTI": "CL=F", 
        "Cobre": "HG=F"
    },
    "Crypto (24/7)": {
        "Bitcoin": "BTC-USD", 
        "Ethereum": "ETH-USD", 
        "Solana": "SOL-USD"
    }
}

st.sidebar.title("📑 Master Sniper v11.7")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Sniper Ejecución", "🕵️ Diagnóstico", "🧬 Historial Flujo", "🔗 Absorción Pro", "🏰 Camarilla", "🧮 RISK MGR"
])

    with tab1:
        st.subheader(f"Centro de Operaciones - {nombre} (05:00 - 06:00 AM)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Diff", f"{row['Z_Diff']:.2f}")
        c2.metric("Skewness", f"{row['Skew']:.2f}")
        c3.metric("R2 Calidad", f"{row['R2']:.3f}")
        
        if abs(row['Z_Diff']) > 1.0 and row['R2'] > 0.05:
            prob = min(50.0 + abs(row['Z_Diff'])*12 + row['R2']*45, 98.4)
            color = "#00ff00" if row['Z_Diff'] < -1.0 else "#ff4b4b"
            direc = "LONG (COMPRA)" if row['Z_Diff'] < -1.0 else "SHORT (VENTA)"
            st.markdown(f"""<div class="signal-card" style="border-color: {color};"><h2 style="color: {color};">🔥 SEÑAL ACTIVA: {direc}</h2><div style="display: flex; justify-content: space-between;"><div><p>Precio Entrada: <b>{row['Close']:.4f}</b></p></div><div style="text-align: right;"><p>Probabilidad</p><h1 style="color: {color};">{prob:.1f}%</h1></div></div></div>""", unsafe_allow_html=True)
        else: st.info("📉 Esperando confluencia (Z-Diff > 1.0 & R2 > 0.05)")
        st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])]).update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("Centro de Diagnóstico Dinámico")
        st.table(get_dynamic_diagnosis(row['Z_Diff'], row['Z_Price'], row['Skew'], row['R2']))

    with tab3:
        st.markdown("<div class='gold-header'>🧬 HISTORIAL DE FLUJO INSTITUCIONAL</div>", unsafe_allow_html=True)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Price'], name="Precio (Z)", line=dict(color='#00d4ff')))
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Diff'], name="Flujo (Z)", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_f.update_layout(template="plotly_dark", height=450), use_container_width=True)

    with tab4:
        st.markdown("<div class='gold-header'>🔗 MASTER DE ABSORCIÓN INSTITUCIONAL</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("### 💡 Interpretación")
            if row['Z_Eff'] > 1.5: st.success("**ALTA EFICIENCIA:** El precio fluye con el volumen.")
            elif row['Z_Eff'] < -1.5: st.warning("**ABSORCIÓN:** Volumen alto frenando el precio.")
            else: st.write("Flujo estándar.")
        with col_b:
            st.plotly_chart(px.bar(data.tail(40), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=350), use_container_width=True)

    with tab5:
        st.markdown("<div class='gold-header'>🏰 NIVELES CAMARILLA PROYECTADOS</div>", unsafe_allow_html=True)
        cl1, cl2, cl3, cl4 = st.columns(4)
        cl1.metric("H4 (Breakout)", f"{row['H4']:.4f}")
        cl2.metric("H3 (Reversión)", f"{row['H3']:.4f}")
        cl3.metric("L3 (Reversión)", f"{row['L3']:.4f}")
        cl4.metric("L4 (Breakout)", f"{row['L4']:.4f}")
        fig_cam = go.Figure(data=[go.Candlestick(x=data.index[-50:], open=data['Open'][-50:], high=data['High'][-50:], low=data['Low'][-50:], close=data['Close'][-50:])])
        for n, c in [('H4', 'red'), ('H3', 'orange'), ('L3', 'lightgreen'), ('L4', 'green')]:
            fig_cam.add_hline(y=row[n], line_dash="dash", line_color=c, annotation_text=n)
        st.plotly_chart(fig_cam.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)
else:
    st.error("Error al conectar con la API.")

with tab6:
    st.subheader("🧮 Risk Manager (RoboForex ECN Edition)")
    
    c_1, c_2 = st.columns(2)
    
    with c_1:
        balance = st.number_input("Balance de la Cuenta (USD)", value=1000.0, step=100.0)
        riesgo_pct = st.slider("Riesgo por Operación (%)", 0.1, 5.0, 1.0, 0.1)
        stop_loss_pips = st.number_input("Stop Loss en Pips / Puntos", value=10.0, step=1.0)
        
    with c_2:
        # Menú dinámico según los activos de RoboForex
        activo_rf = st.selectbox("Activo a Operar:", [
            "Forex (Majors/Minors)", 
            "Oro (XAUUSD)", 
            "Petróleo (WTI/Brent)", 
            "Crypto (BTC/ETH)",
            "Indices (US30/DE40)"
        ])
        
        # Lógica de Valor de Contrato de RoboForex ECN
        if activo_rf == "Forex (Majors/Minors)":
            pip_value = 10.0 # 1 lote = $10/pip
        elif activo_rf == "Oro (XAUUSD)":
            pip_value = 1.0  # En RoboForex, 1 lote de Oro suele ser 100 onzas ($1 por cada 0.01 de movimiento)
        elif activo_rf == "Indices (US30/DE40)":
            pip_value = 1.0  # Depende del contrato, pero suele ser $1 por punto por lote
        else:
            pip_value = 1.0

    # CÁLCULO PROFESIONAL
    riesgo_usd = balance * (riesgo_pct / 100)
    
    # Fórmula: Lote = Riesgo USD / (SL Pips * Valor del Pip)
    if stop_loss_pips > 0:
        lotes_final = riesgo_usd / (stop_loss_pips * pip_value)
        # RoboForex ECN permite microlotes (0.01)
        lotes_final = max(0.01, round(lotes_final, 2))
    else:
        lotes_final = 0.0

    st.markdown("---")
    res1, res2, res3 = st.columns(3)
    
    with res1:
        st.metric("Pérdida Máxima", f"${riesgo_usd:.2f}")
    with res2:
        st.metric("Lotaje Sugerido", f"{lotes_final}")
    with res3:
        # Recomendación de Apalancamiento para RoboForex
        st.write(f"**Consejo ECN:**")
        st.caption(f"Para un SL de {stop_loss_pips} pips, usa {lotes_final} lotes para mantener el riesgo bajo control.")

    st.warning("⚠️ Nota: En RoboForex ECN, recuerda sumar la comisión fija por lote al calcular tu breakeven.")
