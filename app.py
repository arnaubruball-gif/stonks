import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v11.7 - Full Restoration", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .gold-header { color: #ffd700; font-weight: bold; border-bottom: 1px solid #ffd700; padding-bottom: 5px; margin-bottom: 15px; }
    .camarilla-box { background-color: #0a0e14; border: 1px solid #444; padding: 10px; border-radius: 5px; text-align: center; }
    .signal-card { background: linear-gradient(135deg, #1e2530 0%, #0d1117 100%); border: 2px solid #30363d; padding: 25px; border-radius: 15px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_final_data(ticker_id, t):
    p_map = {"1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    dxy = yf.download("DX-Y.NYB", period=p_map[t], interval=t, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
    
    W = 14
    df['Ret'] = df['Close'].pct_change()
    df['SMA'] = df['Close'].rolling(W).mean()
    df['Std'] = df['Close'].rolling(W).std()
    df['Z_Price'] = (df['Close'] - df['SMA']) / (df['Std'] + 1e-10)
    
    # JDetector (Flujo Institucional)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(W).sum() - df['RMF'].pct_change().rolling(W).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(W).mean()) / (diff_val.rolling(W).std() + 1e-10)
    
    # Estructura y Absorción
    df['Skew'] = df['Ret'].rolling(30).skew()
    df['Spread'] = (df['High'] - df['Low'])
    df['V_Eff'] = df['Spread'] / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(W).mean()) / (df['V_Eff'].rolling(W).std() + 1e-10)
    
    r2_s = []
    for i in range(len(df)):
        if i < W: r2_s.append(0); continue
        sub = df.iloc[i-W:i].dropna()
        try: r2_s.append(sm.OLS(sub['Ret'], sm.add_constant(sub['RMF'])).fit().rsquared)
        except: r2_s.append(0)
    df['R2'] = r2_s

    # Camarilla
    daily = yf.download(ticker_id, period="5d", interval="1d", progress=False)
    if isinstance(daily.columns, pd.MultiIndex): daily.columns = daily.columns.get_level_values(0)
    H, L, C = daily['High'].iloc[-2], daily['Low'].iloc[-2], daily['Close'].iloc[-2]
    r_val = H - L
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

# --- ACTIVOS ---
assets = {
    "Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X"},
    "Commodities": {"Oro": "GC=F", "Plata": "SI=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F"},
    "Indices": {"Nasdaq 100": "^IXIC", "S&P 500": "^GSPC", "DAX 40": "^GDAXI", "Nikkei 225": "^N225"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}
}

st.sidebar.title("📑 Master Sniper v11.7")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Sniper Ejecución", "🕵️ Diagnóstico", "🧬 Historial Flujo", "🔗 Absorción Pro", "🏰 Camarilla"])

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
