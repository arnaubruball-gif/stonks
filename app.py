import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Master Sniper v10.7 - Full Suite", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .gold-header { color: #ffd700; font-weight: bold; border-bottom: 1px solid #ffd700; padding-bottom: 5px; margin-bottom: 15px; }
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
    
    # Ventana de Sensibilidad Alta (14 periodos)
    W = 14
    df['Ret'] = df['Close'].pct_change()
    df['SMA'] = df['Close'].rolling(W).mean()
    df['Std'] = df['Close'].rolling(W).std()
    df['Z_Price'] = (df['Close'] - df['SMA']) / (df['Std'] + 1e-10)
    
    # JDetector Sensible
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(W).sum() - df['RMF'].pct_change().rolling(W).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(W).mean()) / (diff_val.rolling(W).std() + 1e-10)
    
    # Skew y R2
    df['Skew'] = df['Ret'].rolling(30).skew()
    r2_s = []
    for i in range(len(df)):
        if i < W: r2_s.append(0); continue
        sub = df.iloc[i-W:i].dropna()
        try: r2_s.append(sm.OLS(sub['Ret'], sm.add_constant(sub['RMF'])).fit().rsquared)
        except: r2_s.append(0)
    df['R2'] = r2_s

    # ABSORCIÓN (DATO ORO)
    df['V_Eff'] = (df['Close'].diff().abs()) / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(W).mean()) / (df['V_Eff'].rolling(W).std() + 1e-10)
    df['DXY_Corr'] = df['Ret'].rolling(W).corr(dxy['Close'].pct_change())
    
    return df

# --- LÓGICA DE DIAGNÓSTICO DINÁMICO (COPIADA DE TU V10) ---
def get_dynamic_diagnosis(z_d, z_p, skew, r2):
    diag = []
    if z_d < -1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🟢 COMPRA", "Significado": "Entrada de dinero institucional (Absorción)"})
    elif z_d > 1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🔴 VENTA", "Significado": "Salida de dinero / Distribución oculta"})
    else: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "⚪ Neutral", "Significado": "Sin presión institucional clara"})
    
    if abs(z_p) > 2: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚠️ EXTREMO", "Significado": "Precio muy alejado de la media. Reversión probable."})
    else: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚓ Estable", "Significado": "Precio en zona de equilibrio (Fair Value)"})
    
    if skew > 0.2: diag.append({"Dato": "Skewness", "Estado": "🚀 Alcista", "Significado": "Asimetría a favor de rebotes rápidos"})
    elif skew < -0.2: diag.append({"Dato": "Skewness", "Estado": "📉 Bajista", "Significado": "Riesgo de caídas bruscas (Fat Tail)"})
    else: diag.append({"Dato": "Skewness", "Estado": "⚖️ Simétrico", "Significado": "Riesgo equilibrado en ambas direcciones"})
    
    if r2 > 0.15: diag.append({"Dato": "R2 (Calidad)", "Estado": "💎 ALTA", "Significado": "Movimiento respaldado por volumen real"})
    else: diag.append({"Dato": "R2 (Calidad)", "Estado": "💨 RUIDO", "Significado": "Cuidado: El precio se mueve sin volumen real"})
    return pd.DataFrame(diag)

# --- INTERFAZ ---
assets = {"Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X"}, "Indices": {"Nasdaq 100": "^IXIC"}, "Crypto": {"Bitcoin": "BTC-USD"}}
st.sidebar.title("📑 Master Sniper v10.7")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temp", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Sniper Ejecución", "🕵️ Centro de Diagnóstico", "🧬 Historial de Flujo", "🔗 Absorción & DXY"])

    with tab1:
        st.subheader(f"Plan Táctico - {nombre} (05:00 - 06:00 AM)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Diff", f"{row['Z_Diff']:.2f}")
        c2.metric("Skewness", f"{row['Skew']:.2f}")
        c3.metric("R2 Calidad", f"{row['R2']:.3f}")
        
        if abs(row['Z_Diff']) > 1.0 and row['R2'] > 0.05:
            color = "#00ff00" if row['Z_Diff'] < -1.0 else "#ff0000"
            st.markdown(f"<div class='diag-box' style='border-color:{color};'><h3>DISPARAR: {'LONG' if row['Z_Diff'] < -1.0 else 'SHORT'}</h3>"
                        f"Precio: <b>{row['Close']:.4f}</b><br>Horizonte: 1-3 días</div>", unsafe_allow_html=True)
        else:
            st.info("Esperando confluencia de alta calidad (Z-Diff > 1.0 & R2 > 0.05)")

        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("Interpretación de Datos en Tiempo Real")
        df_diag = get_dynamic_diagnosis(row['Z_Diff'], row['Z_Price'], row['Skew'], row['R2'])
        st.table(df_diag)
        st.markdown("---")
        st.write("💡 *Usa esta tabla para validar si el disparo de la Tab 1 tiene sentido institucional.*")

    with tab3:
        st.markdown("<div class='gold-header'>🧬 HISTORIAL DE FLUJO INSTITUCIONAL (JDETECTOR)</div>", unsafe_allow_html=True)
        st.write("Si el **Z-Diff** (Amarillo) diverge del **Z-Price** (Azul), el mercado está preparando un giro.")
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(x=data.index, y=data['Z_Price'], name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_flow.add_trace(go.Scatter(x=data.index, y=data['Z_Diff'], name="Z-Diff (Dinero)", line=dict(color='#ffd700', dash='dot')))
        fig_flow.add_hline(y=1.5, line_dash="dash", line_color="red", opacity=0.5)
        fig_flow.add_hline(y=-1.5, line_dash="dash", line_color="green", opacity=0.5)
        st.plotly_chart(fig_flow.update_layout(template="plotly_dark", height=450), use_container_width=True)

    with tab4:
        st.markdown("<div class='gold-header'>🔗 VOLUMEN DE ABSORCIÓN & CORRELACIÓN DXY</div>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Z-Efficiency (Detector de Manos Fuertes)**")
            st.plotly_chart(px.bar(data.tail(50), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=350), use_container_width=True)
        with col_right:
            st.write("**Correlación Dinámica vs DXY**")
            st.plotly_chart(px.line(data.tail(50), y='DXY_Corr').update_layout(template="plotly_dark", height=350), use_container_width=True)
else:
    st.error("Error al conectar con los mercados.")
