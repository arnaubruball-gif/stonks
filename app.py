import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v11.3 - Probability Alpha", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .gold-header { color: #ffd700; font-weight: bold; border-bottom: 1px solid #ffd700; padding-bottom: 5px; margin-bottom: 15px; }
    .camarilla-box { background-color: #0a0e14; border: 1px solid #444; padding: 10px; border-radius: 5px; text-align: center; }
    .prob-display { background: linear-gradient(90deg, #161b22 0%, #0d1117 100%); border: 1px solid #30363d; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; }
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
    
    # JDetector
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(W).sum() - df['RMF'].pct_change().rolling(W).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(W).mean()) / (diff_val.rolling(W).std() + 1e-10)
    
    # Estructura
    df['Skew'] = df['Ret'].rolling(30).skew()
    r2_s = []
    for i in range(len(df)):
        if i < W: r2_s.append(0); continue
        sub = df.iloc[i-W:i].dropna()
        try: r2_s.append(sm.OLS(sub['Ret'], sm.add_constant(sub['RMF'])).fit().rsquared)
        except: r2_s.append(0)
    df['R2'] = r2_s

    # Absorción y DXY
    df['V_Eff'] = (df['Close'].diff().abs()) / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(W).mean()) / (df['V_Eff'].rolling(W).std() + 1e-10)
    df['DXY_Corr'] = df['Ret'].rolling(W).corr(dxy['Close'].pct_change())
    
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
    if z_d < -1.0: diag.append({"Dato": "Z-Diff", "Estado": "🟢 COMPRA", "Significado": "Flujo institucional de entrada"})
    elif z_d > 1.0: diag.append({"Dato": "Z-Diff", "Estado": "🔴 VENTA", "Significado": "Flujo institucional de salida"})
    else: diag.append({"Dato": "Z-Diff", "Estado": "⚪ Neutral", "Significado": "Sin flujo claro"})
    
    if abs(z_p) > 2: diag.append({"Dato": "Z-Price", "Estado": "⚠️ EXTREMO", "Significado": "Precio sobreextendido"})
    else: diag.append({"Dato": "Z-Price", "Estado": "⚓ Estable", "Significado": "Zona de equilibrio"})
    
    if skew > 0.2: diag.append({"Dato": "Skewness", "Estado": "🚀 Alcista", "Significado": "Sesgo de rebote rápido"})
    elif skew < -0.2: diag.append({"Dato": "Skewness", "Estado": "📉 Bajista", "Significado": "Riesgo de caída pesada"})
    else: diag.append({"Dato": "Skewness", "Estado": "⚖️ Simétrico", "Significado": "Equilibrio"})
    
    if r2 > 0.15: diag.append({"Dato": "R2 Calidad", "Estado": "💎 ALTA", "Significado": "Señal fiable"})
    else: diag.append({"Dato": "R2 Calidad", "Estado": "💨 RUIDO", "Significado": "Cuidado con trampas"})
    return pd.DataFrame(diag)

# --- ACTIVOS ---
assets = {
    "Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X"},
    "Commodities": {"Oro": "GC=F", "Plata": "SI=F", "Petróleo WTI": "CL=F", "Gas Natural": "NG=F"},
    "Indices": {"Nasdaq 100": "^IXIC", "S&P 500": "^GSPC", "Dow Jones": "^DJI", "DAX 40": "^GDAXI"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}
}

st.sidebar.title("📑 Sniper v11.3")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temp", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Sniper Ejecución", "🕵️ Diagnóstico", "🧬 Historial Flujo", "🔗 Absorción", "🏰 Camarilla"])

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
        st.subheader("Análisis de Probabilidad y Diagnóstico")
        
        # LÓGICA DE PROBABILIDAD (QUIRÚRGICA)
        base_prob = 50.0
        mod_zdiff = abs(row['Z_Diff']) * 12 if abs(row['Z_Diff']) > 1.0 else 0
        mod_r2 = row['R2'] * 40
        mod_zprice = abs(row['Z_Price']) * 5 if abs(row['Z_Price']) > 1.5 else 0
        
        total_prob = min(base_prob + mod_zdiff + mod_r2 + mod_zprice, 98.4) # Capado al 98% (nada es 100%)
        
        color_p = "#00ff00" if total_prob > 75 else "#ffd700" if total_prob > 60 else "#ff4b4b"
        
        st.markdown(f"""
            <div class="prob-display">
                <h4 style="color: #888; margin-bottom: 5px;">Probabilidad de Éxito de la Señal Actual</h4>
                <h1 style="color: {color_p}; font-size: 3.5rem; margin: 0;">{total_prob:.1f}%</h1>
                <p style="color: #666; font-size: 0.9rem;">Basado en confluencia de Flujo Institucional, R2 de Calidad y Ciclo de Precio</p>
            </div>
        """, unsafe_allow_html=True)

        st.table(get_dynamic_diagnosis(row['Z_Diff'], row['Z_Price'], row['Skew'], row['R2']))

    with tab3:
        st.markdown("<div class='gold-header'>🧬 HISTORIAL DE FLUJO INSTITUCIONAL</div>", unsafe_allow_html=True)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Price'], name="Precio", line=dict(color='#00d4ff')))
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Diff'], name="Flujo", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_f.update_layout(template="plotly_dark", height=450), use_container_width=True)

    with tab4:
        st.markdown("<div class='gold-header'>🔗 ABSORCIÓN & DXY</div>", unsafe_allow_html=True)
        st.plotly_chart(px.bar(data.tail(50), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=400), use_container_width=True)

    with tab5:
        st.markdown("<div class='gold-header'>🏰 NIVELES CAMARILLA PROYECTADOS</div>", unsafe_allow_html=True)
        cl1, cl2, cl3, cl4 = st.columns(4)
        cl1.markdown(f"<div class='camarilla-box'><b style='color:red;'>H4</b><br>{row['H4']:.4f}</div>", unsafe_allow_html=True)
        cl2.markdown(f"<div class='camarilla-box'><b style='color:orange;'>H3</b><br>{row['H3']:.4f}</div>", unsafe_allow_html=True)
        cl3.markdown(f"<div class='camarilla-box'><b style='color:lightgreen;'>L3</b><br>{row['L3']:.4f}</div>", unsafe_allow_html=True)
        cl4.markdown(f"<div class='camarilla-box'><b style='color:green;'>L4</b><br>{row['L4']:.4f}</div>", unsafe_allow_html=True)
        
        fig_cam = go.Figure(data=[go.Candlestick(x=data.index[-50:], open=data['Open'][-50:], high=data['High'][-50:], low=data['Low'][-50:], close=data['Close'][-50:])])
        for n, c in [('H4', 'red'), ('H3', 'orange'), ('L3', 'lightgreen'), ('L4', 'green')]:
            fig_cam.add_hline(y=row[n], line_dash="dash", line_color=c, annotation_text=n)
        st.plotly_chart(fig_cam.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)
else:
    st.error("Error al cargar los datos.")
