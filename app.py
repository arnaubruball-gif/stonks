import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v10.0 - Diagnostic Engine", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .status-active { color: #00ff00; font-weight: bold; }
    .status-warn { color: #ffcc00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_final_data(ticker_id, t):
    p_map = {"1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
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
    
    return df

# --- LÓGICA DE DIAGNÓSTICO DINÁMICO ---
def get_dynamic_diagnosis(z_d, z_p, skew, r2):
    diag = []
    
    # Situación de Flujo
    if z_d < -1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🟢 COMPRA", "Significado": "Entrada de dinero institucional (Absorción)"})
    elif z_d > 1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🔴 VENTA", "Significado": "Salida de dinero / Distribución oculta"})
    else: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "⚪ Neutral", "Significado": "Sin presión institucional clara"})

    # Situación de Precio
    if abs(z_p) > 2: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚠️ EXTREMO", "Significado": "Precio muy alejado de la media. Reversión probable."})
    else: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚓ Estable", "Significado": "Precio en zona de equilibrio (Fair Value)"})

    # Situación de Sesgo
    if skew > 0.2: diag.append({"Dato": "Skewness", "Estado": "🚀 Alcista", "Significado": "Asimetría a favor de rebotes rápidos"})
    elif skew < -0.2: diag.append({"Dato": "Skewness", "Estado": "📉 Bajista", "Significado": "Riesgo de caídas bruscas (Fat Tail)"})
    else: diag.append({"Dato": "Skewness", "Estado": "⚖️ Simétrico", "Significado": "Riesgo equilibrado en ambas direcciones"})

    # Situación de Calidad
    if r2 > 0.15: diag.append({"Dato": "R2 (Calidad)", "Estado": "💎 ALTA", "Significado": "Movimiento respaldado por volumen real"})
    else: diag.append({"Dato": "R2 (Calidad)", "Estado": "💨 RUIDO", "Significado": "Cuidado: El precio se mueve sin volumen real"})
    
    return pd.DataFrame(diag)

# --- INTERFAZ ---
assets = {"Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X"}, "Indices": {"Nasdaq 100": "^IXIC"}, "Crypto": {"Bitcoin": "BTC-USD"}}
st.sidebar.title("📑 Master Sniper v10.0")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
data = get_final_data(assets[cat][nombre], st.sidebar.selectbox("Temp", ["1h", "4h", "1d"]))

if data is not None:
    row = data.iloc[-1]
    
    tab1, tab2 = st.tabs(["🎯 Sniper Ejecución", "🕵️ Centro de Diagnóstico"])

    with tab1:
        st.subheader("Plan Táctico (05:00 - 06:00 AM)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Diff", f"{row['Z_Diff']:.2f}")
        c2.metric("Skewness", f"{row['Skew']:.2f}")
        c3.metric("R2 Calidad", f"{row['R2']:.3f}")
        
        # Lógica de Gatillo
        if abs(row['Z_Diff']) > 1.0 and row['R2'] > 0.05:
            color = "#00ff00" if row['Z_Diff'] < -1.0 else "#ff0000"
            st.markdown(f"<div class='diag-box' style='border-color:{color};'><h3>DISPARAR: {'LONG' if row['Z_Diff'] < -1.0 else 'SHORT'}</h3>"
                        f"Entrada sugerida cerca de: <b>{row['Close']:.4f}</b><br>"
                        f"Horizonte: Intradía / 2-3 días</div>", unsafe_allow_html=True)
        else:
            st.info("Esperando confluencia de alta calidad (Z-Diff > 1.0 & R2 > 0.05)")

        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("Interpretación de Datos en Tiempo Real")
        df_diag = get_dynamic_diagnosis(row['Z_Diff'], row['Z_Price'], row['Skew'], row['R2'])
        
        # Mostramos la tabla dinámica
        st.table(df_diag)
        
        st.markdown("""
        ---
        ### 📖 Guía Rápida de Acción
        * **Z-Diff + R2 Alto:** Es la señal más fuerte. Las instituciones están moviendo el mercado.
        * **Skew a favor del trade:** Aumenta tu Take Profit, el mercado tiene inercia.
        * **R2 Bajo + Z-Diff Alto:** ¡Trampa! Es un movimiento de manipulación (falsos rompimientos).
        """)

else:
    st.error("Error al conectar con los mercados.")
