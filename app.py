import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Alpha Quant v8.5 - Full Execution", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-bottom: 20px; }
    .insider-box { background: linear-gradient(90deg, #1e2130 0%, #0d1117 100%); padding: 15px; border-left: 5px solid #ffd700; border-radius: 8px; margin-bottom: 20px; }
    .veredicto-banner { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 10px; color: white; }
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
def get_alpha_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    dxy = yf.download("DX-Y.NYB", period=p_map[t], interval=t, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
    
    # Básicos y Volatilidad
    df['Ret'] = df['Close'].pct_change()
    df['Volatilidad'] = df['Ret'].rolling(20).std()
    df['Z_Vol'] = (df['Volatilidad'] - df['Volatilidad'].rolling(20).mean()) / (df['Volatilidad'].rolling(20).std() + 1e-10)
    
    # JDetector (Z-Price & Z-Diff)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std20'] = df['Close'].rolling(20).std()
    df['Z_Price'] = (df['Close'] - df['SMA20']) / (df['Std20'] + 1e-10)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(20).sum() - df['RMF'].pct_change().rolling(20).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(20).mean()) / (diff_val.rolling(20).std() + 1e-10)
    
    # R2 Dinámico
    r2_series = []
    for i in range(len(df)):
        if i < 20: r2_series.append(0); continue
        subset = df.iloc[i-20:i].dropna()
        try:
            X = sm.add_constant(subset['RMF'])
            r2_series.append(sm.OLS(subset['Ret'], X).fit().rsquared)
        except: r2_series.append(0)
    df['R2_Dynamic'] = r2_series
    
    # Institutional Edge
    df['V_Eff'] = (df['Close'].diff().abs()) / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(20).mean()) / (df['V_Eff'].rolling(20).std() + 1e-10)
    df['DXY_Corr'] = df['Ret'].rolling(20).corr(dxy['Close'].pct_change())
    
    return df

# --- INTERFAZ ---
assets_dict = {
    "Currencies": {"EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"Nasdaq 100": "^IXIC", "S&P 500": "^GSPC", "DAX 40": "^GDAXI"},
    "Commodities": {"Oro": "GC=F", "Petróleo": "CL=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Alpha Terminal v8.5")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=1)

data = get_alpha_data(ticker, temp)

if data is not None:
    # Variables actuales
    z_p, z_d, z_v, r2 = data['Z_Price'].iloc[-1], data['Z_Diff'].iloc[-1], data['Z_Vol'].iloc[-1], data['R2_Dynamic'].iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    skew = stats.skew(data['Ret'].tail(30).dropna())
    eff = data['Z_Eff'].iloc[-1]
    
    # Lógica de Ejecución (Timing)
    sma, std = data['SMA20'].iloc[-1], data['Std20'].iloc[-1]
    entry_l, entry_s = sma - (2.1 * std), sma + (2.1 * std)
    sl_l, sl_s = sma - (2.8 * std), sma + (2.8 * std)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Ejecución & Probabilidad", "🧬 Flujo JDetector", "🕵️ Estructura Pro", "🔗 Institutional Edge"])

    with tab1:
        # 1. Banners de Probabilidad
        prob_buy = 50.0 + ((-z_p * 12) if z_p < -1.5 else 0) + ((-z_d * 18) if z_d < -1.2 else 0)
        prob_sell = 50.0 + ((z_p * 12) if z_p > 1.5 else 0) + ((z_d * 18) if z_d > 1.2 else 0)
        p_buy, p_sell = (prob_buy/(prob_buy+prob_sell))*100, (prob_sell/(prob_buy+prob_sell))*100
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="prob-box" style="color:#00cc96; border-color:#00cc96;"><h4>Probabilidad COMPRA</h4><h2>{p_buy:.1f}%</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="prob-box" style="color:#ff4b4b; border-color:#ff4b4b;"><h4>Probabilidad VENTA</h4><h2>{p_sell:.1f}%</h2></div>', unsafe_allow_html=True)
        
        # 2. El Playbook de Ejecución
        tipo = "TREND FOLLOWING" if hurst > 0.56 else "MEAN REVERSION" if hurst < 0.44 else "NEUTRAL / RANGO"
        st.markdown(f'<div class="strategy-card"><h3>📋 Playbook: {tipo}</h3><p>Riesgo Volatilidad: {"ALTO 🔥" if z_v > 1.2 else "ESTABLE ⚓"}</p></div>', unsafe_allow_html=True)
        
        # 3. Niveles de Entrada
        e1, e2, e3 = st.columns(3)
        if z_d < -1.4:
            e1.metric("ESTRATEGIA", "LONG", "Trigger Activo")
            e2.metric("Precio de Entrada Sug.", f"{entry_l:.4f}")
            e3.metric("Target (Media)", f"{sma:.4f}")
            st.error(f"Stop Loss Crítico: {sl_l:.4f}")
        elif z_d > 1.4:
            e1.metric("ESTRATEGIA", "SHORT", "Trigger Activo")
            e2.metric("Precio de Entrada Sug.", f"{entry_s:.4f}")
            e3.metric("Target (Media)", f"{sma:.4f}")
            st.error(f"Stop Loss Crítico: {sl_s:.4f}")
        else:
            st.info("⌛ Esperando Confluencia: El flujo institucional (Z-Diff) aún no confirma la dirección.")

        # 4. Gráfico con Niveles
        fig_p = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig_p.add_hline(y=sma, line_dash="dash", line_color="cyan", opacity=0.5, annotation_text="Media (Target)")
        fig_p.add_hline(y=entry_l, line_color="#00cc96", opacity=0.3, annotation_text="Buy Zone")
        fig_p.add_hline(y=entry_s, line_color="#ff4b4b", opacity=0.3, annotation_text="Sell Zone")
        st.plotly_chart(fig_p.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("🧬 Auditoría JDetector (Z-Diff vs Z-Price)")
        st.markdown(f'<div class="veredicto-banner" style="background-color:#1e2130;">ESTADO: {"Divergencia Detectada" if abs(z_d-z_p)>1.5 else "Flujo Alineado"} | R2: {r2:.3f}</div>', unsafe_allow_html=True)
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Price'].tail(80), name="Z-Price (Precio)", line=dict(color='#00d4ff')))
        fig_dual.add_trace(go.Scatter(x=data.index[-80:], y=data['Z_Diff'].tail(80), name="Z-Diff (Dinero)", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_dual.update_layout(template="plotly_dark", height=350), use_container_width=True)
        
        audit_df = data.tail(8).copy()
        audit_df['Veredicto'] = audit_df.apply(lambda x: "🚨 VENTA" if x['Z_Diff'] > 1.4 else ("🟢 COMPRA" if x['Z_Diff'] < -1.4 else "⚪ Neutral"), axis=1)
        st.table(audit_df[['Close', 'Z_Price', 'Z_Diff', 'Veredicto']])

    with tab3:
        st.subheader("🕵️ Inferencia de Estructura")
        st.markdown(f'<div class="veredicto-banner" style="background-color:#3d4463;">SESGO: {"ALCISTA" if skew > 0.2 else "BAJISTA" if skew < -0.2 else "NEUTRAL"}</div>', unsafe_allow_html=True)
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Hurst (Memoria)", f"{hurst:.2f}")
        c_b.metric("Skewness (Inercia)", f"{skew:.2f}")
        c_c.metric("Z-Volatilidad", f"{z_v:.2f}")
        st.plotly_chart(px.area(data.tail(100), y='Ret', title="Aceleración de Retornos").update_layout(template="plotly_dark", height=300), use_container_width=True)

    with tab4:
        st.subheader("🔗 Institutional Edge (DXY & Absorción)")
        st.markdown(f'<div class="insider-box"><b>DXY Alignment:</b> {data["DXY_Corr"].iloc[-1]:.2f} <br><small>Correlación activa con el Dólar.</small></div>', unsafe_allow_html=True)
        
        c_eff, c_corr = st.columns(2)
        with c_eff:
            st.write("**Z-Efficiency (Absorción)**")
            st.plotly_chart(px.bar(data.tail(60), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=300), use_container_width=True)
        with c_corr:
            st.write("**Dinámica DXY**")
            st.plotly_chart(px.line(data.tail(60), y='DXY_Corr').update_layout(template="plotly_dark", height=300), use_container_width=True)
else:
    st.error("No se pudieron obtener datos del mercado.")
