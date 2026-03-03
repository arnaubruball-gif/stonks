import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v11.0 - Master Sniper", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .prob-box { padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #30363d; margin-bottom: 20px; }
    .strategy-card { background-color: #161b22; padding: 15px; border-left: 5px solid #00d4ff; border-radius: 8px; margin-bottom: 20px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .gold-header { color: #ffd700; font-weight: bold; border-bottom: 1px solid #ffd700; padding-bottom: 5px; margin-bottom: 15px; }
    .camarilla-box { background-color: #0a0e14; border: 1px solid #444; padding: 10px; border-radius: 5px; text-align: center; }
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
    # Flujo
    if z_d < -1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🟢 COMPRA", "Significado": "Entrada de dinero institucional (Absorción)"})
    elif z_d > 1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🔴 VENTA", "Significado": "Salida de dinero / Distribución oculta"})
    else: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "⚪ Neutral", "Significado": "Sin presión clara"})
    # Precio
    if abs(z_p) > 2: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚠️ EXTREMO", "Significado": "Reversión a la media inminente"})
    else: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚓ Estable", "Significado": "Zona de Fair Value"})
    # Skewness (Restaurada)
    if skew > 0.2: diag.append({"Dato": "Skewness", "Estado": "🚀 Alcista", "Significado": "Asimetría: Rebotes más rápidos que caídas"})
    elif skew < -0.2: diag.append({"Dato": "Skewness", "Estado": "📉 Bajista", "Significado": "Asimetría: Riesgo de caída brusca"})
    else: diag.append({"Dato": "Skewness", "Estado": "⚖️ Simétrico", "Significado": "Riesgo equilibrado"})
    # Calidad
    if r2 > 0.15: diag.append({"Dato": "R2 (Calidad)", "Estado": "💎 ALTA", "Significado": "Movimiento institucional confirmado"})
    else: diag.append({"Dato": "R2 (Calidad)", "Estado": "💨 RUIDO", "Significado": "Cuidado: Manipulación o bajo volumen"})
    return pd.DataFrame(diag)

# --- INTERFAZ ---
assets = {
    "Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"},
    "Commodities": {"Oro (Gold)": "GC=F", "Plata (Silver)": "SI=F", "Petróleo (WTI)": "CL=F"},
    "Indices": {"Nasdaq 100": "^IXIC", "S&P 500": "^GSPC", "DAX 40": "^GDAXI"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Master Sniper v11.0")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temp", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    hurst = calculate_hurst(data['Close'].values)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Sniper Ejecución", "🕵️ Centro Diagnóstico", "🧬 Historial Flujo", "🔗 Absorción", "🏰 Camarilla Levels"])

    with tab1:
        # Probabilidades Originales
        prob_buy = 50.0 + ((-row['Z_Price'] * 15) if row['Z_Price'] < -1.5 else 0) + ((-row['Z_Diff'] * 20) if row['Z_Diff'] < -1.0 else 0)
        prob_sell = 50.0 + ((row['Z_Price'] * 15) if row['Z_Price'] > 1.5 else 0) + ((row['Z_Diff'] * 20) if row['Z_Diff'] > 1.0 else 0)
        p_buy, p_sell = (prob_buy/(prob_buy+prob_sell))*100, (prob_sell/(prob_buy+prob_sell))*100
        
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="prob-box" style="color:#00cc96; border-color:#00cc96;"><h4>Probabilidad COMPRA</h4><h2>{p_buy:.1f}%</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="prob-box" style="color:#ff4b4b; border-color:#ff4b4b;"><h4>Probabilidad VENTA</h4><h2>{p_sell:.1f}%</h2></div>', unsafe_allow_html=True)
        
        # PLAYBOOK RESTAURADO
        tipo_strat = "TREND FOLLOWING" if hurst > 0.55 else "MEAN REVERSION"
        st.markdown(f'<div class="strategy-card"><h3>📋 Playbook: {tipo_strat}</h3>'
                    f'<p><b>Hurst Exponent:</b> {hurst:.2f} | <b>Z-Diff Actual:</b> {row["Z_Diff"]:.2f}<br>'
                    f'<i>Estrategia recomendada: {"Comprar rupturas" if hurst > 0.55 else "Vender excesos / Comprar suelos"}</i></p></div>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("Interpretación Dinámica")
        st.table(get_dynamic_diagnosis(row['Z_Diff'], row['Z_Price'], row['Skew'], row['R2']))

    with tab3:
        st.markdown("<div class='gold-header'>🧬 HISTORIAL DE FLUJO INSTITUCIONAL</div>")
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Price'], name="Z-Price", line=dict(color='#00d4ff')))
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Diff'], name="Z-Diff", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_f.update_layout(template="plotly_dark", height=450), use_container_width=True)

    with tab4:
        st.markdown("<div class='gold-header'>🔗 ABSORCIÓN & DXY</div>")
        st.plotly_chart(px.bar(data.tail(50), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=400), use_container_width=True)

    with tab5:
        st.markdown("<div class='gold-header'>🏰 NIVELES CAMARILLA ESTRUCTURALES</div>")
        cl1, cl2, cl3, cl4 = st.columns(4)
        cl1.markdown(f"<div class='camarilla-box'><b style='color:red;'>H4</b><br>{row['H4']:.4f}</div>", unsafe_allow_html=True)
        cl2.markdown(f"<div class='camarilla-box'><b style='color:orange;'>H3</b><br>{row['H3']:.4f}</div>", unsafe_allow_html=True)
        cl3.markdown(f"<div class='camarilla-box'><b style='color:lightgreen;'>L3</b><br>{row['L3']:.4f}</div>", unsafe_allow_html=True)
        cl4.markdown(f"<div class='camarilla-box'><b style='color:green;'>L4</b><br>{row['L4']:.4f}</div>", unsafe_allow_html=True)
        
        fig_cam = go.Figure(data=[go.Candlestick(x=data.index[-60:], open=data['Open'][-60:], high=data['High'][-60:], low=data['Low'][-60:], close=data['Close'][-60:])])
        for n, c in [('H4', 'red'), ('H3', 'orange'), ('L3', 'lightgreen'), ('L4', 'green')]:
            fig_cam.add_hline(y=row[n], line_dash="dash", line_color=c, annotation_text=n)
        st.plotly_chart(fig_cam.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

else:
    st.error("Error al cargar.")
