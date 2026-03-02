import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats # Necesaria para la curva de densidad

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant Terminal Pro", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #11151c; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #11151c; border-radius: 5px; color: #888; }
    .stTabs [data-baseweb="tab"]:hover { color: white; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom: 2px solid #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- ACTIVOS ---
assets_dict = {
    "Currencies": {"Dólar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X"},
    "Indices": {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "DAX 40": "^GDAXI", "Nikkei 225": "^N225"},
    "Commodities": {"Oro": "GC=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F"},
    "Bonds": {"Bono 10Y USA": "^TNX", "Bono 2Y USA": "^ZT=F"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

# --- FUNCIONES ---
def calculate_hurst(series):
    if len(series) < 50: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=60)
def get_full_data(ticker_id, t):
    p_map = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "2y"}
    df = yf.download(ticker_id, period=p_map[t], interval=t)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Z-Score'] = (df['Close'] - df['SMA20']) / df['Std']
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Anual'] = df['Returns'].rolling(20).std() * np.sqrt(252) * 100
    df['RSI'] = 100 - (100 / (1 + (df['Returns'].where(df['Returns']>0,0).rolling(14).mean() / -df['Returns'].where(df['Returns']<0,0).rolling(14).mean())))
    if 'Volume' in df.columns:
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    return df

# --- SIDEBAR ---
st.sidebar.title("💎 Quant Selector")
cat = st.sidebar.selectbox("Categoría", list(assets_dict.keys()))
nombre_activo = st.sidebar.selectbox("Activo", list(assets_dict[cat].keys()))
ticker = assets_dict[cat][nombre_activo]
temp = st.sidebar.selectbox("Temporalidad", ["15m", "1h", "4h", "1d"], index=3)

data = get_full_data(ticker, temp)

if data is not None:
    # --- HEADER ---
    last_price = data['Close'].iloc[-1]
    st.title(f"📈 {nombre_activo} | Terminal Intelligence")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"{last_price:.4f}", f"{data['Returns'].iloc[-1]*100:.2f}%")
    m2.metric("Z-Score (Desviación)", f"{data['Z-Score'].iloc[-1]:.2f} σ")
    m3.metric("RSI (Sentimiento)", f"{data['RSI'].iloc[-1]:.1f}")
    m4.metric("Vol. Realizada", f"{data['Vol_Anual'].iloc[-1]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráfico Técnico", "🧬 Curva de Probabilidad", "🎯 Niveles Camarilla", "🚀 Alpha Quant Metrics"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio"))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='#ff9900', width=1.5), name="Media 20p"))
        fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Análisis de Distribución de Retornos (Matemática Pura)")
        c_l, c_r = st.columns([2, 1])
        
        with c_l:
            # LIMPIEZA PARA LA CURVA KDE
            returns_clean = data['Returns'].dropna()
            
            # Crear histograma con Plotly
            fig_dist = px.histogram(returns_clean, nbins=100, histnorm='probability density', 
                                   opacity=0.3, color_discrete_sequence=['#00ffcc'], title="Curva de Densidad de Probabilidad (KDE)")
            
            # Añadir la línea KDE (Curva suave)
            x_range = np.linspace(returns_clean.min(), returns_clean.max(), 200)
            kde = stats.gaussian_kde(returns_clean)
            fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range), line=dict(color='#00ffcc', width=3), name="Curva de Probabilidad"))
            
            # Línea de retorno actual
            fig_dist.add_vline(x=returns_clean.iloc[-1], line_dash="dash", line_color="red", annotation_text="Retorno Actual")
            
            fig_dist.update_layout(template="plotly_dark", height=450, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.info("💡 La campana muestra dónde es 'normal' que se mueva el precio. Si el 'Retorno Actual' está muy en los extremos, la probabilidad de reversión aumenta drásticamente.")

        with c_r:
            score = 0
            if data['Z-Score'].iloc[-1] < -1.5: score += 50
            if data['RSI'].iloc[-1] < 35: score += 50
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score, title={'text': "Score Reversión %"},
                gauge={'axis':{'range':[0,100], 'tickcolor':"white"}, 'bar':{'color':"#00ffcc"}, 
                       'steps':[{'range':[0,40],'color':"#330000"},{'range':[70,100],'color':"#003311"}]}))
            fig_gauge.update_layout(height=400, template="plotly_dark", margin=dict(l=30,r=30,t=50,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab3:
        # Niveles Camarilla (Zonas institucionales)
        H, L, C = data['High'].iloc[-2], data['Low'].iloc[-2], data['Close'].iloc[-2]
        d = H - L
        cam = {'H4': C+d*(1.1/2), 'H3': C+d*(1.1/4), 'L3': C-d*(1.1/4), 'L4': C-d*(1.1/2)}
        
        pc = st.columns(4)
        pc[0].metric("H4 (Breakout Venta)", f"{cam['H4']:.4f}")
        pc[1].metric("H3 (Venta Reversión)", f"{cam['H3']:.4f}")
        pc[2].metric("L3 (Compra Reversión)", f"{cam['L3']:.4f}")
        pc[3].metric("L4 (Breakout Compra)", f"{cam['L4']:.4f}")
        
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Scatter(x=data.index[-50:], y=data['Close'][-50:], name="Precio", line=dict(color="#00ffcc")))
        for l, v in cam.items():
            color = "#ff4444" if "4" in l else "#ffcc00"
            fig_cam.add_hline(y=v, line_dash="dash", line_color=color, annotation_text=l)
        fig_cam.update_layout(height=500, template="plotly_dark", title="Niveles Camarilla Activos (Basados en vela anterior)")
        st.plotly_chart(fig_cam, use_container_width=True)

    with tab4:
        st.subheader("Métricas de Estructura Matemática")
        h_val = calculate_hurst(data['Close'].values)
        r_val = data['RVOL'].iloc[-1] if 'RVOL' in data.columns else 0
        
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Exponente de Hurst", f"{h_val:.2f}")
        ac2.metric("Volumen Relativo (RVOL)", f"{r_val:.2f}x")
        ac3.metric("Momentum (10p)", f"{((data['Close'].iloc[-1]/data['Close'].shift(10).iloc[-1])-1)*100:.2f}%")
        
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(px.line(data, x=data.index, y='Z-Score', title="Z-Score Dinámico").add_hline(y=2, line_color="red").add_hline(y=-2, line_color="green"), use_container_width=True)
        with g2:
            st.plotly_chart(px.bar(data, x=data.index, y='RVOL', title="Anomalías de RVOL").add_hline(y=1.5, line_color="red", line_dash="dash"), use_container_width=True)
            with tab7:
    st.subheader("⚖️ BEER Model: Behavioral Equilibrium Exchange Rate")
    st.write("Calculando el valor justo por diferencial de tasas de interés (Yield Spreads).")
    
    target_f = st.selectbox("Seleccionar Par para Pricing:", ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'])
    
    # 1. Obtener Diferencial de Bonos
    # Para EURUSD comparamos Bono Alemán vs Bono USA
    b_ticker = '^GBSYSG' if 'EUR' in target_f else '^GLY' # Simplificación
    q_ticker = '^TNX'
    
    df_b = yf.download(b_ticker, period='60d', interval='1d', progress=False)
    df_q = yf.download(q_ticker, period='60d', interval='1d', progress=False)
    
    if not df_b.empty and not df_q.empty:
        # Calculamos el Spread de Deuda
        spread = df_b['Close'] - df_q['Close']
        
        # Normalizamos para comparar con el precio
        spread_norm = (spread - spread.mean()) / spread.std()
        price_data = yf.download(target_f, period='60d', progress=False)['Close']
        price_norm = (price_data - price_data.mean()) / price_data.std()
        
        # El BEER es la diferencia entre el Spread y el Precio
        desviacion = price_norm.iloc[-1] - spread_norm.iloc[-1]
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Desviación BEER", f"{desviacion:.2f}", 
                      delta="SOBREVALORADO" if desviacion > 0.5 else "INFRAVALORADO" if desviacion < -0.5 else "Precio Justo")
        
        with c2:
            st.info(f"""
            **Lectura Institucional:**
            - Si la línea **Azul (Precio)** está muy por encima de la **Dorada (Bonos)**: El par debe caer para buscar su equilibrio.
            - Este modelo es el que usan los bancos para saber si el movimiento de una moneda está respaldado por los flujos de capital real.
            """)
            
        # Gráfico de Convergencia BEER
        fig_beer = go.Figure()
        fig_beer.add_trace(go.Scatter(x=price_norm.index, y=price_norm, name="Precio (Norm)", line=dict(color='#00ffcc')))
        fig_beer.add_trace(go.Scatter(x=spread_norm.index, y=spread_norm, name="Valor Justo (Bonos)", line=dict(color='#ffd700')))
        fig_beer.update_layout(template="plotly_dark", height=400, title="Convergencia Precio vs Fundamentales")
        st.plotly_chart(fig_beer, use_container_width=True)
