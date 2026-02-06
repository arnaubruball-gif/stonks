import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Equity Research Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 8px solid; }
    .good { background-color: #052111; border-color: #3fb950; }
    .bad { background-color: #210505; border-color: #f85149; }
    .neutral { background-color: #1c1c1c; border-color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNCIONES DE CÁLCULO ---
def calcular_dcf(fcf, g1, gp, k, debt, cash, shares):
    """Función limpia para evitar errores de sintaxis en la matriz"""
    if k <= gp or shares <= 0: return 0
    # Fase 1 (5 años)
    fcf_list = [fcf * (1 + g1)**i for i in range(1, 6)]
    fcf_d = sum([f / (1 + k)**i for i, f in enumerate(fcf_list, 1)])
    # Fase 2 (Terminal)
    tv = (fcf_list[-1] * (1 + gp)) / (k - gp)
    tv_d = tv / (1 + k)**5
    return (fcf_d + tv_d - debt + cash) / shares

@st.cache_data
def get_extended_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="5y")
    return stock, info, hist

# --- 3. SIDEBAR ---
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/3310/3310111.png"
st.sidebar.image(LOGO_URL, width=60)
st.sidebar.title("Pro Analyzer v3.0")
ticker = st.sidebar.text_input("Ticker", "NVDA").upper()

try:
    stock, info, hist = get_extended_data(ticker)
    
    st.sidebar.divider()
    g_f1 = st.sidebar.slider("Crecimiento Etapa 1 (%)", 0.0, 60.0, 25.0) / 100
    k = st.sidebar.slider("Tasa Descuento (k) %", 5.0, 15.0, 9.0) / 100
    g_p = 0.02 # Crecimiento perpetuo

    # --- 4. HEADER ---
    st.title(f"{info.get('longName', ticker)}")
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    price = info.get('currentPrice', 1)
    col_h1.metric("Precio", f"{price} {info.get('currency')}")
    col_h2.metric("Market Cap", f"{info.get('marketCap', 0)/1e9:.2f}B")
    col_h3.metric("PER Actual", f"{info.get('trailingPE', 0):.2f}")
    col_h4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

    # --- 5. MODELOS DE ANÁLISIS ---
    t_val, t_quality, t_multiples, t_charts = st.tabs(["💎 Valoración", "🛡️ Calidad (Piotroski)", "📊 Márgenes", "📈 Histórico"])

    with t_val:
        fcf_actual = info.get('freeCashflow') or info.get('operatingCashflow', 0) * 0.8
        debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        shares = info.get('sharesOutstanding', 1)
        
        # DCF Proyectado
        val_dcf = calcular_dcf(fcf_actual, g_f1, g_p, k, debt, cash, shares)
        
        # Modelo Graham
        eps = info.get('trailingEps', 0)
        val_graham = (eps * (8.5 + 2 * (g_f1 * 100)) * 4.4) / 4.5
        
        v1, v2 = st.columns(2)
        v1.metric("Valor DCF (Caja)", f"{val_dcf:.2f}")
        v2.metric("Valor Graham (EPS)", f"{val_graham:.2f}")
        
        st.write("### 📉 Matriz de Sensibilidad (DCF)")
        sk = [k-0.01, k, k+0.01]
        sg = [g_f1-0.05, g_f1, g_f1+0.05]
        
        # Matriz generada con la función limpia
        m_list = [[calcular_dcf(fcf_actual, gi, g_p, ki, debt, cash, shares) for gi in sg] for ki in sk]
        df_sens = pd.DataFrame(m_list, columns=[f"G:{g*100:.0f}%" for g in sg], index=[f"k:{k*100:.1f}%" for ki in sk])
        st.table(df_sens.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    with t_quality:
        st.subheader("Checklist de Salud Financiera")
        points = 0
        checks = {
            "Rentabilidad Neta Positiva": info.get('returnOnAssets', 0) > 0,
            "Flujo de Caja Operativo Positivo": fcf_actual > 0,
            "ROE > 15% (Foso Competitivo)": info.get('returnOnEquity', 0) > 0.15,
            "Margen Bruto > 40% (Pricing Power)": info.get('grossMargins', 0) > 0.40,
            "Liquidez (Current Ratio) > 1.5": info.get('currentRatio', 0) > 1.5
        }
        for text, result in checks.items():
            c1, c2 = st.columns([3, 1])
            c1.write(text)
            if result:
                c2.success("PASADO")
                points += 1
            else:
                c2.error("FALLO")
        st.metric("Puntuación de Calidad", f"{points} / 5")

    with t_multiples:
        st.subheader("Márgenes Operativos")
        st.progress(info.get('grossMargins', 0), text=f"Margen Bruto: {info.get('grossMargins', 0)*100:.1f}%")
        st.progress(info.get('ebitdaMargins', 0), text=f"Margen EBITDA: {info.get('ebitdaMargins', 0)*100:.1f}%")
        st.progress(info.get('profitMargins', 0), text=f"Margen Neto: {info.get('profitMargins', 0)*100:.1f}%")

    with t_charts:
        st.subheader("Evolución de Ingresos y Beneficios")
        # Obtenemos datos anuales
        income_stmt = stock.financials.T
        if not income_stmt.empty:
            chart_data = income_stmt[['Total Revenue', 'Net Income']].sort_index()
            st.bar_chart(chart_data)
            
        else:
            st.warning("Datos financieros históricos no disponibles para este ticker.")

    # --- 6. VEREDICTO FINAL ---
    st.divider()
    objetivo_final = (val_dcf * 0.7) + (val_graham * 0.3)
    margin = ((objetivo_final / price) - 1) * 100
    
    if margin > 20 and points >= 4:
        st.markdown(f"<div class='status-box good'><h3>🚀 COMPRA FUERTE</h3>Potencial: {margin:.1f}% | Calidad: {points}/5</div>", unsafe_allow_html=True)
    elif margin > 0:
        st.markdown(f"<div class='status-box neutral'><h3>⚖️ MANTENER</h3>Potencial: {margin:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='status-box bad'><h3>⚠️ SOBREVALORADA</h3>Potencial: {margin:.1f}%</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error en el análisis: {e}")
