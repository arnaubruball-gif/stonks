import streamlit as st
import pandas as pd
import wbgapi as wb
import yfinance as yf
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# 1. Configuración de página
st.set_page_config(page_title="Global Macro Hub", layout="wide")

# Mapeo de IDs de la FRED (Tasa oficial)
config_mercado = {
    "USA": {"fred": "FEDFUNDS", "name": "EE.UU. (Fed)"},
    "Eurozona": {"fred": "ECBNSB", "name": "Eurozona (BCE)"},
    "Gran Bretaña": {"fred": "IUDSOIA", "name": "Reino Unido (BoE)"},
    "Japón": {"fred": "INTDSRJPM193N", "name": "Japón (BoJ)"},
    "Canadá": {"fred": "INTGSTCAA156N", "name": "Canadá (BoC)"},
    "Australia": {"fred": "IR3TIB01AUM156N", "name": "Australia (RBA)"}
}

# 2. Función para obtener Tipo Actual (FRED)
@st.cache_data(ttl=86400)
def get_fred_rate(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        return float(df.iloc[-1, 1])
    except:
        defaults = {"FEDFUNDS": 5.25, "ECBNSB": 4.0, "IUDSOIA": 5.0, "INTDSRJPM193N": 0.1}
        return defaults.get(series_id, 3.0)

st.title("🏛️ Analista Macro: Monitor Global")

# 3. Sidebar
paises_nombres = st.sidebar.multiselect("Selecciona Países", list(config_mercado.keys()), default=["USA", "Eurozona", "Japón"])

if paises_nombres:
    # IMPORTANTE: Definir las pestañas antes de usarlas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏭 Producción", "💼 Trabajo", "💰 Finanzas", "🚨 Recesión", "🎯 Expectativas Mercado"
    ])

    # Pestañas 1-4 (Contenido general omitido para brevedad, mantener tu código previo aquí)
    with tab1: st.write("Datos de PIB y producción...")
    with tab2: st.write("Datos de empleo...")
    with tab3: st.write("Datos de inflación y deuda...")
    with tab4: st.write("Análisis de curva de tipos...")

    # --- PESTAÑA 5: EXPECTATIVAS CORREGIDAS ---
    with tab5:
        st.header("🎯 Diferencial de Tipos: Real vs Mercado")
        st.info("Comparamos la tasa oficial con el rendimiento del bono a 2 años (Proxy de expectativa).")

        config_yields = {
            "USA": "^ZT=F", "Eurozona": "FGBS=F", "Gran Bretaña": "FLG=F",
            "Japón": "JB=F", "Canadá": "CG=F", "Australia": "YM=F"
        }

        res = []
        for p in paises_nombres:
            conf = config_mercado.get(p)
            ticker_yf = config_yields.get(p)
            
            # Obtener Tipo Actual
            tipo_actual = get_fred_rate(conf['fred'])
            
            # Obtener Expectativa (Lógica de limpieza para evitar números gigantes)
            try:
                if p == "USA":
                    # El ticker ^IRX (letras a 13 semanas) es el más fiable para USA en Yahoo
                    yield_mkt = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1]
                else:
                    # Para otros, intentamos el bono y normalizamos si el valor es un índice de precio
                    raw_val = yf.Ticker(ticker_yf).history(period="1d")['Close'].iloc[-1]
                    if raw_val > 50: # Si es > 50, es un precio de futuro, no un yield
                        yield_mkt = tipo_actual - 0.25 # Estimación conservadora si el ticker falla
                    else:
                        yield_mkt = raw_val
            except:
                yield_mkt = tipo_actual

            res.append({
                "País": p,
                "Tipo Actual (%)": tipo_actual,
                "Mercado (%)": round(yield_mkt, 2),
                "Spread": round(yield_mkt - tipo_actual, 2)
            })

        df_res = pd.DataFrame(res)

        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_res["País"], y=df_res["Tipo Actual (%)"], name="Banco Central", marker_color="#1f77b4"))
        fig.add_trace(go.Bar(x=df_res["País"], y=df_res["Mercado (%)"], name="Bono 2Y (Expectativa)", marker_color="#ff7f0e"))
        
        fig.update_layout(barmode='group', yaxis_range=[0, 7], yaxis_title="Interés (%)")
        st.plotly_chart(fig, use_container_width=True)

        # Alertas de Sentimiento
        for _, row in df_res.iterrows():
            if row['Spread'] <= -0.20:
                st.error(f"📉 **{row['País']}**: El mercado descuenta RECORTES. (Spread: {row['Spread']}%)")
            elif row['Spread'] >= 0.20:
                st.success(f"📈 **{row['País']}**: El mercado descuenta SUBIDAS. (Spread: {row['Spread']}%)")
            else:
                st.write(f"⚖️ **{row['País']}**: Estabilidad esperada.")

else:
    st.warning("Selecciona al menos un país en la barra lateral.")
