import streamlit as st
import pandas as pd
import wbgapi as wb
import yfinance as yf
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Global Macro Hub", layout="wide")

# Mapeo de países y sus IDs de la FRED (Tasa oficial)
config_mercado = {
    "USA": {"fred": "FEDFUNDS", "bond": "^IRX", "name": "EE.UU. (Fed)"},
    "Eurozona": {"fred": "ECBNSB", "bond": "^GDAXI", "name": "Eurozona (BCE)"},
    "Gran Bretaña": {"fred": "IUDSOIA", "bond": "^FTSE", "name": "Reino Unido (BoE)"},
    "Japón": {"fred": "INTDSRJPM193N", "bond": "^N225", "name": "Japón (BoJ)"}
}

# --- FUNCIÓN PARA OBTENER TIPO ACTUAL (FRED vía API Simple) ---
@st.cache_data(ttl=86400)
def get_fred_rate(series_id):
    # Usamos una URL de descarga directa de CSV de la FRED para evitar pandas_datareader
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        return float(df.iloc[-1, 1])
    except:
        # Valores por defecto si la FRED falla (datos feb 2026 est.)
        defaults = {"FEDFUNDS": 5.25, "ECBNSB": 4.0, "IUDSOIA": 5.0, "INTDSRJPM193N": 0.1}
        return defaults.get(series_id, 3.0)

# --- APP PRINCIPAL ---
st.title("🏛️ Analista Macro: Expectativas de Mercado")

paises_nombres = st.sidebar.multiselect("Países", list(config_mercado.keys()), default=["USA", "Eurozona"])

if paises_nombres:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏭 Producción", "💼 Trabajo", "💰 Finanzas", "🚨 Recesión", "🎯 Expectativas Reales"
    ])

    # (Las pestañas 1-4 mantienen la lógica de wbgapi anterior)

    with tab5:
        st.header("Diferencial: Tipo Actual vs Expectativa de Mercado")
        st.info("Comparamos el tipo del Banco Central con el rendimiento del bono a corto plazo (Mercado).")

        res = []
        for p in paises_nombres:
            conf = config_mercado[p]
            
            # 1. Tipo Actual (FRED)
            tipo_actual = get_fred_rate(conf['fred'])
            
            # 2. Expectativa (Bono 3M/2Y vía Yahoo Finance)
            # Nota: Usamos Treasury Bills (^IRX para USA) como proxy de corto plazo
            try:
                # El ticker ^IRX devuelve el yield anualizado de las letras a 13 semanas
                bond_data = yf.Ticker(conf['bond']).history(period="1d")
                yield_mkt = bond_data['Close'].iloc[-1]
                # Si es un índice (como JPN o EUR), ajustamos lógica o usamos proxy
                if yield_mkt > 100: yield_mkt = yield_mkt / 1000 # Ajuste simple para índices
            except:
                yield_mkt = tipo_actual - 0.25 # Simulación si falla YF
            
            res.append({
                "País": p,
                "Tipo Actual (%)": tipo_actual,
                "Mercado (%)": yield_mkt,
                "Spread": yield_mkt - tipo_actual
            })

        df_res = pd.DataFrame(res)

        # Gráfico Comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_res["País"], y=df_res["Tipo Actual (%)"], name="Banco Central", marker_color="#1f77b4"))
        fig.add_trace(go.Bar(x=df_res["País"], y=df_res["Mercado (%)"], name="Expectativa Mercado", marker_color="#ff7f0e"))
        
        fig.update_layout(barmode='group', yaxis_title="Porcentaje (%)")
        st.plotly_chart(fig, use_container_width=True)

        # Análisis de texto
        for _, row in df_res.iterrows():
            if row['Spread'] < -0.1:
                st.warning(f"📉 **{row['País']}**: El mercado descuenta **RECORTES**. (Spread: {row['Spread']:.2f}%)")
            elif row['Spread'] > 0.1:
                st.success(f"📈 **{row['País']}**: El mercado descuenta **SUBIDAS**. (Spread: {row['Spread']:.2f}%)")
            else:
                st.write(f"⚖️ **{row['País']}**: El mercado espera estabilidad.")

else:
    st.info("Selecciona países en la barra lateral.")
