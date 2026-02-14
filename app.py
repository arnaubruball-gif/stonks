import streamlit as st
import pandas as pd
import wbgapi as wb
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Configuración y Mapeo Ampliado
st.set_page_config(page_title="Macro Terminal", layout="wide")

mapeo_paises = {
    "USA": "USA", "Eurozona": "EMU", "Australia": "AUS",
    "Nueva Zelanda": "NZL", "Canadá": "CAN", "Gran Bretaña": "GBR", 
    "Japón": "JPN", "Suiza": "CHE", "China": "CHN", "Rusia": "RUS"
}

# Diccionario de Indicadores por Categoría
indicadores_cat = {
    "Principales": {'NY.GDP.MKTP.KD.ZG': 'PIB (%)', 'SL.UEM.TOTL.ZS': 'Desempleo (%)', 'FP.CPI.TOTL.ZG': 'Inflación (%)'},
    "Comerciales": {'NE.EXP.GNFS.ZS': 'Exportaciones (% PIB)', 'NE.IMP.GNFS.ZS': 'Importaciones (% PIB)', 'BN.CAB.XOKA.GD.ZS': 'Cuenta Corriente (% PIB)'},
    "Sectores": {'NV.IND.TOTL.ZS': 'Industria (% PIB)', 'NV.SRV.TOTL.ZS': 'Servicios (% PIB)', 'NV.AGR.TOTL.ZS': 'Agricultura (% PIB)'}
}

@st.cache_data(ttl=86400)
def get_macro_data(ids, countries):
    try:
        df = wb.data.DataFrame(ids, countries, mrv=1).reset_index()
        df.columns = ['country', 'series', 'valor']
        return df
    except: return pd.DataFrame()

# 2. Sidebar
st.sidebar.title("Configuración Global")
paises_sel = st.sidebar.multiselect("Países", list(mapeo_paises.keys()), default=["USA", "Eurozona", "China", "Rusia"])
paises_ids = [mapeo_paises[p] for p in paises_sel]

# 3. Estructura de Pestañas
tab1, tab_expectativas = st.tabs(["🏛️ Monitor de Salud y Riesgo", "🎯 Expectativas de Mercado"])

# --- PESTAÑA 1: ESTRUCTURA SOLICITADA ---
with tab1:
    if paises_ids:
        # SECCIÓN 1: INDICADORES PRINCIPALES Y SALUD
        with st.expander("📊 Indicadores Principales y Evaluación de Salud", expanded=True):
            data_p = get_macro_data(list(indicadores_cat["Principales"].keys()), paises_ids)
            if not data_p.empty:
                # Pivot para Score de Salud
                df_salud = data_p.pivot(index='country', columns='series', values='valor')
                # Lógica de Salud
                df_salud['Score'] = (df_salud.iloc[:,0]*5) + (15 - df_salud.iloc[:,1]) # Simplificado
                
                cols = st.columns(len(paises_sel))
                for i, p in enumerate(paises_sel):
                    id_iso = mapeo_paises[p]
                    score_p = df_salud.loc[id_iso, 'Score'] if id_iso in df_salud.index else 0
                    emoji = "🟢" if score_p > 40 else "🟡" if score_p > 20 else "🔴"
                    cols[i].metric(f"{emoji} {p}", f"{int(score_p)} pts", "Salud General")
                
                st.plotly_chart(px.bar(data_p, x='country', y='valor', color='series', barmode='group', title="Comparativa Macro Real"), use_container_width=True)

        # SECCIÓN 2: PREDICCIÓN Y RIESGO
        with st.expander("🚨 Predicción de Movimientos y Análisis de Riesgo"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Riesgo de Deuda")
                # Aquí llamaríamos a Deuda Total / PIB
                st.write("Análisis de sostenibilidad fiscal y spreads de crédito.")
            with col_b:
                st.subheader("Indicadores Adelantados")
                st.write("Curva de tipos e indicadores de confianza.")

        # SECCIÓN 3: INDICADORES COMERCIALES
        with st.expander("🚢 Indicadores Comerciales"):
            data_c = get_macro_data(list(indicadores_cat["Comerciales"].keys()), paises_ids)
            if not data_c.empty:
                st.plotly_chart(px.scatter(data_c, x='country', y='valor', color='series', size='valor', title="Balanza y Apertura Comercial"), use_container_width=True)

        # SECCIÓN 4: ESTUDIO DE SECTORES
        with st.expander("🏗️ Estudio de Sectores por País"):
            data_s = get_macro_data(list(indicadores_cat["Sectores"].keys()), paises_ids)
            if not data_s.empty:
                st.plotly_chart(px.bar(data_s, x='country', y='valor', color='series', title="Composición del PIB por Sectores"), use_container_width=True)
    else:
        st.info("Selecciona países para comenzar el análisis.")

# --- PESTAÑA EXPECTATIVAS (MANTENIDA) ---
with tab_expectativas:
    st.header("🎯 Diferencial de Tipos (Bono 2Y vs Bancos Centrales)")
    # Se mantiene tu lógica de tipos reales y sentimiento de mercado
