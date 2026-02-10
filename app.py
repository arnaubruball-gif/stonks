import streamlit as st
import pandas as pd
import wbgapi as wb
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Configuración y Mapeos
st.set_page_config(page_title="Global Macro Hub", layout="wide")

mapeo_paises = {
    "USA": "USA", "Eurozona": "EMU", "Australia": "AUS",
    "Nueva Zelanda": "NZL", "Canadá": "CAN", "Gran Bretaña": "GBR", 
    "Japón": "JPN", "Suiza": "CHE"
}

# Indicadores clave para el Score de Salud
# PIB (+), Desempleo (-), Inflación (Cerca de 2%)
indicadores_salud = {
    'PIB': 'NY.GDP.MKTP.KD.ZG',
    'Desempleo': 'SL.UEM.TOTL.ZS',
    'Inflación': 'FP.CPI.TOTL.ZG'
}

@st.cache_data(ttl=86400)
def get_full_macro_data(countries):
    try:
        codes = list(indicadores_salud.values())
        df = wb.data.DataFrame(codes, countries, mrv=1).reset_index()
        # Limpieza de nombres de columnas según la respuesta de WBGAPI
        df.columns = ['economy', 'series', 'valor']
        return df
    except:
        return pd.DataFrame()

# 2. Lógica del Ranking de Salud
def calcular_ranking(df_raw):
    if df_raw.empty: return pd.DataFrame()
    
    # Pivotamos para tener una fila por país
    df = df_raw.pivot(index='economy', columns='series', values='valor')
    inv_map = {v: k for k, v in indicadores_salud.items()}
    df = df.rename(columns=inv_map)
    
    # Cálculo de Score (Simplificado 0-100)
    # Crecimiento > 2% es bueno, Desempleo < 5% es bueno, Inflación 2% es óptimo
    df['Score'] = (
        (df['PIB'].clip(-2, 5) + 2) * 5 +          # Max 35 pts
        (20 - df['Desempleo'].clip(2, 15)) * 3 +   # Max 54 pts
        (10 - abs(df['Inflación'] - 2).clip(0, 10)) # Max 10 pts
    )
    
    # Normalizar score a 0-100
    df['Score'] = df['Score'].apply(lambda x: min(max(x * 1.2, 0), 100))
    return df.sort_values(by='Score', ascending=False)

# 3. Interfaz Principal
st.title("🏛️ Dashboard Macro: Ranking de Salud Global")
paises_sel = st.sidebar.multiselect("Países en Análisis", list(mapeo_paises.keys()), default=list(mapeo_paises.keys())[:5])
paises_ids = [mapeo_paises[p] for p in paises_sel]

if paises_ids:
    # Creamos las pestañas (Salud primero)
    tab_salud, tab_macro, tab_expectativas = st.tabs(["🏥 Salud del País", "📊 Datos Detallados", "🎯 Expectativas Mercado"])

    # --- PESTAÑA SALUD Y RANKING ---
    with tab_salud:
        st.header("🏆 Ranking de Salud Económica")
        raw_data = get_full_macro_data(paises_ids)
        df_ranking = calcular_ranking(raw_data)
        
        if not df_ranking.empty:
            # Re-mapear códigos ISO a nombres comunes para el usuario
            iso_to_name = {v: k for k, v in mapeo_paises.items()}
            df_ranking.index = [iso_to_name.get(x, x) for x in df_ranking.index]
            
            # Gráfico de Ranking
            fig_rank = px.bar(df_ranking, x=df_ranking.index, y='Score', color='Score',
                             color_continuous_scale='RdYlGn', title="Puntuación de Salud (0-100)")
            st.plotly_chart(fig_rank, use_container_width=True)
            
            # Tabla Resumen
            st.subheader("Ficha Técnica por País")
            st.dataframe(df_ranking[['PIB', 'Desempleo', 'Inflación', 'Score']].style.background_gradient(cmap='RdYlGn', subset=['Score']))
            
            # Diagnóstico Visual
            cols = st.columns(len(df_ranking))
            for i, (pais, row) in enumerate(df_ranking.iterrows()):
                with cols[i]:
                    emoji = "🟢" if row['Score'] > 70 else "🟡" if row['Score'] > 40 else "🔴"
                    st.metric(f"{emoji} {pais}", f"{int(row['Score'])} pts")

    # --- PESTAÑA DATOS DETALLADOS ---
    with tab_macro:
        st.header("📈 Evolución Histórica")
        # Aquí puedes mantener tus gráficos de líneas anteriores de PIB e Inflación
        st.info("Utiliza esta pestaña para ver la tendencia de los últimos años.")

    # --- PESTAÑA EXPECTATIVAS (MANTENIDA) ---
    with tab_expectativas:
        # Aquí se mantiene tu código de barras (Tipo Actual vs Bono 2Y)
        st.header("🎯 Sentimiento de Mercado")
        st.write("Datos en tiempo real de tipos de interés y bonos.")
        # [Insertar aquí el bloque de código de la pestaña 5 anterior]

else:
    st.warning("Selecciona países en el sidebar para generar el ranking.")
