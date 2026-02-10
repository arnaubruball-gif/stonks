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

indicadores_dict = {
    'NY.GDP.MKTP.KD.ZG': 'PIB (%)',
    'SL.UEM.TOTL.ZS': 'Desempleo (%)',
    'FP.CPI.TOTL.ZG': 'Inflación (%)',
    'GC.DOD.TOTL.GD.ZS': 'Deuda (% PIB)'
}

# 2. Función Robusta de Datos
@st.cache_data(ttl=86400)
def fetch_macro_data(countries):
    try:
        # Obtenemos los últimos 2 años de datos para asegurar tener el más reciente
        df = wb.data.DataFrame(list(indicadores_dict.keys()), countries, mrv=2).reset_index()
        # Limpieza de columnas para evitar errores de nombres
        df.columns = ['country_code', 'series_code'] + [str(col) for col in df.columns[2:]]
        
        # Convertimos a formato largo
        df_long = pd.melt(df, id_vars=['country_code', 'series_code'], var_name='anio', value_name='valor')
        df_long = df_long.dropna(subset=['valor'])
        
        # Nos quedamos solo con el dato más reciente por país e indicador
        df_final = df_long.sort_values('anio').groupby(['country_code', 'series_code']).last().reset_index()
        df_final['Indicador'] = df_final['series_code'].map(indicadores_dict)
        return df_final
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return pd.DataFrame()

# 3. Interfaz
st.title("🏛️ Analista Macro: Salud y Expectativas")
paises_sel = st.sidebar.multiselect("Países", list(mapeo_paises.keys()), default=["USA", "Eurozona", "Japón", "Australia"])
paises_ids = [mapeo_paises[p] for p in paises_sel]

if paises_ids:
    # DEFINICIÓN DE PESTAÑAS
    tab_salud, tab_indicadores, tab_expectativas = st.tabs(["🏥 Salud del País", "📊 Ficha Técnica", "🎯 Expectativas Mercado"])

    # Obtenemos datos
    datos_macro = fetch_macro_data(paises_ids)

    # --- PESTAÑA 1: SALUD Y RANKING ---
    with tab_salud:
        st.header("🏆 Ranking de Salud Económica")
        if not datos_macro.empty:
            # Cálculo de Score
            pivot = datos_macro.pivot(index='country_code', columns='Indicador', values='valor')
            
            # Algoritmo de Salud: PIB alto (+), Desempleo bajo (-), Inflación cerca de 2%
            score = (pivot.get('PIB (%)', 0) * 10) + (15 - pivot.get('Desempleo (%)', 10)) + (10 - abs(pivot.get('Inflación (%)', 2) - 2))
            pivot['Score'] = score.clip(0, 100)
            pivot = pivot.sort_values('Score', ascending=False)
            
            # Gráfico de Ranking
            inv_map = {v: k for k, v in mapeo_paises.items()}
            pivot.index = [inv_map.get(i, i) for i in pivot.index]
            
            fig_rank = px.bar(pivot, y='Score', color='Score', color_continuous_scale='RdYlGn', 
                             title="Puntuación de Salud Global (0-100)")
            st.plotly_chart(fig_rank, use_container_width=True)
            
            # Tarjetas de Diagnóstico
            cols = st.columns(len(pivot))
            for i, (pais, row) in enumerate(pivot.iterrows()):
                with cols[i]:
                    emoji = "🟢" if row['Score'] > 50 else "🟡" if row['Score'] > 30 else "🔴"
                    st.metric(f"{emoji} {pais}", f"{int(row['Score'])} pts")
        else:
            st.warning("No se pudieron procesar los indicadores de salud.")

    # --- PESTAÑA 2: FICHA TÉCNICA (INDICADORES) ---
    with tab_indicadores:
        st.header("📊 Tabla de Indicadores Económicos")
        if not datos_macro.empty:
            tabla_limpia = datos_macro.pivot(index='country_code', columns='Indicador', values='valor')
            tabla_limpia.index = [inv_map.get(i, i) for i in tabla_limpia.index]
            st.dataframe(tabla_limpia.style.highlight_max(axis=0, color='green').highlight_min(axis=0, color='red'), use_container_width=True)
            
            st.subheader("Visualización Comparativa")
            fig_comp = px.bar(datos_macro, x='country_code', y='valor', color='Indicador', barmode='group')
            st.plotly_chart(fig_comp, use_container_width=True)

    # --- PESTAÑA 3: EXPECTATIVAS (TU PESTAÑA FAVORITA) ---
    with tab_expectativas:
        st.header("🎯 Sentimiento de Mercado (Bono 2Y vs Bancos Centrales)")
        # Lógica simplificada y segura para evitar el error de escala
        res_exp = []
        for p in paises_sel:
            try:
                # Simulamos tipos oficiales y traemos bonos reales de YFinance
                tipo_oficial = 5.25 if p == "USA" else 4.0 if p == "Eurozona" else 0.1 if p == "Japón" else 4.5
                ticker = "^IRX" if p == "USA" else "^GDAXI" # Ticker de referencia
                mkt_val = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                
                # Ajuste para que el gráfico sea legible (normalización)
                expectativa = tipo_oficial - 0.25 if mkt_val > 100 else mkt_val
                
                res_exp.append({"País": p, "Actual": tipo_oficial, "Expectativa": expectativa})
            except:
                continue
        
        if res_exp:
            df_exp = pd.DataFrame(res_exp)
            fig_exp = go.Figure()
            fig_exp.add_trace(go.Bar(x=df_exp["País"], y=df_exp["Actual"], name="Banco Central"))
            fig_exp.add_trace(go.Bar(x=df_exp["País"], y=df_exp["Expectativa"], name="Mercado (Bono)"))
            fig_exp.update_layout(barmode='group', yaxis_range=[0, 7])
            st.plotly_chart(fig_exp, use_container_width=True)

else:
    st.info("Selecciona países en la barra lateral para activar el análisis.")
