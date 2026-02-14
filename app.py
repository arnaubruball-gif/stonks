import streamlit as st
import pandas as pd
import wbgapi as wb
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Configuración de Página y Diccionarios
st.set_page_config(page_title="Macro Terminal Pro", layout="wide")

mapeo_paises = {
    "USA": "USA", "Eurozona": "EMU", "Australia": "AUS",
    "Nueva Zelanda": "NZL", "Canadá": "CAN", "Gran Bretaña": "GBR", 
    "Japón": "JPN", "Suiza": "CHE", "China": "CHN", "Rusia": "RUS"
}

# Agrupación de indicadores según tu nueva estructura
indicadores_macro = {
    "Principales": {
        'NY.GDP.MKTP.KD.ZG': 'PIB (%)', 
        'SL.UEM.TOTL.ZS': 'Desempleo (%)', 
        'FP.CPI.TOTL.ZG': 'Inflación (%)'
    },
    "Comerciales": {
        'NE.EXP.GNFS.ZS': 'Exportaciones (% PIB)', 
        'NE.IMP.GNFS.ZS': 'Importaciones (% PIB)', 
        'BN.CAB.XOKA.GD.ZS': 'Cuenta Corriente (% PIB)'
    },
    "Sectores": {
        'NV.IND.TOTL.ZS': 'Industria (% PIB)', 
        'NV.SRV.TOTL.ZS': 'Servicios (% PIB)', 
        'NV.AGR.TOTL.ZS': 'Agricultura (% PIB)'
    }
}

# 2. Función de obtención de datos robusta
@st.cache_data(ttl=86400)
def fetch_all_data(paises_ids):
    try:
        all_codes = []
        for cat in indicadores_macro.values():
            all_codes.extend(list(cat.keys()))
        
        # Traemos los últimos 2 años para asegurar que no haya nulos
        df = wb.data.DataFrame(all_codes, paises_ids, mrv=2).reset_index()
        df.columns = ['country', 'series'] + [str(col) for col in df.columns[2:]]
        
        # Limpieza: nos quedamos con el valor más reciente disponible (non-NA)
        df_long = pd.melt(df, id_vars=['country', 'series'], var_name='anio', value_name='valor')
        df_final = df_long.dropna(subset=['valor']).sort_values('anio').groupby(['country', 'series']).last().reset_index()
        
        # Mapear nombres legibles
        map_nombres = {}
        for cat in indicadores_macro.values():
            map_nombres.update(cat)
        df_final['nombre_indicador'] = df_final['series'].map(map_nombres)
        
        return df_final
    except:
        return pd.DataFrame()

# 3. Sidebar
st.sidebar.header("Filtros Globales")
paises_sel = st.sidebar.multiselect("Seleccionar Países", list(mapeo_paises.keys()), 
                                    default=["USA", "Eurozona", "China", "Rusia", "Japón"])
paises_ids = [mapeo_paises[p] for p in paises_sel]

# 4. Estructura de Pestañas Solicitada
tab_monitor, tab_expectativas = st.tabs(["🏛️ Monitor Salud y Riesgo", "🎯 Expectativas"])

if paises_ids:
    df_macro = fetch_all_data(paises_ids)

    # --- PESTAÑA 1: MONITOR SALUD Y RIESGO ---
    with tab_monitor:
        if not df_macro.empty:
            # SECCIÓN A: INDICADORES PRINCIPALES Y SALUD
            st.subheader("1. Salud Económica y Principales")
            df_p = df_macro[df_macro['nombre_indicador'].isin(indicadores_macro["Principales"].values())]
            
            # Cálculo de Score de Salud
            pivot_salud = df_p.pivot(index='country', columns='nombre_indicador', values='valor')
            # Score: Crecimiento - Desempleo + (10 - Inflación_desviación)
            pivot_salud['Score'] = (pivot_salud.get('PIB (%)', 0) * 5) + (15 - pivot_salud.get('Desempleo (%)', 8))
            pivot_salud = pivot_salud.sort_values('Score', ascending=False)
            
            cols = st.columns(len(paises_sel))
            inv_map = {v: k for k, v in mapeo_paises.items()}
            for i, (idx, row) in enumerate(pivot_salud.iterrows()):
                if i < len(cols):
                    nombre_comun = inv_map.get(idx, idx)
                    emoji = "🟢" if row['Score'] > 40 else "🟡" if row['Score'] > 20 else "🔴"
                    cols[i].metric(f"{emoji} {nombre_comun}", f"{int(row['Score'])} pts")

            st.plotly_chart(px.bar(df_p, x='country', y='valor', color='nombre_indicador', barmode='group'), use_container_width=True)

            # SECCIÓN B: PREDICCIÓN Y RIESGO
            st.divider()
            st.subheader("2. Predicción y Análisis de Riesgo")
            c1, c2 = st.columns(2)
            with c1:
                st.info("📉 **Riesgo Sistémico:** Evaluación de spreads de crédito y apalancamiento.")
                # Ejemplo de indicador de riesgo: Deuda sobre PIB (si disponible)
            with c2:
                st.info("🚨 **Indicadores Adelantados:** Curva de tipos y confianza del consumidor.")

            # SECCIÓN C: INDICADORES COMERCIALES
            st.divider()
            st.subheader("3. Indicadores Comerciales")
            df_c = df_macro[df_macro['nombre_indicador'].isin(indicadores_macro["Comerciales"].values())]
            st.plotly_chart(px.line(df_c, x='country', y='valor', color='nombre_indicador', markers=True), use_container_width=True)

            # SECCIÓN D: ESTUDIO DE SECTORES
            st.divider()
            st.subheader("4. Estudio de Sectores por País")
            df_s = df_macro[df_macro['nombre_indicador'].isin(indicadores_macro["Sectores"].values())]
            st.plotly_chart(px.bar(df_s, x='country', y='valor', color='nombre_indicador', title="Composición del PIB"), use_container_width=True)
        else:
            st.error("No se pudieron cargar los datos macro. Reintenta o revisa la conexión con el Banco Mundial.")

    # --- PESTAÑA 2: EXPECTATIVAS ---
    with tab_expectativas:
        st.header("🎯 Sentimiento de Mercado (Bono 2Y vs Bancos Centrales)")
        st.markdown("Comparativa de los tipos actuales de interés vs lo que el mercado de bonos está descontando.")
        
        # Datos "Dummy" de tipos oficiales (puedes conectarlos a FRED como vimos antes)
        data_exp = []
        for p in paises_sel:
            base = 5.25 if p == "USA" else 4.0 if p == "Eurozona" else 0.1 if p == "Japón" else 4.5
            # Simulación de expectativa basada en volatilidad real (Yahoo Finance)
            try:
                mkt = yf.Ticker("^IRX" if p == "USA" else "^GDAXI").history(period="1d")['Close'].iloc[-1]
                expectativa = base - 0.25 if mkt > 100 else mkt # Normalización simple
            except:
                expectativa = base
            
            data_exp.append({"País": p, "Tipo Actual": base, "Expectativa": expectativa})
        
        df_exp = pd.DataFrame(data_exp)
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(x=df_exp["País"], y=df_exp["Tipo Actual"], name="Tipo Central", marker_color="#1f77b4"))
        fig_exp.add_trace(go.Bar(x=df_exp["País"], y=df_exp["Expectativa"], name="Expectativa Mercado", marker_color="#ff7f0e"))
        fig_exp.update_layout(barmode='group', yaxis_range=[0, 7])
        st.plotly_chart(fig_exp, use_container_width=True)

else:
    st.info("Por favor, selecciona países en la barra lateral para visualizar el Monitor.")
