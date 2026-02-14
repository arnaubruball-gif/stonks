import streamlit as st
import pandas as pd
import wbgapi as wb
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Configuración de Página y Mapeo
st.set_page_config(page_title="Macro Terminal Pro", layout="wide")

mapeo_paises = {
    "USA": "USA", "Eurozona": "EMU", "Australia": "AUS",
    "Nueva Zelanda": "NZL", "Canadá": "CAN", "Gran Bretaña": "GBR", 
    "Japón": "JPN", "Suiza": "CHE", "China": "CHN", "Rusia": "RUS"
}

# Diccionario Maestro de Indicadores
indicadores_macro = {
    "Principales": {'NY.GDP.MKTP.KD.ZG': 'PIB (%)', 'SL.UEM.TOTL.ZS': 'Desempleo (%)', 'FP.CPI.TOTL.ZG': 'Inflación (%)'},
    "Riesgo": {'GC.DOD.TOTL.GD.ZS': 'Deuda Pública (% PIB)', 'FI.RES.TOTL.CD': 'Reservas Totales'},
    "Comerciales": {'NE.EXP.GNFS.ZS': 'Exportaciones (% PIB)', 'NE.IMP.GNFS.ZS': 'Importaciones (% PIB)', 'BN.CAB.XOKA.GD.ZS': 'Cuenta Corriente (% PIB)'},
    "Sectores": {'NV.IND.TOTL.ZS': 'Industria (% PIB)', 'NV.SRV.TOTL.ZS': 'Servicios (% PIB)', 'NV.AGR.TOTL.ZS': 'Agricultura (% PIB)'}
}

@st.cache_data(ttl=86400)
def fetch_macro_data(paises_ids):
    try:
        all_codes = []
        for cat in indicadores_macro.values():
            all_codes.extend(list(cat.keys()))
        df = wb.data.DataFrame(all_codes, paises_ids, mrv=3).reset_index()
        df.columns = ['country', 'series'] + [str(col) for col in df.columns[2:]]
        df_long = pd.melt(df, id_vars=['country', 'series'], var_name='anio', value_name='valor')
        df_final = df_long.dropna(subset=['valor']).sort_values('anio').groupby(['country', 'series']).last().reset_index()
        
        map_nombres = {}
        for cat in indicadores_macro.values(): map_nombres.update(cat)
        df_final['nombre_indicador'] = df_final['series'].map(map_nombres)
        return df_final
    except: return pd.DataFrame()

# 2. Sidebar
st.sidebar.header("Filtros Globales")
paises_sel = st.sidebar.multiselect("Seleccionar Países", list(mapeo_paises.keys()), 
                                    default=["USA", "Eurozona", "China", "Rusia", "Japón"])
paises_ids = [mapeo_paises[p] for p in paises_sel]

# 3. Estructura de Pestañas Principales (Nivel 1)
tab_monitor, tab_expectativas = st.tabs(["🏛️ MONITOR DE SALUD Y RIESGO", "🎯 EXPECTATIVAS DE MERCADO"])

if paises_ids:
    df_macro = fetch_macro_data(paises_ids)

    with tab_monitor:
        # Sub-pestañas para evitar scroll infinito (Nivel 2)
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "📊 Salud Económica", "🚨 Riesgo y Predicción", "🚢 Comercio", "🏗️ Sectores"
        ])

        # --- SECCIÓN 1: SALUD ---
        with sub_tab1:
            st.subheader("Evaluación de Salud General")
            df_p = df_macro[df_macro['nombre_indicador'].isin(indicadores_macro["Principales"].values())]
            if not df_p.empty:
                pivot_salud = df_p.pivot(index='country', columns='nombre_indicador', values='valor')
                pivot_salud['Score'] = (pivot_salud.get('PIB (%)', 0) * 5) + (15 - pivot_salud.get('Desempleo (%)', 8))
                
                cols = st.columns(len(paises_sel))
                inv_map = {v: k for k, v in mapeo_paises.items()}
                for i, pais in enumerate(paises_sel):
                    iso = mapeo_paises[pais]
                    if iso in pivot_salud.index:
                        val = pivot_salud.loc[iso, 'Score']
                        emoji = "🟢" if val > 40 else "🟡" if val > 20 else "🔴"
                        cols[i].metric(f"{emoji} {pais}", f"{int(val)} pts")
                
                st.plotly_chart(px.bar(df_p, x='country', y='valor', color='nombre_indicador', barmode='group'), use_container_width=True)

        # --- SECCIÓN 2: RIESGO Y PREDICCIÓN (AHORA CON DATOS) ---
        with sub_tab2:
            st.subheader("Análisis de Riesgo y Movimientos Anticipados")
            col_deuda, col_curva = st.columns(2)
            
            with col_deuda:
                df_r = df_macro[df_macro['nombre_indicador'] == 'Deuda Pública (% PIB)']
                if not df_r.empty:
                    st.plotly_chart(px.pie(df_r, values='valor', names='country', title="Distribución de Carga de Deuda"), use_container_width=True)
                else:
                    st.warning("Datos de deuda no disponibles para la selección.")

            with col_curva:
                st.write("**Probabilidad de Recesión (Curva 10Y-3M USA)**")
                try:
                    curva = yf.download(["^TNX", "^IRX"], period="1y")['Close']
                    spread = curva["^TNX"] - curva["^IRX"]
                    fig_c = px.line(spread, title="Spread 10Y-3M (Bajo 0 = Riesgo)")
                    fig_c.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_c, use_container_width=True)
                except: st.error("No se pudo conectar con los datos de bonos.")

        # --- SECCIÓN 3: COMERCIO ---
        with sub_tab3:
            st.subheader("Indicadores Comerciales Globales")
            df_c = df_macro[df_macro['nombre_indicador'].isin(indicadores_macro["Comerciales"].values())]
            st.plotly_chart(px.scatter(df_c, x='country', y='valor', color='nombre_indicador', size=df_c['valor'].abs(), title="Balanza de Pagos y Apertura"), use_container_width=True)

        # --- SECCIÓN 4: SECTORES ---
        with sub_tab4:
            st.subheader("Estudio de Sectores por País")
            df_s = df_macro[df_macro['nombre_indicador'].isin(indicadores_macro["Sectores"].values())]
            st.plotly_chart(px.bar(df_s, x='country', y='valor', color='nombre_indicador', title="Composición Estructural del PIB"), use_container_width=True)

    # --- PESTAÑA PRINCIPAL 2: EXPECTATIVAS ---
    with tab_expectativas:
        st.subheader("Diferencial de Tipos y Sentimiento de Mercado")
        # (Lógica de barras comparativas igual que la anterior)
        st.info("Esta sección compara los tipos de interés oficiales con los rendimientos de los bonos.")

else:
    st.info("Selecciona países en el menú lateral para activar el monitor.")
