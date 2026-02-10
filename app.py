import streamlit as st
import pandas as pd
import wbgapi as wb
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# 1. Configuración de página
st.set_page_config(page_title="Global Macro Comparison", layout="wide")

# 2. Diccionario de indicadores y Mapeo de Países
# He mapeado los nombres a sus códigos ISO para el Banco Mundial
mapeo_paises = {
    "Suiza": "CHE",
    "Eurozona": "EMU",
    "Australia": "AUS",
    "Nueva Zelanda": "NZL",
    "USA": "USA",
    "Canadá": "CAN",
    "Gran Bretaña": "GBR",
    "Japón": "JPN"
}

indicadores_wb = {
    'Producción': {'NY.GDP.MKTP.KD.ZG': 'Crecimiento PIB (%)', 'NV.IND.TOTL.ZS': 'Valor Ind. (% PIB)'},
    'Trabajo': {'SL.UEM.TOTL.ZS': 'Desempleo (%)', 'SL.TLF.CACT.ZS': 'Tasa Participación (%)'},
    'Finanzas': {'FP.CPI.TOTL.ZG': 'Inflación (%)', 'GC.DOD.TOTL.GD.ZS': 'Deuda Pública (% PIB)'}
}

st.title("🌍 Monitor de Divisas y Macro Global")

# 3. Sidebar
paises_nombres = st.sidebar.multiselect(
    "Selecciona Países/Regiones", 
    list(mapeo_paises.keys()), 
    default=list(mapeo_paises.keys())
)
paises_ids = [mapeo_paises[name] for name in paises_nombres]

@st.cache_data
def get_macro_data(codes, countries):
    if not countries or not codes: return pd.DataFrame()
    try:
        df = wb.data.DataFrame(codes, countries, mrv=10)
        df = df.reset_index()
        if 'level_1' in df.columns:
            df = df.rename(columns={'level_0': 'economy', 'level_1': 'series'})
        elif 'id' in df.columns:
            df = df.rename(columns={'id': 'economy'})
            
        df_long = pd.melt(df, id_vars=['economy', 'series'], var_name='Año', value_name='Valor')
        df_long['Año'] = df_long['Año'].str.replace('YR', '').astype(int)
        return df_long.dropna(subset=['Valor'])
    except:
        return pd.DataFrame()

# 4. Lógica de Pestañas
if paises_ids:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏭 Producción", "💼 Trabajo", "💰 Finanzas", "🚨 Alertas Recesión", "📈 Expectativas Tipos Reales"
    ])

    # --- Pestañas 1, 2 y 3 (Se mantienen con la lógica robusta anterior) ---
    with tab1:
        data_p = get_macro_data(list(indicadores_wb['Producción'].keys()), paises_ids)
        if not data_p.empty:
            data_p['Indicador'] = data_p['series'].map(indicadores_wb['Producción'])
            st.plotly_chart(px.line(data_p, x='Año', y='Valor', color='economy', facet_col='Indicador', markers=True), use_container_width=True)

    with tab2:
        data_l = get_macro_data(list(indicadores_wb['Trabajo'].keys()), paises_ids)
        if not data_l.empty:
            data_l['Indicador'] = data_l['series'].map(indicadores_wb['Trabajo'])
            st.plotly_chart(px.bar(data_l, x='Año', y='Valor', color='economy', barmode='group', facet_row='Indicador'), use_container_width=True)

    with tab3:
        data_f = get_macro_data(list(indicadores_wb['Finanzas'].keys()), paises_ids)
        if not data_f.empty:
            data_f['Indicador'] = data_f['series'].map(indicadores_wb['Finanzas'])
            st.plotly_chart(px.line(data_f[data_f['Indicador']=='Inflación (%)'], x='Año', y='Valor', color='economy', title="Histórico Inflación (%)"), use_container_width=True)

    with tab4:
        if "USA" in paises_ids:
            try:
                bonos = yf.download(['^TNX', '^IRX'], period='1y', progress=False)['Close']
                diff = bonos['^TNX'] - bonos['^IRX']
                st.plotly_chart(px.area(diff, title="Curva de Tipos USA (10Y - 3M)"), use_container_width=True)
            except: st.warning("Datos de bonos no disponibles.")
        else: st.info("Selecciona 'USA' para ver la curva de tipos.")

    # --- Pestaña 5: EXPECTATIVAS TIPOS REALES (MULTIPAÍS) ---
    with tab5:
        st.header("Comparativa de Tipos Reales Proyectados")
        st.markdown("Diferencial esperado: **Tipo Nominal - Inflación Objetivo**")

        # Simulamos tipos nominales actuales aproximados (puedes ajustarlos manualmente aquí)
        tipos_base = {
            "USA": 5.5, "EMU": 4.5, "AUS": 4.35, "NZL": 5.5, 
            "CAN": 5.0, "GBR": 5.25, "CHE": 1.75, "JPN": 0.1
        }

        col_cfg, col_vis = st.columns([1, 3])
        
        with col_cfg:
            st.subheader("Ajuste de Inflación Esperada")
            expectativas = {}
            for p_nom in paises_nombres:
                id_iso = mapeo_paises[p_nom]
                # Sugerimos la inflación actual del BM o 2% por defecto
                expectativas[id_iso] = st.slider(f"Inflación {p_nom} (%)", -1.0, 10.0, 2.0)
            
            meses_proj = st.slider("Horizonte (meses)", 1, 24, 12)

        with col_vis:
            df_proyecciones = []
            for p_nom in paises_nombres:
                id_iso = mapeo_paises[p_nom]
                tipo_nom = tipos_base.get(id_iso, 3.0)
                inf_final = expectativas[id_iso]
                
                # Obtenemos inflación actual para la trayectoria
                df_act = get_macro_data(['FP.CPI.TOTL.ZG'], [id_iso])
                inf_inicial = float(df_act['Valor'].iloc[-1]) if not df_act.empty else 3.0
                
                # Generamos la curva de tipo real (Nominal - Inflación)
                trayectoria_inf = np.linspace(inf_inicial, inf_final, meses_proj + 1)
                tipos_reales = tipo_nom - trayectoria_inf
                
                for m, val in enumerate(tipos_reales):
                    df_proyecciones.append({
                        'Mes': m,
                        'País': p_nom,
                        'Tipo Real (%)': val
                    })
            
            df_final_proj = pd.DataFrame(df_proyecciones)
            fig_comp = px.line(df_final_proj, x='Mes', y='Tipo Real (%)', color='País',
                              title="Comparativa de Rendimientos Reales Esperados",
                              line_shape="spline", markers=True)
            fig_comp.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.info("💡 **Interpretación:** Los países con las líneas más altas ofrecen mayor rentabilidad real, lo que suele atraer capital y fortalecer su divisa.")

else:
    st.info("Selecciona países en el menú de la izquierda.")
