import streamlit as st
import pandas as pd
import wbgapi as wb
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# 1. Configuración de página
st.set_page_config(page_title="Macro Strategist Hub", layout="wide")

# 2. Diccionario de indicadores
indicadores_wb = {
    'Producción': {'NY.GDP.MKTP.KD.ZG': 'Crecimiento PIB (%)', 'NV.IND.TOTL.ZS': 'Valor Ind. (% PIB)'},
    'Trabajo': {'SL.UEM.TOTL.ZS': 'Desempleo (%)', 'SL.TLF.CACT.ZS': 'Tasa Participación (%)'},
    'Finanzas': {'FP.CPI.TOTL.ZG': 'Inflación (%)', 'GC.DOD.TOTL.GD.ZS': 'Deuda Pública (% PIB)'}
}

st.title("🏛️ Dashboard Macro: Análisis Sectorial")

# 3. Sidebar para selección
paises = st.sidebar.multiselect("Países", ["USA", "ESP", "DEU", "FRA", "CHN", "MEX", "BRA", "ARG"], default=["USA", "ESP"])

# 4. Función de obtención de datos (Corregida para evitar KeyError)
@st.cache_data
def get_macro_data(codes, countries):
    if not countries or not codes:
        return pd.DataFrame()
    try:
        # Forzamos que siempre devuelva economy y series como columnas
        df = wb.data.DataFrame(codes, countries, mrv=10, labels=False)
        df = df.reset_index()
        
        # El Banco Mundial a veces devuelve 'economy' y 'series' o nombres compuestos. 
        # Aseguramos compatibilidad:
        if 'level_1' in df.columns: # Caso común en índices multinivel
            df = df.rename(columns={'level_0': 'economy', 'level_1': 'series'})
        
        # Transformar de formato ancho a largo
        df_long = pd.melt(df, id_vars=['economy', 'series'], var_name='Año', value_name='Valor')
        df_long['Año'] = df_long['Año'].str.replace('YR', '').astype(int)
        
        # Limpiar valores nulos para no romper los gráficos
        return df_long.dropna(subset=['Valor'])
    except Exception as e:
        st.error(f"Error en la descarga de datos: {e}")
        return pd.DataFrame()

# 5. Lógica Principal
if paises:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏭 Producción", "💼 Trabajo", "💰 Finanzas", "🚨 Alertas de Recesión", "📈 Expectativas Diferencial"
    ])

    # --- Pestaña 1: Producción ---
    with tab1:
        st.header("Indicadores de Producción")
        codes = list(indicadores_wb['Producción'].keys())
        data_p = get_macro_data(codes, paises)
        if not data_p.empty:
            data_p['Indicador'] = data_p['series'].map(indicadores_wb['Producción'])
            st.plotly_chart(px.line(data_p, x='Año', y='Valor', color='economy', facet_col='Indicador', markers=True), use_container_width=True)

    # --- Pestaña 2: Trabajo ---
    with tab2:
        st.header("Mercado Laboral")
        codes = list(indicadores_wb['Trabajo'].keys())
        data_l = get_macro_data(codes, paises)
        if not data_l.empty:
            data_l['Indicador'] = data_l['series'].map(indicadores_wb['Trabajo'])
            st.plotly_chart(px.bar(data_l, x='Año', y='Valor', color='economy', barmode='group', facet_row='Indicador'), use_container_width=True)

    # --- Pestaña 3: Finanzas ---
    with tab3:
        st.header("Estabilidad Financiera")
        codes = list(indicadores_wb['Finanzas'].keys())
        data_f = get_macro_data(codes, paises)
        if not data_f.empty:
            data_f['Indicador'] = data_f['series'].map(indicadores_wb['Finanzas'])
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(px.line(data_f[data_f['Indicador']=='Inflación (%)'], x='Año', y='Valor', color='economy', title="Inflación (%)"), use_container_width=True)
            with col_b:
                st.plotly_chart(px.bar(data_f[data_f['Indicador']=='Deuda Pública (% PIB)'], x='Año', y='Valor', color='economy', title="Deuda Pública (% PIB)"), use_container_width=True)

    # --- Pestaña 4: Alertas ---
    with tab4:
        st.header("Curva de Tipos (USA)")
        if "USA" in paises:
            try:
                # 10Y (^TNX) vs 3M (^IRX)
                bonos = yf.download(['^TNX', '^IRX'], period='1y')['Close']
                diff = bonos['^TNX'] - bonos['^IRX']
                st.plotly_chart(px.area(diff, title="Spread 10Y - 3M (USA)"), use_container_width=True)
                if diff.iloc[-1] < 0:
                    st.error(f"🔴 CURVA INVERTIDA: {diff.iloc[-1]:.2f} pts. Riesgo de recesión alto.")
                else:
                    st.success(f"🟢 Curva normal: {diff.iloc[-1]:.2f} pts.")
            except:
                st.warning("No se pudieron cargar los datos de bonos en este momento.")

    # --- Pestaña 5: Expectativas Diferencial ---
    with tab5:
        st.header("Proyección de Diferencial Real")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            pais_sel = st.selectbox("País para análisis", paises)
            # Obtenemos la última inflación registrada para sugerir el valor
            df_inf = get_macro_data(['FP.CPI.TOTL.ZG'], [pais_sel])
            last_inf = float(df_inf['Valor'].iloc[-1]) if not df_inf.empty else 3.0
            
            tipo_nom = st.number_input("Tipo Interés Nominal Actual (%)", value=5.25)
            inf_target = st.slider("Expectativa Inflación (12 meses)", -2.0, 20.0, last_inf)
            
        with col2:
            meses_list = np.arange(0, 13)
            # Simulación de trayectoria lineal de inflación hacia el target
            inf_path = np.linspace(last_inf, inf_target, 13)
            real_rate = tipo_nom - inf_path
            
            df_plot = pd.DataFrame({'Mes': meses_list, 'Inflación': inf_path, 'Tipo Real': real_rate})
            fig = px.line(df_plot, x='Mes', y=['Inflación', 'Tipo Real'], 
                         title=f"Evolución Esperada: {pais_sel}",
                         color_discrete_sequence=["#FF4B4B", "#00CC96"])
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Selecciona países en la barra lateral para comenzar.")
