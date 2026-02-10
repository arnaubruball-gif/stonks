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
paises = st.sidebar.multiselect("Países", ["USA", "ESP", "DEU", "FRA", "CHN", "MEX", "BRA"], default=["USA", "ESP"])

# 4. Función de obtención de datos
@st.cache_data
def get_macro_data(codes, countries):
    if not countries:
        return pd.DataFrame()
    df = wb.data.DataFrame(codes, countries, mrv=10).reset_index()
    df = pd.melt(df, id_vars=['economy', 'series'], var_name='Año', value_name='Valor')
    df['Año'] = df['Año'].str.replace('YR', '').astype(int)
    return df

# 5. Lógica Principal
if paises:
    # Creamos las 5 pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏭 Producción", 
        "💼 Trabajo", 
        "💰 Finanzas", 
        "🚨 Alertas de Recesión",
        "📈 Expectativas Diferencial"
    ])

    with tab1:
        st.header("Indicadores de Producción")
        codes = list(indicadores_wb['Producción'].keys())
        data_prod = get_macro_data(codes, paises)
        data_prod['Indicador'] = data_prod['series'].map(indicadores_wb['Producción'])
        st.plotly_chart(px.line(data_prod, x='Año', y='Valor', color='economy', facet_col='Indicador', markers=True), use_container_width=True)

    with tab2:
        st.header("Mercado Laboral")
        codes = list(indicadores_wb['Trabajo'].keys())
        data_lab = get_macro_data(codes, paises)
        data_lab['Indicador'] = data_lab['series'].map(indicadores_wb['Trabajo'])
        st.plotly_chart(px.bar(data_lab, x='Año', y='Valor', color='economy', barmode='group', facet_row='Indicador'), use_container_width=True)

    with tab3:
        st.header("Estabilidad Financiera")
        codes = list(indicadores_wb['Finanzas'].keys())
        data_fin = get_macro_data(codes, paises)
        data_fin['Indicador'] = data_fin['series'].map(indicadores_wb['Finanzas'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.line(data_fin[data_fin['Indicador']=='Inflación (%)'], x='Año', y='Valor', color='economy', title="Inflación"), use_container_width=True)
        with col_b:
            st.plotly_chart(px.bar(data_fin[data_fin['Indicador']=='Deuda Pública (% PIB)'], x='Año', y='Valor', color='economy', title="Deuda Pública"), use_container_width=True)

    with tab4:
        st.header("Señales de Alerta Temprana")
        if "USA" in paises:
            try:
                bonos = yf.download(['^TNX', '^IRX'], start=(datetime.now() - timedelta(days=365)))['Close']
                diff = bonos['^TNX'] - bonos['^IRX']
                st.plotly_chart(px.area(diff, title="Curva de Tipos USA (10Y - 3M)"), use_container_width=True)
                if diff.iloc[-1] < 0:
                    st.error(f"🔴 CURVA INVERTIDA: {diff.iloc[-1]:.2f}%")
                else:
                    st.success(f"🟢 Curva normal: {diff.iloc[-1]:.2f}%")
            except:
                st.warning("Error obteniendo datos de bonos en tiempo real.")
        else:
            st.info("Selecciona 'USA' para ver alertas de recesión por curva de tipos.")

    with tab5:
        st.header("Expectativas de Diferencial de Tipos")
        col_in, col_gr = st.columns([1, 2])
        
        with col_in:
            pais_sel = st.selectbox("País para proyectar", paises)
            # Extraer inflación más reciente para ese país
            codes_inf = ['FP.CPI.TOTL.ZG']
            df_inf = get_macro_data(codes_inf, [pais_sel])
            inf_actual = df_inf['Valor'].iloc[-1] if not df_inf.empty else 3.0
            
            tipo_nom = st.number_input("Tipo Interés Nominal (%)", value=5.0)
            inf_exp = st.slider("Inflación esperada (%)", -2.0, 20.0, float(inf_actual))
            meses = st.slider("Meses vista", 1, 24, 12)
            
        with col_gr:
            # Proyección simple
            x = np.linspace(0, meses, meses+1)
            y_inf = np.linspace(inf_actual, inf_exp, meses+1)
            y_diff = tipo_nom - y_inf
            
            df_proj = pd.DataFrame({'Mes': x, 'Inflación': y_inf, 'Diferencial Real': y_diff})
            fig_p = px.line(df_proj, x='Mes', y=['Inflación', 'Diferencial Real'], 
                           title=f"Rendimiento Real Esperado en {pais_sel}")
            fig_p.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_p, use_container_width=True)

else:
    st.warning("Selecciona al menos un país en la barra lateral para cargar los datos.")
