import streamlit as st
import pandas as pd
import wbgapi as wb
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Macro Strategist Hub", layout="wide")

# --- Configuración de Indicadores ---
indicadores_wb = {
    'Producción': {'NY.GDP.MKTP.KD.ZG': 'Crecimiento PIB (%)', 'NV.IND.TOTL.ZS': 'Valor Ind. (% PIB)'},
    'Trabajo': {'SL.UEM.TOTL.ZS': 'Desempleo (%)', 'SL.TLF.CACT.ZS': 'Tasa Participación (%)'},
    'Finanzas': {'FP.CPI.TOTL.ZG': 'Inflación (%)', 'GC.DOD.TOTL.GD.ZS': 'Deuda Pública (% PIB)'}
}

st.title("🏛️ Dashboard Macro: Análisis Sectorial")

# Sidebar
paises = st.sidebar.multiselect("Países", ["USA", "ESP", "DEU", "FRA", "CHN"], default=["USA", "ESP"])

# --- Lógica de Datos ---
@st.cache_data
def get_macro_data(codes, countries):
    df = wb.data.DataFrame(codes, countries, mrv=10).reset_index()
    df = pd.melt(df, id_vars=['economy', 'series'], var_name='Año', value_name='Valor')
    df['Año'] = df['Año'].str.replace('YR', '').astype(int)
    return df

# --- Pestañas ---
tab1, tab2, tab3, tab4 = st.tabs(["🏭 Producción", "💼 Trabajo", "💰 Finanzas", "🚨 Alertas de Recesión"])

with tab1:
    st.header("Indicadores de Producción y Crecimiento")
    codes = list(indicadores_wb['Producción'].keys())
    data = get_macro_data(codes, paises)
    data['Indicador'] = data['series'].map(indicadores_wb['Producción'])
    fig = px.line(data, x='Año', y='Valor', color='economy', facet_col='Indicador', markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Mercado Laboral")
    codes = list(indicadores_wb['Trabajo'].keys())
    data = get_macro_data(codes, paises)
    data['Indicador'] = data['series'].map(indicadores_wb['Trabajo'])
    fig = px.bar(data, x='Año', y='Valor', color='economy', barmode='group', facet_row='Indicador')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Estabilidad Financiera e Inflación")
    col_a, col_b = st.columns(2)
    codes = list(indicadores_wb['Finanzas'].keys())
    data = get_macro_data(codes, paises)
    data['Indicador'] = data['series'].map(indicadores_wb['Finanzas'])
    
    with col_a:
        st.subheader("Inflación Anual")
        fig_inf = px.line(data[data['Indicador']=='Inflación (%)'], x='Año', y='Valor', color='economy')
        st.plotly_chart(fig_inf)
    with col_b:
        st.subheader("Deuda Pública sobre PIB")
        fig_deuda = px.bar(data[data['Indicador']=='Deuda Pública (% PIB)'], x='Año', y='Valor', color='economy')
        st.plotly_chart(fig_deuda)

with tab4:
    st.header("Señales de Alerta Temprana")
    st.info("La inversión de la curva de tipos (10Y - 2Y) suele preceder a una recesión.")
    
    # Obtener Bonos de USA via Yahoo Finance
    try:
        if "USA" in paises:
            bonos = yf.download(['^TNX', '^IRX'], start=(datetime.now() - timedelta(days=365)))['Close']
            # ^TNX = 10Y, ^IRX = 13-week o aproximamos con 2Y si prefieres el ticker '^TYX'
            diff = bonos['^TNX'] - bonos['^IRX']
            
            fig_yield = px.area(diff, title="Diferencial de Tipos USA (10Y - 3M)", 
                               labels={'value': 'Spread (%)', 'Date': 'Fecha'})
            st.plotly_chart(fig_yield, use_container_width=True)
            
            if diff.iloc[-1] < 0:
                st.error(f"🔴 **CURVA INVERTIDA**: El diferencial es de {diff.iloc[-1]:.2f}%. Riesgo de recesión elevado.")
            else:
                st.success(f"🟢 Curva normalizada: {diff.iloc[-1]:.2f}%")
        else:
            st.warning("Selecciona 'USA' para ver el indicador de recesión por diferencial de bonos.")
    except Exception as e:
        st.error(f"Error al obtener datos de bonos: {e}")
