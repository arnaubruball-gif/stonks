import streamlit as st
import pandas as pd
import wbgapi as wb
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Macro Analyzer Pro", layout="wide")

# 1. Configuración de indicadores ampliados
indicadores = {
    'NY.GDP.MKTP.KD.ZG': 'Crecimiento PIB (%)',
    'FP.CPI.TOTL.ZG': 'Inflación (%)',
    'SL.UEM.TOTL.ZS': 'Desempleo (%)',
    'BN.CAB.XOKA.GD.ZS': 'Cuenta Corriente (% PIB)',
    'SI.POV.GINI': 'Índice Gini (Desigualdad)'
}

st.title("📊 Monitor Macroeconómico Avanzado")

# Sidebar
paises_ids = st.sidebar.multiselect("Selecciona Países", ["USA", "ESP", "MEX", "BRA", "ARG", "DEU"], default=["USA", "ESP"])
meses_proyeccion = st.sidebar.slider("Meses de Proyección (Diferencial)", 1, 12, 6)

@st.cache_data
def obtener_datos(paises):
    df = wb.data.DataFrame(indicadores.keys(), paises, mrv=5).reset_index()
    df = pd.melt(df, id_vars=['economy', 'series'], var_name='Año', value_name='Valor')
    df['Indicador'] = df['series'].map(indicadores)
    return df

if paises_ids:
    data = obtener_datos(paises_ids)
    
    # 2. SECCIÓN: Diferencial de Tipos (Expectativas)
    st.header("🎯 Expectativas: Diferencial de Tipos (Tipos - Inflación)")
    st.info("Este gráfico proyecta el 'Tipo Real'. Un diferencial positivo sugiere política restrictiva; negativo sugiere pérdida de poder adquisitivo.")

    # Simulación de datos de tipos (Para el ejemplo, ya que cambian diario)
    proyecciones = []
    for p in paises_ids:
        # Asumimos tipos base actuales (estimados) y tendencia de inflación
        tipo_nominal = 5.25 if p == "USA" else 4.5 if p == "ESP" else 10.0
        inflacion_actual = data[(data['economy']==p) & (data['Indicador']=='Inflación (%)')]['Valor'].mean()
        
        for m in range(meses_proyeccion + 1):
            # Simulamos una convergencia de la inflación hacia el objetivo
            inf_proyectada = inflacion_actual - (m * 0.1) 
            diferencial = tipo_nominal - inf_proyectada
            proyecciones.append({'País': p, 'Mes': m, 'Diferencial': diferencial})
    
    df_proj = pd.DataFrame(proyecciones)
    fig_diff = px.line(df_proj, x='Mes', y='Diferencial', color='País', 
                      title=f"Evolución del Diferencial Real (Próximos {meses_proyeccion} meses)",
                      labels={'Diferencial': 'Tipo Nominal - Inflación (%)'})
    st.plotly_chart(fig_diff, use_container_width=True)

    # 3. SECCIÓN: Alertas y Nuevos Indicadores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Salud Comercial (Cuenta Corriente)")
        fig_cc = px.bar(data[data['Indicador'] == 'Cuenta Corriente (% PIB)'], 
                        x='Año', y='Valor', color='economy', barmode='group')
        st.plotly_chart(fig_cc, use_container_width=True)
        
    with col2:
        st.subheader("⚖️ Desigualdad (Índice Gini)")
        # El Gini no se mide cada año, tomamos el último disponible
        gini_data = data[data['Indicador'] == 'Índice Gini (Desigualdad)'].dropna()
        if not gini_data.empty:
            fig_gini = px.scatter(gini_data, x='economy', y='Valor', size='Valor', color='economy')
            st.plotly_chart(fig_gini, use_container_width=True)
        else:
            st.warning("No hay datos recientes de Gini para estos países.")

    # 4. Lógica de Advertencia Mejorada
    st.divider()
    st.subheader("🚩 Análisis de Riesgo")
    for p in paises_ids:
        p_data = data[data['economy'] == p]
        # Alerta: Déficit de cuenta corriente > 5%
        cc_val = p_data[p_data['Indicador'] == 'Cuenta Corriente (% PIB)']['Valor'].iloc[-1]
        if cc_val < -5:
            st.error(f"**{p}**: Riesgo de crisis de balanza de pagos. Déficit: {cc_val:.1f}%")
        
        # Alerta: Crecimiento Negativo
        pib_val = p_data[p_data['Indicador'] == 'Crecimiento PIB (%)']['Valor'].iloc[-1]
        if pib_val < 0:
            st.warning(f"**{p}**: Economía en contracción (Recesión técnica). PIB: {pib_val:.1f}%")

else:
    st.warning("Selecciona países para comenzar el análisis.")
