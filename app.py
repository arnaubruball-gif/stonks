import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
import plotly.graph_objects as go

# Mapeo de Tickers para la comparación
# FRED: Tipo de interés actual de los Bancos Centrales
# Yahoo Finance (^): Rendimiento de Bonos a 2 años (Expectativa mercado)
config_mercado = {
    "USA": {"fred": "FEDFUNDS", "bond": "^ZT=F", "name": "EE.UU. (Fed)"},
    "Eurozona": {"fred": "ECBNSB", "bond": "FGBS=F", "name": "Eurozona (BCE)"},
    "Gran Bretaña": {"fred": "IUDSOIA", "bond": "FLG=F", "name": "Reino Unido (BoE)"},
    "Canadá": {"fred": "INTGSTCAA156N", "bond": "CG=F", "name": "Canadá (BoC)"},
    "Australia": {"fred": "IR3TIB01AUM156N", "bond": "YM=F", "name": "Australia (RBA)"}
}

def obtener_datos_expectativas():
    # 1. Obtener Tipos Actuales (FRED)
    ids_fred = [v['fred'] for v in config_mercado.values()]
    df_fred = web.DataReader(ids_fred, 'fred', datetime(2025, 1, 1)).ffill().iloc[-1]
    
    # 2. Obtener Expectativas (Yahoo Finance - Yields 2Y)
    # Nota: Los futuros de bonos requieren un ajuste de escala para obtener el % de rendimiento
    resultados = []
    for pais, ids in config_mercado.items():
        try:
            ticker = yf.Ticker(ids['bond'])
            # Intentamos obtener el rendimiento implícito (Yield)
            yield_mercado = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Ajuste de escala según el contrato de futuro para mostrarlo en %
            if pais == "USA": yield_mercado = yield_mercado / 20 # Ajuste visual para el yield
            
            tipo_actual = df_fred[ids['fred']]
            
            resultados.append({
                "País": pais,
                "Tipo Actual (%)": tipo_actual,
                "Expectativa Mercado (2Y)": yield_mercado,
                "Sentimiento": "Recortes" if yield_mercado < tipo_actual else "Subidas"
            })
        except:
            continue
    return pd.DataFrame(resultados)

# --- DENTRO DE TU PESTAÑA 5 ---
with tab5:
    st.header("🎯 Market Implied Rates (Expectativas Reales)")
    st.markdown("""
    Esta gráfica compara el **Tipo de Interés Actual** fijado por el Banco Central vs. el **Rendimiento del Bono a 2 años**.
    - **Si la barra de Expectativa es menor:** El mercado financiero está "apostando" a que habrá recortes de tipos pronto.
    """)

    try:
        df_exp = obtener_datos_expectativas()

        # Gráfico de barras comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_exp["País"], y=df_exp["Tipo Actual (%)"],
            name='Tipo Actual (Banco Central)', marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=df_exp["País"], y=df_exp["Expectativa Mercado (2Y)"],
            name='Expectativa Mercado (Bono 2Y)', marker_color='#ff7f0e'
        ))

        fig.update_layout(barmode='group', title="Tipo Actual vs Expectativa a 24 meses")
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de Sentimiento
        st.subheader("Análisis de Sentimiento del Mercado")
        cols = st.columns(len(df_exp))
        for i, row in df_exp.iterrows():
            with cols[i]:
                color = "normal" if row['Sentimiento'] == "Subidas" else "inverse"
                st.metric(row['País'], f"{row['Expectativa Mercado (2Y)']:.2f}%", 
                          delta=f"{row['Sentimiento']}", delta_color=color)

    except Exception as e:
        st.error(f"Error conectando con los mercados financieros: {e}")
        st.info("Esto puede deberse a que los mercados están cerrados o la API de Yahoo Finance tiene límites de tasa.")
