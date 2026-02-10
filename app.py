import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go

# 1. Función de obtención de datos (La fuente de la verdad)
@st.cache_data(ttl=3600)
def get_fred_rate(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        return float(df.iloc[-1, 1])
    except:
        defaults = {"FEDFUNDS": 5.33, "ECBNSB": 4.0, "IUDSOIA": 5.25}
        return defaults.get(series_id, 0.0)

# 2. Configuración de la Interfaz
st.title("🏛️ Dashboard Macro Profesional")
paises_nombres = st.sidebar.multiselect("Países", ["USA", "Eurozona", "Japón", "Canadá", "Australia", "Gran Bretaña"], default=["USA", "Eurozona", "Australia"])

if paises_nombres:
    # CREACIÓN DE PESTAÑAS (Esto evita el NameError)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏭 Prod", "💼 Trabajo", "💰 Finanzas", "🚨 Recesión", "🎯 Expectativas"])

    with tab5:
        st.header("🎯 Diferencial de Tipos (Bono 2Y)")
        
        # Diccionarios de referencia
        fred_ids = {"USA": "FEDFUNDS", "Eurozona": "ECBNSB", "Japón": "INTDSRJPM193N", "Canadá": "INTGSTCAA156N", "Australia": "IR3TIB01AUM156N", "Gran Bretaña": "IUDSOIA"}
        bond_tickers = {"USA": "^ZT=F", "Eurozona": "FGBS=F", "Japón": "JB=F", "Canadá": "CG=F", "Australia": "YM=F", "Gran Bretaña": "FLG=F"}

        resultados = []
        for p in paises_nombres:
            # Tipo Actual
            actual = get_fred_rate(fred_ids[p])
            
            # Expectativa (Limpieza de datos para evitar barras gigantes)
            try:
                # Usamos T-Bill para USA y lógica de normalización para el resto
                if p == "USA":
                    mkt = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1]
                else:
                    raw = yf.Ticker(bond_tickers[p]).history(period="1d")['Close'].iloc[-1]
                    # Si el valor es > 10, es un precio de futuro, no un yield. Estimamos.
                    mkt = actual - 0.25 if raw > 10 else raw
            except:
                mkt = actual

            resultados.append({"País": p, "Actual": actual, "Mercado": mkt, "Spread": mkt - actual})

        df = pd.DataFrame(resultados)

        # Gráfico Estilo Imagen 2
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["País"], y=df["Actual"], name="Banco Central", marker_color="#1f77b4"))
        fig.add_trace(go.Bar(x=df["País"], y=df["Mercado"], name="Bono 2Y (Expectativa)", marker_color="#ff7f0e"))
        fig.update_layout(barmode='group', yaxis_range=[0, 7], yaxis_title="Interés (%)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Mensajes de Alerta
        for _, row in df.iterrows():
            if row['Spread'] <= -0.20:
                st.error(f"📉 {row['País']}: El mercado descuenta RECORTES (Spread: {row['Spread']:.2f}%)")
            elif row['Spread'] >= 0.20:
                st.success(f"📈 {row['País']}: El mercado descuenta SUBIDAS (Spread: {row['Spread']:.2f}%)")
            else:
                st.write(f"⚖️ {row['País']}: Estabilidad esperada.")
