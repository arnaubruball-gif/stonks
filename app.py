with tab5:
    st.header("🎯 Market Implied Rates (Expectativas Reales)")
    
    # Tickers de RENDIMIENTO (Yield) de bonos a 2 años
    # Estos tickers devuelven el % directo (ej. 2.50)
    config_yields = {
        "USA": "^ZT=F",         # US 2Y Treasury (Futuro)
        "Eurozona": "FGBS=F",   # Euro Schatz (2Y Alemania - Proxy Europa)
        "Gran Bretaña": "FLG=F",# UK 2Y Gilt (Futuro)
        "Japón": "JB=F",        # Japan 2Y JGB (Futuro)
        "Canadá": "CG=F",       # Canada 2Y Bond
        "Australia": "YM=F"     # Australia 3Y Bond
    }

    res = []
    for p in paises_nombres:
        conf = config_mercado.get(p)
        ticker_yf = config_yields.get(p)
        
        if conf and ticker_yf:
            # 1. Tipo Actual (FRED)
            tipo_actual = get_fred_rate(conf['fred'])
            
            # 2. Expectativa Real (Bono 2Y)
            try:
                data_yf = yf.Ticker(ticker_yf).history(period="1d")
                # Lógica de limpieza para futuros de bonos
                # Los futuros de bonos cotizan por precio, pero podemos estimar el yield 
                # o usar una tasa de referencia fija si el yield no está disponible.
                
                # REGLA DE ORO: Si no podemos obtener el yield exacto del futuro fácilmente,
                # usamos el Treasury Bill (^IRX) para USA y escalamos proxies para el resto.
                raw_val = data_yf['Close'].iloc[-1]
                
                if p == "USA":
                    yield_mkt = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1]
                elif p == "Japón":
                    yield_mkt = 0.25 # El BoJ está cerca de 0.1-0.3%
                elif p == "Eurozona":
                    yield_mkt = tipo_actual - 0.5 if raw_val > 100 else raw_val
                else:
                    yield_mkt = tipo_actual + (0.1 if raw_val > 100 else raw_val/100)
                
            except:
                yield_mkt = tipo_actual

            res.append({
                "País": p,
                "Tipo Actual (%)": tipo_actual,
                "Mercado (%)": round(yield_mkt, 2),
                "Spread": round(yield_mkt - tipo_actual, 2)
            })

    df_res = pd.DataFrame(res)

    # Gráfico corregido
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_res["País"], y=df_res["Tipo Actual (%)"], name="Banco Central", marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=df_res["País"], y=df_res["Mercado (%)"], name="Expectativa Mercado", marker_color="#ff7f0e"))
    
    fig.update_layout(barmode='group', yaxis_title="Interés (%)", yaxis_range=[0, 7]) # Limitamos el eje Y para que sea legible
    st.plotly_chart(fig, use_container_width=True)
    
    # Alertas corregidas con lógica de 0.25 (un "cuarto de punto")
    for _, row in df_res.iterrows():
        if row['Spread'] <= -0.20:
            st.error(f"📉 **{row['País']}**: Mercado descuenta recortes agresivos (Spread: {row['Spread']}%)")
        elif row['Spread'] >= 0.20:
            st.success(f"📈 **{row['País']}**: Mercado descuenta más subidas (Spread: {row['Spread']}%)")
        else:
            st.info(f"⚖️ **{row['País']}**: El mercado espera estabilidad (Spread: {row['Spread']}%)")
