# --- Pestañas Actualizadas ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏭 Producción", 
    "💼 Trabajo", 
    "💰 Finanzas", 
    "🚨 Alertas de Recesión",
    "📈 Expectativas Diferencial"
])

# (Las pestañas 1, 2, 3 y 4 se mantienen con el código anterior)

# --- NUEVA PESTAÑA: EXPECTATIVAS ---
with tab5:
    st.header("Expectativas de Política Monetaria")
    st.markdown("""
    Esta sección calcula el **Diferencial de Tipos Reales** proyectado. 
    Un diferencial positivo indica que los tipos están por encima de la inflación (atractivo para inversores), 
    mientras que uno negativo indica pérdida de poder adquisitivo.
    """)

    col_input, col_graph = st.columns([1, 2])

    with col_input:
        st.subheader("Parámetros de Proyección")
        pais_sel = st.selectbox("Seleccionar País para proyecciones", paises)
        
        # Obtenemos datos actuales para la base
        inf_actual = data[(data['economy'] == pais_sel) & (data['Indicador'] == 'Inflación (%)')]['Valor'].iloc[-1]
        
        tipo_nom_manual = st.number_input(f"Tipo Interés Nominal actual ({pais_sel}) %", value=5.25)
        expectativa_inflacion = st.slider("Expectativa de Inflación a 12 meses (%)", 
                                          min_value=-2.0, max_value=20.0, value=float(inf_actual))
        
        meses = st.number_input("Horizonte temporal (meses)", min_value=1, max_value=24, value=12)

    with col_graph:
        # Simulación de trayectoria
        meses_lista = list(range(meses + 1))
        # Trayectoria lineal desde inflación actual a la esperada
        trayectoria_inf = np.linspace(inf_actual, expectativa_inflacion, meses + 1)
        diferenciales = tipo_nom_manual - trayectoria_inf
        
        df_proj = pd.DataFrame({
            'Mes': meses_lista,
            'Inflación Proyectada': trayectoria_inf,
            'Diferencial Real (Tipo - Inf)': diferenciales
        })

        fig_proj = px.line(df_proj, x='Mes', y=['Inflación Proyectada', 'Diferencial Real (Tipo - Inf)'],
                          title=f"Proyección de Diferencial Real para {pais_sel}",
                          labels={'value': 'Porcentaje (%)', 'variable': 'Indicador'},
                          color_discrete_map={
                              'Inflación Proyectada': '#EF553B',
                              'Diferencial Real (Tipo - Inf)': '#00CC96'
                          })
        
        # Añadir línea de equilibrio en 0
        fig_proj.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Punto de Equilibrio")
        
        st.plotly_chart(fig_proj, use_container_width=True)

    # Nota explicativa sobre el impacto
    if diferenciales[-1] > 0:
        st.success(f"Análisis: En {meses} meses, con un diferencial de {diferenciales[-1]:.2f}%, la moneda de {pais_sel} tendería a fortalecerse frente a divisas con tipos reales negativos.")
    else:
        st.warning(f"Análisis: Un diferencial de {diferenciales[-1]:.2f}% sugiere que los tipos no compensan la inflación. Riesgo de fuga de capitales o debilidad de la divisa.")
