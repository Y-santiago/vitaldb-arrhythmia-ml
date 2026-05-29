/* ============================================================
   Página: Conclusiones
   ============================================================ */
const ConclusionsPage = () => {
  return (
    <div className="page-fade" data-screen-label="09 Conclusiones">
      <PageHead
        title="Conclusiones"
        lead="Resumen de hallazgos, limitaciones y próximos pasos del proyecto académico de clasificación de arritmias intraoperatorias."
        right={<Badge kind="warn">contexto académico</Badge>}
      />

      <section className="grid c2">
        <Card padLg title="Lo que funcionó" sub="fortalezas"
          right={<Badge kind="ok">✓</Badge>}>
          <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13.5, lineHeight:1.75}}>
            <li>Pipeline reproducible end-to-end con artefactos versionados.</li>
            <li>Split por <code style={{fontFamily:"var(--mono)"}}>case_id</code> elimina data leakage entre pacientes.</li>
            <li><b>F1-macro 0.742</b> con un modelo lineal simple, entrenando en menos de 4 min.</li>
            <li>Buen desempeño en 5 de las 10 clases (F1 &gt; 0.75).</li>
            <li>Vector de features compacto (12 dim) facilita interpretación.</li>
          </ul>
        </Card>

        <Card padLg title="Lo que debe mejorar" sub="debilidades"
          right={<Badge kind="warn">!</Badge>}>
          <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13.5, lineHeight:1.75}}>
            <li><b>Clases minoritarias</b> (VT, Asys, AFlut) con F1 &lt; 0.55.</li>
            <li>Confusión sistemática AFib ↔ AFlut y VT ↔ PVC.</li>
            <li>Features espectrales aportan poco al modelo lineal.</li>
            <li>Modelo no usa información de morfología fina del QRS.</li>
            <li>Sin calibración estricta — las probabilidades son aproximaciones.</li>
          </ul>
        </Card>

        <Card padLg title="Riesgos metodológicos" sub="caveats"
          right={<Badge kind="err">⚠</Badge>}>
          <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13.5, lineHeight:1.75}}>
            <li>Etiquetas anotadas semi-automáticamente: posibles ruidos de etiqueta en clases raras.</li>
            <li>Cohorte VitalDB es <i>quirúrgica</i> — generalización a contexto ambulatorio no garantizada.</li>
            <li>Solo se usa derivación II; sistemas reales combinan 12 derivaciones.</li>
            <li>Métricas se reportan en CV; un hold-out externo sería ideal.</li>
            <li>El umbral de decisión no se optimizó por costo clínico.</li>
          </ul>
        </Card>

        <Card padLg title="Próximos pasos" sub="roadmap"
          right={<Badge kind="info">→</Badge>}>
          <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13.5, lineHeight:1.75}}>
            <li>Probar <b>1D-CNN</b> sobre ventana cruda (embedding aprendido).</li>
            <li>Oversampling dirigido (SMOTE) o focal loss para clases raras.</li>
            <li>Augmentation con ruido sintético y desplazamientos.</li>
            <li>Validación externa con MIT-BIH y/o PTB-XL.</li>
            <li>Análisis de errores estratificado por paciente y tipo de cirugía.</li>
            <li>Reemplazar este placeholder de pipeline por diagrama interactivo final.</li>
          </ul>
        </Card>
      </section>

      <section className="section">
        <Callout kind="err" title="Aviso académico final">
          Esta aplicación es una <b>demo académica</b> con fines didácticos. No constituye
          dispositivo médico ni herramienta de soporte a la decisión clínica. Las
          predicciones mostradas provienen de un pipeline entrenado únicamente sobre la
          VitalDB Arrhythmia Database, sin validación clínica externa.{" "}
          <b>No usar en pacientes reales.</b>
        </Callout>
      </section>

      <section className="section grid c4">
        <Metric accent="teal" label="F1-macro final"     value="0.742" delta="+0.030 baseline" />
        <Metric accent="blue" label="Clases con F1>0.75" value="5 / 10" />
        <Metric accent="warn" label="Clases F1<0.55"     value="3 / 10" />
        <Metric             label="Latencia inferencia" value="0.4" unit=" ms/ventana" />
      </section>

      <section className="section">
        <Card title="Recomendaciones para convertir a Streamlit" sub="notas" padLg>
          <div className="grid c2">
            <div>
              <h4 style={{margin:"0 0 8px", color:"var(--fg-0)", fontSize:13}}>Mapeo directo</h4>
              <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13, lineHeight:1.7}}>
                <li>Sidebar → <code style={{fontFamily:"var(--mono)"}}>st.sidebar.radio()</code> con 9 secciones.</li>
                <li>Métricas (Metric cards) → <code style={{fontFamily:"var(--mono)"}}>st.metric()</code> dentro de <code style={{fontFamily:"var(--mono)"}}>st.columns(4)</code>.</li>
                <li>Tablas comparativas → <code style={{fontFamily:"var(--mono)"}}>st.dataframe()</code> con estilos pandas.</li>
                <li>Segmented controls → <code style={{fontFamily:"var(--mono)"}}>st.radio(horizontal=True)</code> o <code style={{fontFamily:"var(--mono)"}}>st.segmented_control()</code>.</li>
                <li>Selectores demo → <code style={{fontFamily:"var(--mono)"}}>st.selectbox()</code>.</li>
              </ul>
            </div>
            <div>
              <h4 style={{margin:"0 0 8px", color:"var(--fg-0)", fontSize:13}}>Gráficos</h4>
              <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13, lineHeight:1.7}}>
                <li>BarChartV / BarChartH → <code style={{fontFamily:"var(--mono)"}}>plotly.express.bar()</code>.</li>
                <li>ECG → <code style={{fontFamily:"var(--mono)"}}>plotly.graph_objects.Scatter()</code> con shapes para ventana y línea vertical.</li>
                <li>Confusion matrix → <code style={{fontFamily:"var(--mono)"}}>plotly.imshow()</code> o <code style={{fontFamily:"var(--mono)"}}>px.imshow()</code>.</li>
                <li>Correlation heatmap → mismo, con <code style={{fontFamily:"var(--mono)"}}>color_continuous_scale="RdBu_r"</code>.</li>
                <li>Pipeline → <code style={{fontFamily:"var(--mono)"}}>st.components.v1.html()</code> con Mermaid o SVG.</li>
              </ul>
            </div>
          </div>
          <hr className="hr-soft" />
          <div className="grid c2">
            <div>
              <h4 style={{margin:"0 0 8px", color:"var(--fg-0)", fontSize:13}}>Estado & cacheo</h4>
              <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13, lineHeight:1.7}}>
                <li><code style={{fontFamily:"var(--mono)"}}>@st.cache_data</code> para leer datasets de ventanas y métricas.</li>
                <li><code style={{fontFamily:"var(--mono)"}}>@st.cache_resource</code> para el modelo serializado.</li>
                <li><code style={{fontFamily:"var(--mono)"}}>st.session_state</code> para el caso demo seleccionado.</li>
              </ul>
            </div>
            <div>
              <h4 style={{margin:"0 0 8px", color:"var(--fg-0)", fontSize:13}}>Estilos</h4>
              <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13, lineHeight:1.7}}>
                <li>Tema oscuro vía <code style={{fontFamily:"var(--mono)"}}>.streamlit/config.toml</code> con paleta de este diseño.</li>
                <li>Card personalizadas → CSS inyectado con <code style={{fontFamily:"var(--mono)"}}>st.markdown(&lt;style&gt;…)</code>.</li>
                <li>IBM Plex Sans + IBM Plex Mono importadas via <code style={{fontFamily:"var(--mono)"}}>@import url(...)</code>.</li>
              </ul>
            </div>
          </div>
        </Card>
      </section>
    </div>
  );
};

window.ConclusionsPage = ConclusionsPage;
