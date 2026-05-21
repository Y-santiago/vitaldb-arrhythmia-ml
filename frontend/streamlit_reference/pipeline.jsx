/* ============================================================
   Página: Pipeline del proyecto
   ============================================================ */
const PipelinePage = () => {
  return (
    <div className="page-fade" data-screen-label="02 Pipeline">
      <PageHead
        title="Pipeline del proyecto"
        lead="Recorrido completo desde los registros crudos de VitalDB hasta la demo interactiva. 9 etapas listas, 2 placeholders para la sección final."
        right={<Seg value="flow" onChange={()=>{}} options={[{label:"Flujo",value:"flow"},{label:"Tabla",value:"tbl"}]}/>}
      />

      <Card title="Pipeline · vista de nodos" sub="end-to-end" padLg
        right={<span className="ecg-mono" style={{color:"var(--fg-3)"}}>11 etapas · 9 done · 1 en curso · 1 pendiente</span>}>
        <div className="pipe">
          {DATA.pipeline.map(p => (
            <div key={p.n} className={`pipe-node ${p.curr ? "curr" : (p.done ? "done" : "todo")}`}>
              <div className="pipe-num">{String(p.n).padStart(2, "0")}</div>
              <div className="pipe-name">{p.name}</div>
              <div className="pipe-sub">{p.sub}</div>
              {p.curr && <Badge kind="info" style={{marginTop:8}}>en curso</Badge>}
            </div>
          ))}
        </div>
      </Card>

      <section className="section grid c3">
        <Card title="Tiempo total" sub="estimado">
          <div style={{ fontSize: 28, fontWeight: 600, color: "var(--fg-0)" }}>~4 h 12 min</div>
          <p style={{ fontSize: 12, color: "var(--fg-2)" }}>
            corrida completa end-to-end · 1 worker · sin reentrenar embeddings.
          </p>
          <KV rows={[
            ["Carga + EDA", "18 min"],
            ["Limpieza",    "26 min"],
            ["Ventanas",    "1 h 04 min"],
            ["Modelado",    "2 h 14 min"],
            ["Evaluación",  "10 min"],
          ]}/>
        </Card>

        <Card title="Artefactos generados" sub="outputs">
          <KV rows={[
            ["sessions/",         "461 .parquet"],
            ["windows.parquet",   "638 142 filas"],
            ["features.parquet",  "12 columnas"],
            ["models/best.pkl",   "LinearSVC"],
            ["reports/metrics/",  "3 archivos"],
            ["reports/cm.png",    "matriz confusión"],
          ]}/>
        </Card>

        <Card title="Decisiones clave" sub="metodología">
          <ul style={{ margin: 0, paddingLeft: 18, color: "var(--fg-1)", fontSize: 13, lineHeight: 1.7 }}>
            <li>Split por <code style={{fontFamily:"var(--mono)"}}>case_id</code> (sin leak).</li>
            <li>Filtro banda 0.5 – 40 Hz + notch 50 Hz.</li>
            <li>Ventana 2.0 s con 50% overlap y latido central.</li>
            <li>Class weight balanced en LinearSVC.</li>
            <li>RandomizedSearchCV con StratifiedGroupKFold.</li>
          </ul>
        </Card>
      </section>

      <section className="section">
        <Card title="Visualización HTML final del pipeline" sub="placeholder"
          right={<Badge kind="muted">a integrar</Badge>}>
          <div style={{
            height: 220,
            borderRadius: 12,
            border: "1px dashed var(--line-3)",
            background:
              "repeating-linear-gradient(45deg, transparent 0 12px, rgba(74,140,255,.035) 12px 24px), var(--bg-1)",
            display: "grid", placeItems: "center",
            color: "var(--fg-2)",
            position: "relative",
            overflow: "hidden",
          }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-3)", letterSpacing: ".12em", textTransform: "uppercase" }}>
                st.components.v1.html(...)
              </div>
              <div style={{ marginTop: 8, fontSize: 14, color: "var(--fg-1)" }}>
                Aquí se renderizará el diagrama interactivo del pipeline (Mermaid / D3 / Plotly Sankey).
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: "var(--fg-3)" }}>
                Reemplazar este placeholder por el HTML exportado desde notebook.
              </div>
            </div>
            {/* decorative ECG line */}
            <svg style={{ position: "absolute", left: 0, right: 0, bottom: 0, opacity: .15 }} viewBox="0 0 900 60" preserveAspectRatio="none" width="100%" height="60">
              <path d="M0 30 L120 30 L130 10 L140 50 L150 30 L300 30 L310 5 L320 55 L330 30 L500 30 L510 12 L520 48 L530 30 L900 30" fill="none" stroke="var(--teal)" strokeWidth="1.5" />
            </svg>
          </div>
        </Card>
      </section>
    </div>
  );
};

window.PipelinePage = PipelinePage;
