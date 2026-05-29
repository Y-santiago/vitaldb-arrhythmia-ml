/* ============================================================
   Página: Inicio  (landing / dashboard)
   ============================================================ */
const HomePage = () => {
  const m = DATA.topMetrics;
  return (
    <div className="page-fade" data-screen-label="01 Inicio">
      <section className="hero">
        <div className="hero-grid">
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              <Badge kind="muted">VitalDB · 2026</Badge>
              <Badge kind="info">Machine Learning</Badge>
              <Badge kind="warn">Demo académica</Badge>
            </div>
            <h1>Detección y clasificación de <em>arritmias intraoperatorias</em> con señales ECG y modelos ML.</h1>
            <p>
              Esta demo recorre el pipeline completo: limpieza de señales ECG de la
              VitalDB Arrhythmia Database, generación de ventanas, ingeniería de
              features y comparación de cinco modelos clásicos de clasificación.
            </p>
            <div className="tags">
              <Badge kind="muted">scikit-learn</Badge>
              <Badge kind="muted">XGBoost</Badge>
              <Badge kind="muted">10 clases</Badge>
              <Badge kind="muted">GroupKFold (case_id)</Badge>
            </div>
          </div>
          <div className="ecg-mini">
            <ECGStrip width={520} height={140} beats={6} />
          </div>
        </div>
        <div className="hero-foot">
          <span>VitalDB Arrhythmia Database</span><span className="sep">·</span>
          <span>v1.0.3 pipeline</span><span className="sep">·</span>
          <span>Última corrida · 2026-05-18 14:02 UTC</span>
          <span className="sep">·</span>
          <span style={{ marginLeft: "auto", display: "inline-flex", gap: 6, alignItems: "center" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--teal)" }}></span>
            Pipeline OK · 11/11 etapas
          </span>
        </div>
      </section>

      <section className="section">
        <div className="grid c4">
          <Metric accent="blue" label="Casos quirúrgicos" value={fmtInt(m.cases)} delta="VitalDB" />
          <Metric accent="teal" label="Ventanas ECG"       value="638 k" delta="+5.4% vs piloto" />
          <Metric accent="warn" label="Clases de ritmo"    value={m.classes} delta="multiclase" />
          <Metric             label="Modelos comparados"  value={m.models} delta="baseline + grid" />
        </div>
      </section>

      <section className="section grid c3">
        <Card title="Mejor modelo" sub="resumen" padLg
          right={<Badge kind="ok">winner</Badge>}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginTop: 6 }}>
            <div style={{ fontSize: 28, fontWeight: 600, color: "var(--fg-0)", letterSpacing: "-.01em" }}>
              {m.bestModel}
            </div>
            <span className="ecg-mono">scikit-learn · linear svm</span>
          </div>
          <hr className="hr-soft" />
          <KV rows={[
            ["F1-macro",      <b style={{color:"var(--teal)"}}>0.742</b>],
            ["Precision (M)", "0.751"],
            ["Recall (M)",    "0.738"],
            ["Accuracy",      "0.918"],
            ["t. entren.",    "184 s"],
          ]}/>
        </Card>

        <Card title="Resumen del dataset" sub="VitalDB" padLg>
          <KV rows={[
            ["Casos",            "482 → 461 válidos"],
            ["Señales",          "ECG II (250 Hz)"],
            ["Ventana",          "2.0 s · 50% overlap"],
            ["Ventanas válidas", "638 142"],
            ["Clases",           "10 (NSR + 9 ritmos)"],
            ["Desbalance",       "65% clase mayoritaria"],
          ]}/>
        </Card>

        <Card title="Métrica principal" sub="F1-macro" padLg>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <div style={{ fontSize: 38, fontWeight: 600, color: "var(--fg-0)", letterSpacing: "-.02em" }}>0.742</div>
            <Badge kind="ok">+0.030 vs baseline</Badge>
          </div>
          <p style={{ fontSize: 12, color: "var(--fg-2)", marginTop: 8 }}>
            F1-macro promedia el F1 de cada clase con peso uniforme — penaliza al modelo
            cuando falla en clases minoritarias.
          </p>
          <BarChartH
            data={[
              { label: "LinearSVC", value: 0.742 },
              { label: "RF",         value: 0.721 },
              { label: "XGBoost",    value: 0.715 },
              { label: "MLP",        value: 0.689 },
              { label: "DT",         value: 0.612 },
            ]}
            maxValue={0.8}
            valueFormat={v => v.toFixed(3)}
          />
        </Card>
      </section>

      <section className="section">
        <Callout kind="warn" title="Aviso · esto es una demo académica">
          La aplicación no permite cargar datos nuevos ni realizar diagnóstico clínico.
          Los resultados mostrados provienen de un pipeline ya ejecutado sobre la VitalDB
          Arrhythmia Database. No usar para decisiones médicas reales.
        </Callout>
      </section>
    </div>
  );
};

window.HomePage = HomePage;
