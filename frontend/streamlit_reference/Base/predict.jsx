/* ============================================================
   Página: Jugar con predicciones (la más importante)
   ============================================================ */
const PredictPage = () => {
  const cases = DATA.demoCases;
  const [caseIdx, setCaseIdx] = React.useState(0);
  const [window_, setWindow] = React.useState(0);
  const [catFilter, setCatFilter] = React.useState("Todos");

  const cats = ["Todos", ...Array.from(new Set(cases.map(c => c.category)))];
  const filtered = catFilter === "Todos" ? cases : cases.filter(c => c.category === catFilter);
  // ensure caseIdx in range when filter changes
  React.useEffect(() => { setCaseIdx(0); }, [catFilter]);
  const cur = filtered[caseIdx] || cases[0];

  // Map class -> ECG shape
  const shapeFor = (cls) => ({
    NSR: "nsr", SB: "brady", ST: "tachy",
    PAC: "afib", PVC: "pvc", AFib: "afib",
    AFlut: "aflut", JR: "brady", VT: "vt", Asys: "nsr",
  })[cls] || "nsr";

  const windows = Array.from({length: 5}).map((_,i)=>cur.window - 2 + i);

  return (
    <div className="page-fade" data-screen-label="07 Jugar">
      <PageHead
        title="Jugar con predicciones"
        lead="Casos demo preseleccionados que ilustran el comportamiento del modelo en distintos escenarios. Esta sección es solo visual: no se ejecuta inferencia en vivo."
        right={<Badge kind="warn">demo · sin inferencia en vivo</Badge>}
      />

      {/* Selectors */}
      <Card title="Panel de selección" sub="casos demo" padLg>
        <div className="grid" style={{gridTemplateColumns:"1fr 1fr 1fr 1fr", gap: 16, alignItems:"flex-end"}}>
          <div>
            <span className="lbl-input">Categoría</span>
            <select className="select" value={catFilter} onChange={e => setCatFilter(e.target.value)}>
              {cats.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <span className="lbl-input">Caso demo</span>
            <select className="select" value={caseIdx} onChange={e => setCaseIdx(Number(e.target.value))}>
              {filtered.map((c,i) => <option key={c.id} value={i}>{c.id} · {c.real} → {c.pred} {c.correct ? "✓" : "✗"}</option>)}
            </select>
          </div>
          <div>
            <span className="lbl-input">Ventana</span>
            <select className="select" value={window_} onChange={e => setWindow(Number(e.target.value))}>
              {windows.map((w,i) => <option key={w} value={i}>w{String(w).padStart(3,"0")} {i===2 ? "(actual)" : ""}</option>)}
            </select>
          </div>
          <div style={{display:"flex", gap: 8, justifyContent:"flex-end"}}>
            <button className="btn ghost" onClick={() => setCaseIdx(Math.max(0, caseIdx-1))}>← anterior</button>
            <button className="btn primary" onClick={() => setCaseIdx(Math.min(filtered.length-1, caseIdx+1))}>siguiente →</button>
          </div>
        </div>

        <div style={{display:"flex", gap:8, marginTop:14, flexWrap:"wrap"}}>
          {cats.slice(1).map(c => {
            const active = catFilter === c;
            return (
              <button key={c} className={`btn ${active ? "primary" : ""}`}
                style={{padding:"6px 12px", fontSize:12}}
                onClick={() => setCatFilter(c)}>
                {c}
              </button>
            );
          })}
        </div>
      </Card>

      {/* Verdict */}
      <section className="section">
        <div className={`verdict ${cur.correct ? "ok" : "err"}`}>
          <div className="icon">
            {cur.correct ? (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            ) : (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            )}
          </div>
          <div style={{flex:1}}>
            <div className="head">
              {cur.correct ? "Predicción correcta" : "Predicción incorrecta"}
              <span style={{marginLeft:10, color:"var(--fg-3)", fontFamily:"var(--mono)", fontSize:11.5}}>· {cur.id}</span>
            </div>
            <div className="body">
              Modelo respondió <b style={{color:"var(--fg-0)"}}>{cur.pred}</b>; etiqueta real <b style={{color:"var(--fg-0)"}}>{cur.real}</b>.
              {" "}Caso etiquetado como <i>{cur.category}</i>.
            </div>
          </div>
          <div style={{display:"flex", gap:8}}>
            <Badge kind={cur.categoryTag}>{cur.category}</Badge>
            <Badge kind="muted">latido t = {cur.beatT.toFixed(2)} s</Badge>
          </div>
        </div>
      </section>

      {/* ECG */}
      <section className="section">
        <div className="ecg-frame">
          <div className="ecg-toolbar">
            <div className="left">
              <span className="ecg-mono"><b>ECG II</b> · case <b>#{cur.caseId}</b> · window <b>w{String(cur.window).padStart(3,"0")}</b></span>
            </div>
            <div className="right">
              <Badge kind="muted">250 Hz</Badge>
              <Badge kind="muted">2.0 s</Badge>
              <Badge kind="info">filtrada 0.5–40 Hz</Badge>
            </div>
          </div>
          <ECGSignal
            width={1200} height={260}
            shape={shapeFor(cur.real)}
            beats={cur.real === "Asys" ? 4 : (cur.real === "VT" ? 11 : 7)}
            seed={cur.caseId}
            realLabel={cur.real}
            predLabel={cur.pred}
            correct={cur.correct}
            pvcIdx={cur.real === "PVC" ? 3 : -1}
            noise={0.025}
          />
          <div style={{display:"flex", gap:14, justifyContent:"space-between", marginTop:10, fontFamily:"var(--mono)", fontSize:11.5, color:"var(--fg-2)"}}>
            <span>🟧 latido central · 🟦 ventana analizada · 🟢/🔴 trazo según resultado</span>
            <span>amplitud relativa · mV (escala estándar 1mV)</span>
          </div>
        </div>
      </section>

      {/* Panel resultado + features + probabilidades */}
      <section className="section grid" style={{gridTemplateColumns:"1.1fr 1fr 1fr"}}>
        <Card title="Resultado de predicción" sub="resumen">
          <div className="grid c2" style={{gap: 10}}>
            <Metric label="Clase real"   value={cur.real}  accent="blue" />
            <Metric label="Clase predicha" value={cur.pred} accent={cur.correct ? "teal" : "warn"} />
            <Metric label="Resultado"    value={cur.correct ? "Correcto" : "Incorrecto"} accent={cur.correct ? "teal" : "warn"} />
            <Metric label="Confianza"    value={Math.round(Math.max(...Object.values(cur.probs))*100) + "%"} accent="blue" />
          </div>
          <hr className="hr-soft" />
          <KV rows={[
            ["Modelo",        "LinearSVC (best)"],
            ["Tipo de caso",  cur.category],
            ["Latido central",`t = ${cur.beatT.toFixed(2)} s`],
            ["Latencia inf.", "0.4 ms / ventana"],
            ["Identifier",    cur.id],
          ]}/>
        </Card>

        <Card title="Probabilidades por clase" sub="softmax estimado">
          <ProbBars probs={cur.probs} correct={cur.correct} />
          <p style={{fontSize:11.5, color:"var(--fg-3)", marginTop: 12, fontFamily:"var(--mono)"}}>
            * LinearSVC no produce probabilidades nativas — se usa CalibratedClassifierCV (sigmoide, 5-fold).
          </p>
        </Card>

        <Card title="Features de la ventana" sub="vector de entrada">
          <div className="tbl-wrap" style={{border:"none"}}>
            <table className="dt">
              <tbody>
                {Object.entries(cur.features).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{fontFamily:"var(--mono)", color:"var(--fg-1)", padding:"7px 14px"}}>{k}</td>
                    <td className="num" style={{padding:"7px 14px"}}>{typeof v === "number" ? v.toFixed(3) : v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      {/* Explicación */}
      <section className="section">
        <Callout kind={cur.correct ? "ok" : "warn"} title="¿Por qué se seleccionó este caso?">
          {cur.note}
        </Callout>
      </section>
    </div>
  );
};

window.PredictPage = PredictPage;
