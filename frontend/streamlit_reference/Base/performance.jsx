/* ============================================================
   Página: Rendimiento del modelo
   ============================================================ */
const PerformancePage = () => {
  const models = DATA.models;
  const winner = models[0];

  return (
    <div className="page-fade" data-screen-label="04 Performance">
      <PageHead
        title="Rendimiento del modelo"
        lead="Comparación de cinco modelos clásicos sobre el conjunto de test (GroupKFold por case_id). Métrica principal: F1-macro."
        right={
          <Seg value="f1" onChange={()=>{}} options={[
            {label:"F1-macro", value:"f1"},
            {label:"Accuracy", value:"acc"},
            {label:"P / R / F1", value:"prf"},
          ]}/>
        }
      />

      {/* Winner card */}
      <Card padLg className="" title="Modelo ganador" sub="best on test"
        right={<Badge kind="ok">✓ winner</Badge>}>
        <div className="grid" style={{gridTemplateColumns:"1.2fr 1fr 1fr 1fr 1fr", alignItems:"center", gap: 24}}>
          <div>
            <div style={{fontFamily:"var(--mono)", fontSize:11, color:"var(--fg-3)", textTransform:"uppercase", letterSpacing:".1em"}}>scikit-learn · linear svm</div>
            <div style={{fontSize: 34, fontWeight: 600, color:"var(--fg-0)", marginTop:4}}>{winner.id}</div>
            <div style={{display:"flex", gap:6, marginTop:10, flexWrap:"wrap"}}>
              <Badge kind="info">linear</Badge>
              <Badge kind="muted">L2</Badge>
              <Badge kind="muted">hinge</Badge>
              <Badge kind="ok">balanced</Badge>
            </div>
          </div>
          <Metric accent="teal" label="F1-macro"  value={winner.f1.toFixed(3)} delta="+0.021 vs RF" />
          <Metric              label="Precision"  value={winner.prec.toFixed(3)} />
          <Metric              label="Recall"     value={winner.rec.toFixed(3)} />
          <Metric accent="blue" label="Accuracy"  value={winner.acc.toFixed(3)} />
        </div>
      </Card>

      {/* Comparison */}
      <section className="section grid" style={{gridTemplateColumns:"1.4fr 1fr"}}>
        <Card title="F1-macro · comparación" sub="test set">
          <BarChartV
            keys={["prec","rec","f1"]}
            labels={["Precision","Recall","F1-macro"]}
            colors={["#4a8cff", "#7c3aed", "#2dd4bf"]}
            data={models.map(m => ({ label: m.id, prec: m.prec, rec: m.rec, f1: m.f1 }))}
            valueFormat={v => v.toFixed(2)}
            yMax={1}
            height={260}
          />
        </Card>

        <Card title="Tiempo de entrenamiento" sub="segundos">
          <BarChartH
            data={models
              .slice().sort((a,b)=>a.time-b.time)
              .map(m => ({ label: m.id, value: m.time }))}
            color="var(--warn)"
            valueFormat={v => v + " s"}
            accentTop={false}
          />
          <Callout kind="info" title="Trade-off velocidad / calidad" style={{marginTop:12}}>
            LinearSVC entrena en 3 min y obtiene mejor F1-macro que XGBoost (7.5 min) y MLP (14.7 min)
            en este dataset.
          </Callout>
        </Card>
      </section>

      {/* Table */}
      <section className="section">
        <Card title="Tabla comparativa" sub="modelos · CV-test" padLg>
          <div className="tbl-wrap" style={{border:"none"}}>
            <table className="dt">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Modelo</th>
                  <th>F1-macro</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Accuracy</th>
                  <th>Tiempo</th>
                  <th>Estado</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={m.id} className={i===0 ? "winner" : ""}>
                    <td className="num">{i+1}</td>
                    <td style={{fontWeight: i===0 ? 600 : 500}}>{m.id}</td>
                    <td className="num">{m.f1.toFixed(3)}</td>
                    <td className="num">{m.prec.toFixed(3)}</td>
                    <td className="num">{m.rec.toFixed(3)}</td>
                    <td className="num">{m.acc.toFixed(3)}</td>
                    <td className="num">{m.time} s</td>
                    <td>
                      {i===0 ? <Badge kind="ok">winner</Badge> :
                       i===models.length-1 ? <Badge kind="warn">baseline</Badge> :
                       <Badge kind="muted">ok</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      {/* Hyperparams */}
      <section className="section grid c2">
        <Card title="Mejores hiperparámetros" sub="GridSearchCV / RandomizedSearchCV">
          <div style={{display:"flex", flexDirection:"column", gap: 10}}>
            {models.map(m => (
              <div key={m.id} style={{
                display:"grid", gridTemplateColumns:"130px 1fr",
                gap: 14, alignItems:"center",
                padding:"10px 12px", border:"1px solid var(--line-1)",
                borderRadius: 8, background:"var(--bg-1)"
              }}>
                <div style={{fontWeight:600, color:"var(--fg-0)"}}>{m.id}</div>
                <code style={{fontFamily:"var(--mono)", fontSize:12, color:"var(--fg-1)"}}>{m.params}</code>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Estrategia de validación" sub="metodología">
          <KV rows={[
            ["Split",        "StratifiedGroupKFold · 5-fold · group=case_id"],
            ["Test",         "10% casos retenidos"],
            ["Búsqueda",     "RandomizedSearchCV · 60 trials"],
            ["Scoring",      "f1_macro"],
            ["Class weight", "balanced (sklearn)"],
            ["Preproc",      "StandardScaler · sin PCA"],
            ["Seed",         "42"],
          ]}/>
          <Callout kind="ok" title="Sin data leakage">
            El split por <code style={{fontFamily:"var(--mono)"}}>case_id</code> garantiza
            que las ventanas del mismo paciente nunca aparecen en train y test al mismo
            tiempo.
          </Callout>
        </Card>
      </section>
    </div>
  );
};

window.PerformancePage = PerformancePage;
