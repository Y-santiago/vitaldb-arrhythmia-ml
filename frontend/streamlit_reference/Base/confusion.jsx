/* ============================================================
   Página: Matriz de confusión
   ============================================================ */
const ConfusionPage = () => {
  const [mode, setMode] = React.useState("abs");
  const classes = DATA.classes;
  return (
    <div className="page-fade" data-screen-label="06 Matriz">
      <PageHead
        title="Matriz de confusión"
        lead="Filas = clase real, columnas = clase predicha. La diagonal son aciertos; fuera de la diagonal aparecen los errores más frecuentes."
        right={
          <Seg value={mode} onChange={setMode} options={[
            {label:"Conteos absolutos", value:"abs"},
            {label:"Normalizada (por fila)", value:"norm"},
          ]}/>
        }
      />

      <section className="grid" style={{gridTemplateColumns:"1.5fr 1fr"}}>
        <Card title="Confusion matrix" sub={mode==="abs" ? "ventanas" : "% por fila"} padLg
          right={
            <div className="cm-legend">
              <span>low</span>
              <div className="bar" />
              <span>high</span>
            </div>
          }>
          <ConfusionMatrix classes={classes} normalized={mode==="norm"} />
          <div style={{marginTop: 14, display:"flex", justifyContent:"space-between", fontFamily:"var(--mono)", fontSize:11, color:"var(--fg-3)"}}>
            <span>Predicted →</span>
            <span>diag (verdaderos positivos) · off-diag (errores)</span>
          </div>
        </Card>

        <div className="stack">
          <Card title="Cómo leer la matriz" sub="guía rápida">
            <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13, lineHeight:1.7}}>
              <li><b style={{color:"var(--fg-0)"}}>Filas</b>: clase verdadera (lo que el rótulo dice que es).</li>
              <li><b style={{color:"var(--fg-0)"}}>Columnas</b>: clase predicha por el modelo.</li>
              <li>La <b style={{color:"var(--teal)"}}>diagonal</b> son aciertos (TP).</li>
              <li>Fuera de la diagonal son <b style={{color:"var(--err)"}}>errores</b> — la columna indica con qué clase confundió.</li>
              <li>La vista <i>normalizada</i> divide cada fila por su soporte: cada celda es un % del total de esa clase.</li>
            </ul>
          </Card>

          <Card title="Resumen" sub="agregado">
            <div className="grid c2" style={{gap: 10}}>
              <Metric accent="teal" label="Aciertos totales" value="585 871" />
              <Metric accent="warn" label="Errores totales"  value="52 271" />
              <Metric              label="Macro recall"     value="0.738" />
              <Metric              label="Macro precision"  value="0.751" />
            </div>
          </Card>
        </div>
      </section>

      <section className="section grid c2">
        <Card title="Errores más frecuentes" sub="confusiones reales · k">
          <div className="tbl-wrap" style={{border:"none"}}>
            <table className="dt">
              <thead>
                <tr>
                  <th>Real</th>
                  <th></th>
                  <th>Predicha</th>
                  <th>Casos</th>
                  <th>% de la fila</th>
                  <th>Severidad</th>
                </tr>
              </thead>
              <tbody>
                {DATA.topConfusions.map((c,i) => (
                  <tr key={i}>
                    <td style={{fontFamily:"var(--mono)", fontWeight:600, color:"var(--fg-0)"}}>{c.a}</td>
                    <td style={{color:"var(--fg-3)"}}>→</td>
                    <td style={{fontFamily:"var(--mono)", color:"var(--err)"}}>{c.b}</td>
                    <td className="num">{fmtInt(c.n)}</td>
                    <td className="num">{(c.pct*100).toFixed(1)}%</td>
                    <td>
                      <Badge kind={c.pct > 0.3 ? "err" : c.pct > 0.15 ? "warn" : "info"}>
                        {c.pct > 0.3 ? "alta" : c.pct > 0.15 ? "media" : "baja"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Clases más confundidas" sub="pares">
          <div className="grid c2">
            {[
              { pair: "AFib ↔ AFlut", note: "Sin onda F nítida, las features RR no separan", k: "err" },
              { pair: "VT ↔ PVC",     note: "Ráfagas cortas de VT se ven como PVCs aisladas", k: "err" },
              { pair: "Asys ↔ NSR",   note: "Silencios cortos durante anestesia se confunden", k: "err" },
              { pair: "PAC ↔ AFib",   note: "Variabilidad RR moderada, frontera difusa", k: "warn" },
            ].map((c,i) => (
              <div key={i} className="class-card" style={{borderColor: c.k==="err" ? "rgba(248,113,113,.25)" : "rgba(251,191,36,.25)"}}>
                <div className="name" style={{color: c.k==="err" ? "var(--err)" : "var(--warn)"}}>{c.pair}</div>
                <div style={{fontSize: 12, color:"var(--fg-2)"}}>{c.note}</div>
                <Badge kind={c.k}>{c.k === "err" ? "confusión crítica" : "confusión frecuente"}</Badge>
              </div>
            ))}
          </div>
          <Callout kind="warn" style={{marginTop: 12}} title="Implicación clínica simulada">
            Las confusiones <b>VT ↔ PVC</b> y <b>Asys ↔ NSR</b> son las más sensibles
            desde el punto de vista clínico — en una versión real del sistema requerirían
            un umbral de confianza alto antes de generar una alerta.
          </Callout>
        </Card>
      </section>
    </div>
  );
};

window.ConfusionPage = ConfusionPage;
