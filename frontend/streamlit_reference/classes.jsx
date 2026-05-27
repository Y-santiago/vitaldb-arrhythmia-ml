/* ============================================================
   Página: Evaluación por clase
   ============================================================ */
const ClassesPage = () => {
  const classes = DATA.classes;
  const easy = classes.filter(c => c.f1 >= 0.78);
  const hard = classes.filter(c => c.f1 < 0.55);

  return (
    <div className="page-fade" data-screen-label="05 Clases">
      <PageHead
        title="Evaluación por clase"
        lead="Detalle de precision, recall y F1 por cada uno de los 10 ritmos. Permite identificar dónde el modelo funciona y dónde falla."
        right={<Badge kind="info">LinearSVC · test set</Badge>}
      />

      <Callout kind="warn" title="Desbalance de clases">
        El soporte va de 412 k ventanas (NSR) a apenas 805 (Asystole). Las clases
        minoritarias requieren especial cuidado al interpretar precision/recall: con
        pocas muestras, pequeñas variaciones tienen efectos grandes.
      </Callout>

      <section className="section grid c2">
        <Card title="F1-score por clase" sub="barras horizontales">
          <BarChartH
            data={[...classes].sort((a,b)=>b.f1-a.f1).map(c => ({ label: c.id, value: c.f1 }))}
            maxValue={1}
            valueFormat={v => v.toFixed(3)}
          />
        </Card>
        <Card title="Support por clase" sub="ventanas">
          <BarChartH
            data={[...classes].sort((a,b)=>b.support-a.support).map(c => ({ label: c.id, value: c.support }))}
            color="var(--blue)"
            valueFormat={v => v >= 1000 ? Math.round(v/1000) + "k" : v.toString()}
            accentTop={false}
          />
          <Callout kind="info" style={{marginTop:12}}>
            NSR concentra ~65% del soporte. Asystole y VT representan menos del 0.5%
            del dataset.
          </Callout>
        </Card>
      </section>

      <section className="section">
        <Card title="Reporte por clase" sub="precision · recall · f1 · support" padLg>
          <div className="tbl-wrap" style={{border:"none"}}>
            <table className="dt">
              <thead>
                <tr>
                  <th>Clase</th>
                  <th>Nombre</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Support</th>
                  <th>Etiqueta</th>
                </tr>
              </thead>
              <tbody>
                {classes.map(c => {
                  const label = c.f1 >= 0.78 ? {k:"ok", t:"Buen desempeño"} :
                                c.f1 >= 0.6  ? {k:"info", t:"Aceptable"} :
                                c.f1 >= 0.45 ? {k:"warn", t:"Clase difícil"} :
                                               {k:"err", t:"Requiere revisión"};
                  const minority = c.support < 5000;
                  return (
                    <tr key={c.id}>
                      <td style={{fontFamily:"var(--mono)", fontWeight:600, color:"var(--fg-0)"}}>{c.id}</td>
                      <td style={{color:"var(--fg-1)"}}>{c.name}</td>
                      <td className="num">{c.prec.toFixed(3)}</td>
                      <td className="num">{c.rec.toFixed(3)}</td>
                      <td className="num" style={{color: c.f1 >= 0.78 ? "var(--ok)" : c.f1 < 0.5 ? "var(--err)" : "var(--fg-0)"}}>
                        {c.f1.toFixed(3)}
                      </td>
                      <td className="num">{fmtInt(c.support)}</td>
                      <td style={{display:"flex", gap:6}}>
                        <Badge kind={label.k}>{label.t}</Badge>
                        {minority && <Badge kind="warn">minoritaria</Badge>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      <section className="section grid c2">
        <Card title="Clases con buen desempeño" sub={`${easy.length} de 10`}
          right={<Badge kind="ok">F1 ≥ 0.78</Badge>}>
          <div className="grid c2">
            {easy.map(c => (
              <div key={c.id} className="class-card">
                <div className="name">{c.id} <small>{c.name}</small></div>
                <div style={{height: 6, background:"var(--bg-1)", borderRadius:3, overflow:"hidden", border:"1px solid var(--line-1)"}}>
                  <div style={{width: (c.f1*100)+"%", height:"100%", background:"linear-gradient(90deg, var(--teal), var(--blue))"}}/>
                </div>
                <div className="mini">
                  <span>F1 <b>{c.f1.toFixed(2)}</b></span>
                  <span>P <b>{c.prec.toFixed(2)}</b></span>
                  <span>R <b>{c.rec.toFixed(2)}</b></span>
                  <span style={{marginLeft:"auto"}}>n=<b>{fmtInt(c.support)}</b></span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Clases difíciles" sub={`${hard.length} de 10`}
          right={<Badge kind="err">F1 &lt; 0.55</Badge>}>
          <div className="grid c2">
            {hard.map(c => (
              <div key={c.id} className="class-card" style={{borderColor:"rgba(248,113,113,.25)"}}>
                <div className="name" style={{color:"var(--err)"}}>{c.id} <small style={{color:"var(--fg-3)"}}>{c.name}</small></div>
                <div style={{height: 6, background:"var(--bg-1)", borderRadius:3, overflow:"hidden", border:"1px solid var(--line-1)"}}>
                  <div style={{width: (c.f1*100)+"%", height:"100%", background:"linear-gradient(90deg, var(--err), var(--warn))"}}/>
                </div>
                <div className="mini">
                  <span>F1 <b>{c.f1.toFixed(2)}</b></span>
                  <span>P <b>{c.prec.toFixed(2)}</b></span>
                  <span>R <b>{c.rec.toFixed(2)}</b></span>
                  <span style={{marginLeft:"auto"}}>n=<b>{fmtInt(c.support)}</b></span>
                </div>
                <Badge kind="warn">{c.support < 5000 ? "minoritaria · revisar" : "revisar features"}</Badge>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </div>
  );
};

window.ClassesPage = ClassesPage;
