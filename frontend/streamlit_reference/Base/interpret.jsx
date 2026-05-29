/* ============================================================
   Página: Interpretabilidad
   ============================================================ */
const InterpretPage = () => {
  const feats = DATA.features;
  return (
    <div className="page-fade" data-screen-label="08 Interpretabilidad">
      <PageHead
        title="Interpretabilidad"
        lead="Qué features pesan más en la decisión del modelo y cómo se relacionan entre sí. Importancia calculada con permutation importance sobre el conjunto de validación."
        right={<Badge kind="info">permutation importance · 30 perm</Badge>}
      />

      <Callout kind="warn" title="Advertencia">
        El modelo no interpreta señales como un especialista clínico; usa features
        numéricas extraídas de ventanas ECG (intervalos RR, amplitudes, energía,
        cruces por cero). Las explicaciones a continuación describen <i>qué mira el
        modelo</i>, no fisiopatología.
      </Callout>

      <section className="section grid" style={{gridTemplateColumns:"1.4fr 1fr"}}>
        <Card title="Top features" sub="permutation importance">
          <BarChartH
            data={feats.map(f => ({ label: f.name, value: f.imp, note: f.desc }))}
            maxValue={0.2}
            valueFormat={v => v.toFixed(3)}
          />
        </Card>

        <Card title="Features destacadas" sub="explicación">
          <div style={{display:"flex", flexDirection:"column", gap:10}}>
            {feats.slice(0,4).map(f => (
              <div key={f.name} style={{
                padding:"12px 14px", border:"1px solid var(--line-1)",
                borderRadius:10, background:"var(--bg-1)"
              }}>
                <div style={{display:"flex", justifyContent:"space-between", alignItems:"baseline"}}>
                  <code style={{fontFamily:"var(--mono)", fontSize:12.5, color:"var(--fg-0)"}}>{f.name}</code>
                  <span style={{fontFamily:"var(--mono)", fontSize:11.5, color:"var(--teal)"}}>{f.imp.toFixed(3)}</span>
                </div>
                <p style={{margin:"4px 0 0", fontSize:12.5, color:"var(--fg-2)"}}>{f.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>

      <section className="section grid" style={{gridTemplateColumns:"1fr 1fr"}}>
        <Card title="Mapa de correlación entre features" sub="heatmap (Pearson)">
          <CorrelationHeatmap features={feats.slice(0,8).map(f => f.name)} />
          <div style={{display:"flex", justifyContent:"space-between", marginTop:10}}>
            <span className="ecg-mono" style={{color:"var(--fg-3)"}}>diagonal = 1.00 (autocorrelación)</span>
            <div className="cm-legend">
              <span>-1</span>
              <div style={{width:120, height:10, borderRadius:3, background:"linear-gradient(90deg, #f87171, #2a3454, #2dd4bf)", border:"1px solid var(--line-2)"}}/>
              <span>+1</span>
            </div>
          </div>
        </Card>

        <Card title="Insights del modelo" sub="lectura cualitativa">
          <ul style={{margin:0, paddingLeft:18, color:"var(--fg-1)", fontSize:13, lineHeight:1.8}}>
            <li>
              Las features de <b>variabilidad RR</b> (<code style={{fontFamily:"var(--mono)"}}>rr_prev, rr_next, rr_ratio</code>)
              concentran ~49% de la importancia. Tiene sentido: las arritmias se manifiestan
              principalmente en irregularidades del ritmo.
            </li>
            <li>
              <code style={{fontFamily:"var(--mono)"}}>qrs_proxy</code> (duración aproximada del QRS)
              es la cuarta más relevante; permite distinguir orígenes <i>supraventriculares</i> vs
              <i>ventriculares</i> (QRS ancho).
            </li>
            <li>
              <code style={{fontFamily:"var(--mono)"}}>signal_energy</code> y <code style={{fontFamily:"var(--mono)"}}>amplitude_std</code>
              ayudan a marcar ventanas anómalas pero no son discriminantes entre clases similares.
            </li>
            <li>
              Las features <b>espectrales</b> aportan poco — un modelo lineal no aprovecha bien
              relaciones no lineales en este vector.
            </li>
          </ul>
          <Callout kind="info" style={{marginTop:12}} title="Lo que el modelo no ve">
            La forma exacta del QRS, la presencia de onda P o T anómala, o el contexto
            multi-derivación. Versiones futuras podrían incorporar embeddings 1D-CNN
            sobre la ventana cruda.
          </Callout>
        </Card>
      </section>
    </div>
  );
};

/* Sub: heatmap de correlación (placeholder visual deterministico) */
const CorrelationHeatmap = ({ features }) => {
  // Generate a "plausible" symmetric correlation matrix
  const N = features.length;
  const corr = Array.from({length: N}, (_, i) =>
    Array.from({length: N}, (_, j) => {
      if (i === j) return 1;
      const seed = (i * 31 + j * 17 + 11) % 100 / 100;
      const base = 0.05 + 0.65 * seed * (i > j ? 1 : 1);
      const sign = ((i + j) % 3 === 0) ? -1 : 1;
      // make features 0,1,2 correlated among themselves
      if ((i<3 && j<3)) return 0.6 * sign + 0.2 * seed;
      // make 4,5 correlated
      if ((i===4 && j===5) || (i===5 && j===4)) return 0.72;
      return Math.max(-0.9, Math.min(0.9, sign * base));
    })
  );
  // Symmetrize
  for (let i = 0; i < N; i++) for (let j = i+1; j < N; j++) corr[j][i] = corr[i][j];

  const colorFor = v => {
    // -1 (red) ... 0 (dark) ... +1 (teal)
    if (v >= 0) {
      const t = v;
      const c = [
        Math.round(42 + (45 - 42) * t),
        Math.round(52 + (212 - 52) * t),
        Math.round(84 + (191 - 84) * t),
      ];
      return `rgb(${c.join(",")})`;
    } else {
      const t = -v;
      const c = [
        Math.round(42 + (248 - 42) * t),
        Math.round(52 + (113 - 52) * t),
        Math.round(84 + (113 - 84) * t),
      ];
      return `rgb(${c.join(",")})`;
    }
  };

  return (
    <div style={{overflowX:"auto"}}>
      <table className="cm">
        <thead>
          <tr>
            <th className="corner"></th>
            {features.map(f => (
              <th key={f} style={{transform:"rotate(-35deg)", transformOrigin:"left bottom", height:60, fontFamily:"var(--mono)", fontSize:10}}>{f}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((f, i) => (
            <tr key={f}>
              <th style={{textAlign:"right", paddingRight:8, color:"var(--fg-2)", fontFamily:"var(--mono)", fontSize:10}}>{f}</th>
              {features.map((g, j) => (
                <td key={g} className="cell" style={{
                  background: colorFor(corr[i][j]),
                  color: Math.abs(corr[i][j]) > 0.4 ? "#fff" : "rgba(255,255,255,.5)",
                  width: 60, height: 36,
                }}>
                  {corr[i][j].toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

window.InterpretPage = InterpretPage;
