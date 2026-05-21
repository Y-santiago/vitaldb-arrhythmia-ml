/* ============================================================
   Página: Dataset y limpieza
   ============================================================ */
const DatasetPage = () => {
  const c = DATA.cleaning;
  return (
    <div className="page-fade" data-screen-label="03 Dataset">
      <PageHead
        title="Dataset y limpieza"
        lead="Auditoría de calidad sobre la VitalDB Arrhythmia Database. Se eliminan ventanas con NaN, saturación y SNR bajo antes de generar las features."
        right={<Badge kind="info">VitalDB · ECG II · 250 Hz</Badge>}
      />

      <section className="grid c6">
        <Metric accent="blue" label="Señales auditadas"  value={fmtInt(c.signalsAudited)} />
        <Metric accent="blue" label="Ventanas generadas" value="712 380" />
        <Metric accent="teal" label="Ventanas válidas"   value="638 142" delta="89.6%" />
        <Metric accent="warn" label="Descartadas"        value="74 238" delta="-10.4%" />
        <Metric accent="teal" label="Casos válidos"      value={fmtInt(c.casesValid)} />
        <Metric accent="warn" label="Casos excluidos"    value={fmtInt(c.casesExcluded)} />
      </section>

      <section className="section grid" style={{gridTemplateColumns:"1.4fr 1fr"}}>
        <Card title="Distribución de duración de señales" sub="histograma">
          <Histogram color="var(--blue)" bins={[
            {label:"0", n: 12}, {label:"", n: 38}, {label:"", n: 92}, {label:"30m", n: 184},
            {label:"", n: 312}, {label:"", n: 421}, {label:"1h", n: 498}, {label:"", n: 462},
            {label:"", n: 348}, {label:"2h", n: 246}, {label:"", n: 158}, {label:"", n: 94},
            {label:"3h", n: 58}, {label:"", n: 32}, {label:"", n: 18}, {label:"4h+", n: 8},
          ]}/>
          <div style={{display:"flex", justifyContent:"space-between", marginTop:6, fontSize:11, fontFamily:"var(--mono)", color:"var(--fg-3)"}}>
            <span>duración del registro ECG por caso</span>
            <span>mediana 1h 02m · IQR 38m</span>
          </div>
        </Card>

        <Card title="Razones de descarte" sub="barras">
          <BarChartH
            data={DATA.discardReasons.map(r => ({ label: r.reason, value: r.n }))}
            valueFormat={v => fmtInt(v)}
          />
        </Card>
      </section>

      <section className="section grid" style={{gridTemplateColumns:"1.4fr 1fr"}}>
        <Card title="Distribución de clases" sub="ventanas válidas">
          <BarChartV
            keys={["v"]} labels={["ventanas"]} colors={["var(--blue)"]}
            valueFormat={v => v >= 1000 ? Math.round(v/1000) + "k" : v.toString()}
            data={DATA.classes.map(cl => ({ label: cl.id, v: cl.support }))}
            height={220}
            yMax={420000}
          />
          <Callout kind="warn" title="Desbalance importante">
            La clase NSR concentra ~65% de las ventanas. Asystole y VT representan
            &lt; 0.5% cada una. Esto justifica usar F1-macro como métrica principal
            y <code style={{fontFamily:"var(--mono)"}}>class_weight=balanced</code>.
          </Callout>
        </Card>

        <Card title="Antes / después de limpieza" sub="ventanas">
          <BarChartV
            keys={["before","after"]} labels={["antes","después"]}
            colors={["var(--line-3)", "var(--teal)"]}
            valueFormat={v => v >= 1000 ? Math.round(v/1000) + "k" : v.toString()}
            data={[
              { label: "Tot",   before: 712380, after: 638142 },
              { label: "NaN",   before: 38_412, after: 0 },
              { label: "Sat",   before: 22_104, after: 412 },
              { label: "SNR-",  before: 14_982, after: 1_812 },
              { label: "Pulso", before: 9_140,  after: 91 },
            ]}
            height={220}
            yMax={750000}
          />
        </Card>
      </section>

      <section className="section">
        <Card title="Resumen tabla de limpieza" sub="por fase" padLg>
          <div className="tbl-wrap" style={{border:"none"}}>
            <table className="dt">
              <thead>
                <tr>
                  <th>Fase</th>
                  <th>Operación</th>
                  <th>Entrada</th>
                  <th>Salida</th>
                  <th>Δ</th>
                  <th>Estado</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>1</td><td>Lectura VitalDB</td><td className="num">482 casos</td><td className="num">482</td><td className="num">0</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>2</td><td>Exclusión casos cortos (&lt; 10 min)</td><td className="num">482</td><td className="num">472</td><td className="num">-10</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>3</td><td>Eliminar casos sin ECG II</td><td className="num">472</td><td className="num">461</td><td className="num">-11</td><td><Badge kind="warn">aviso</Badge></td></tr>
                <tr><td>4</td><td>Bandpass 0.5–40 Hz + notch 50 Hz</td><td className="num">461</td><td className="num">461</td><td className="num">0</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>5</td><td>Generación de ventanas (2.0 s, 50%)</td><td className="num">461</td><td className="num">712 380 w</td><td className="num">+712 k</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>6</td><td>Filtro NaN &gt; 5% por ventana</td><td className="num">712 380</td><td className="num">673 968</td><td className="num">-38 412</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>7</td><td>Filtro saturación / clipping</td><td className="num">673 968</td><td className="num">655 556</td><td className="num">-18 412</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>8</td><td>Filtro SNR &lt; 8 dB</td><td className="num">655 556</td><td className="num">642 746</td><td className="num">-12 810</td><td><Badge kind="ok">ok</Badge></td></tr>
                <tr><td>9</td><td>Filtro pulso ausente</td><td className="num">642 746</td><td className="num">638 142</td><td className="num">-4 604</td><td><Badge kind="ok">ok</Badge></td></tr>
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      <section className="section grid c3">
        <Card title="Cobertura de señal" sub="% válido por caso">
          <div style={{display:"flex", alignItems:"baseline", gap:8}}>
            <div style={{fontSize: 26, fontWeight: 600, color:"var(--fg-0)"}}>92.4%</div>
            <Badge kind="ok">mediana</Badge>
          </div>
          <div style={{display:"flex", gap:4, marginTop:12}}>
            {Array.from({length:40}).map((_,i)=>(
              <div key={i} style={{
                flex:1, height: 24,
                background: i<37 ? `linear-gradient(180deg, var(--teal), var(--blue-2))` : "var(--err-bg)",
                opacity: i<37 ? 0.4 + i/80 : 0.7,
                borderRadius: 2,
              }}/>
            ))}
          </div>
          <div style={{display:"flex", justifyContent:"space-between", fontFamily:"var(--mono)", fontSize:10.5, color:"var(--fg-3)", marginTop:6}}>
            <span>0% válido</span><span>100% válido</span>
          </div>
        </Card>

        <Card title="Distribución de SNR" sub="dB">
          <div style={{display:"flex", alignItems:"baseline", gap:8}}>
            <div style={{fontSize: 26, fontWeight: 600, color:"var(--fg-0)"}}>14.2 dB</div>
            <Badge kind="ok">mediana</Badge>
          </div>
          <Histogram color="var(--teal)" bins={Array.from({length:14}).map((_,i)=>(
            { label: i%4===0 ? (i+2)+"dB" : "", n: Math.round(50 + 400 * Math.exp(-Math.pow((i-7)/3,2))) }
          ))}/>
        </Card>

        <Card title="NaN por ventana" sub="post-filtros">
          <div style={{display:"flex", alignItems:"baseline", gap:8}}>
            <div style={{fontSize: 26, fontWeight: 600, color:"var(--fg-0)"}}>0.04%</div>
            <Badge kind="ok">residual</Badge>
          </div>
          <p style={{fontSize: 12, color:"var(--fg-2)", marginTop: 6}}>
            Tras el filtro de 5%, las ventanas restantes contienen NaN únicamente en
            los bordes de adquisición. Se interpolan linealmente.
          </p>
          <div style={{marginTop:10, padding:"10px 12px", border:"1px dashed var(--line-2)", borderRadius:8, fontFamily:"var(--mono)", fontSize:11, color:"var(--fg-2)"}}>
            interp = signal.interpolate(method=&quot;linear&quot;, limit=8)
          </div>
        </Card>
      </section>
    </div>
  );
};

window.DatasetPage = DatasetPage;
