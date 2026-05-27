/* ============================================================
   App principal — sidebar + router
   ============================================================ */

const NAV = [
  { id: "home",        n: 1, label: "Inicio",                Page: HomePage },
  { id: "pipeline",    n: 2, label: "Pipeline del proyecto", Page: PipelinePage },
  { id: "dataset",     n: 3, label: "Dataset y limpieza",    Page: DatasetPage },
  { id: "performance", n: 4, label: "Rendimiento del modelo",Page: PerformancePage },
  { id: "classes",     n: 5, label: "Evaluación por clase",  Page: ClassesPage },
  { id: "confusion",   n: 6, label: "Matriz de confusión",   Page: ConfusionPage },
  { id: "predict",     n: 7, label: "Jugar con predicciones",Page: PredictPage },
  { id: "interpret",   n: 8, label: "Interpretabilidad",     Page: InterpretPage },
  { id: "conclusions", n: 9, label: "Conclusiones",          Page: ConclusionsPage },
];

const Sidebar = ({ active, onSelect }) => (
  <aside className="sidebar">
    <div className="sb-brand">
      <div className="sb-logo" aria-hidden>
        {/* Heart + ECG glyph */}
        <svg viewBox="0 0 24 24" fill="none" stroke="#2dd4bf" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 12 H6 L8 8 L11 16 L14 6 L16 12 H22" />
        </svg>
      </div>
      <div>
        <div className="sb-title">ECG Arrhythmia ML</div>
        <div className="sb-sub">Clasificación multiclase de ritmos intraoperatorios</div>
      </div>
    </div>

    <div className="sb-section">Secciones</div>
    <nav className="sb-nav">
      {NAV.map(item => (
        <div key={item.id}
             className={`sb-item ${active === item.id ? "active" : ""}`}
             onClick={() => onSelect(item.id)}>
          <span className="num">{String(item.n).padStart(2, "0")}</span>
          <span>{item.label}</span>
          <span className="dot" />
        </div>
      ))}
    </nav>

    <div className="sb-section">Estado</div>
    <div style={{padding:"4px 18px 14px", display:"flex", flexDirection:"column", gap:8}}>
      <div style={{display:"flex", justifyContent:"space-between", fontSize:11.5, color:"var(--fg-2)"}}>
        <span style={{fontFamily:"var(--mono)"}}>pipeline</span>
        <span style={{color:"var(--ok)"}}>● ok</span>
      </div>
      <div style={{display:"flex", justifyContent:"space-between", fontSize:11.5, color:"var(--fg-2)"}}>
        <span style={{fontFamily:"var(--mono)"}}>best F1-macro</span>
        <span style={{color:"var(--fg-0)", fontFamily:"var(--mono)"}}>0.742</span>
      </div>
      <div style={{display:"flex", justifyContent:"space-between", fontSize:11.5, color:"var(--fg-2)"}}>
        <span style={{fontFamily:"var(--mono)"}}>best model</span>
        <span style={{color:"var(--fg-0)", fontFamily:"var(--mono)"}}>LinearSVC</span>
      </div>
    </div>

    <div className="sb-foot">
      <div style={{marginBottom:8}}>
        <span className="stamp">DEMO ACADÉMICA</span>
      </div>
      <div>
        Datos: <span style={{color:"var(--fg-1)"}}>VitalDB Arrhythmia DB</span><br/>
        Pipeline v1.0.3 · Reporte generado el 18 may 2026
      </div>
    </div>
  </aside>
);

const App = () => {
  // url-hash sync (#section) so refresh restores the page
  const initial = () => {
    const h = (location.hash || "").replace("#", "");
    return NAV.find(n => n.id === h) ? h : "home";
  };
  const [active, setActive] = React.useState(initial);
  React.useEffect(() => { location.hash = active; }, [active]);
  React.useEffect(() => {
    const onHash = () => setActive(initial());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const current = NAV.find(n => n.id === active) || NAV[0];
  const PageComp = current.Page;

  return (
    <div className="app">
      <Sidebar active={active} onSelect={setActive} />
      <main className="main">
        <div className="topbar">
          <div className="crumbs">
            ECG Arrhythmia ML
            <span className="sep">/</span>
            <span className="here">{current.label}</span>
          </div>
          <div className="topbar-actions">
            <span className="pill"><span className="led"/> live · demo</span>
            <span className="pill">v1.0.3</span>
            <span className="pill" style={{background:"var(--warn-bg)", color:"var(--warn)", borderColor:"rgba(251,191,36,.25)"}}>
              NO clinical use
            </span>
          </div>
        </div>
        <PageComp />
      </main>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
