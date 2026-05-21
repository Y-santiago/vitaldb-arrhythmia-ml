/* Componentes UI reutilizables: Card, Metric, Badge, Callout, Table, Seg, PageHead */

const Card = ({ title, sub, right, children, className = "", padLg }) => (
  <section className={`card ${padLg ? "pad-lg" : ""} ${className}`}>
    {(title || right) && (
      <header className="card-title">
        <h3>{title}</h3>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {sub && <span className="sub">{sub}</span>}
          {right}
        </div>
      </header>
    )}
    {children}
  </section>
);

const Metric = ({ label, value, unit, delta, accent }) => (
  <div className={`card metric ${accent ? "accent-" + accent : ""}`}>
    <div className="lbl">{label}</div>
    <div className="val">
      {value}{unit && <span className="unit">{unit}</span>}
    </div>
    {delta && <div className={`delta ${delta.startsWith("−") || delta.startsWith("-") ? "neg" : ""}`}>{delta}</div>}
  </div>
);

const Badge = ({ kind = "muted", children }) => (
  <span className={`badge ${kind}`}>{children}</span>
);

const Callout = ({ kind = "info", title, children }) => (
  <div className={`callout ${kind !== "info" ? kind : ""}`}>
    {title && <span className="ttl">{title}</span>}
    {children}
  </div>
);

const Seg = ({ options, value, onChange }) => (
  <div className="seg">
    {options.map(o => (
      <button key={o.value} className={value === o.value ? "on" : ""} onClick={() => onChange(o.value)}>
        {o.label}
      </button>
    ))}
  </div>
);

const PageHead = ({ title, lead, right }) => (
  <header className="page-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 24 }}>
    <div>
      <h1>{title}</h1>
      {lead && <p>{lead}</p>}
    </div>
    {right && <div style={{ flex: "none" }}>{right}</div>}
  </header>
);

const KV = ({ rows }) => (
  <dl className="kv">
    {rows.map((r, i) => (
      <React.Fragment key={i}>
        <dt>{r[0]}</dt>
        <dd>{r[1]}</dd>
      </React.Fragment>
    ))}
  </dl>
);

const fmtInt = n => n.toLocaleString("en-US").replace(/,/g, " ");
const fmtPct = n => (n * 100).toFixed(1) + "%";

Object.assign(window, { Card, Metric, Badge, Callout, Seg, PageHead, KV, fmtInt, fmtPct });
