/* ============================================================
   Charts: SVG simulations (placeholders bonitos)
   - ECGSignal: línea ECG realista (PQRST)
   - BarChart: barras horizontales o verticales
   - SparkBars: micro barras
   - ConfusionMatrix: heatmap
   - GridBg: rejilla papel ECG
   ============================================================ */

/* ---------- ECG generator ---------- */
/* Genera puntos de un latido tipo PQRST con parámetros */
function pqrst(t, opts = {}) {
  const { p = 0.10, q = -0.10, r = 1.0, s = -0.25, twave = 0.30 } = opts;
  // t en [0,1] = un ciclo
  const g = (mu, sd, a) => a * Math.exp(-Math.pow((t - mu) / sd, 2));
  return (
    g(0.18, 0.035, p) +
    g(0.40, 0.012, q) +
    g(0.44, 0.018, r) +
    g(0.48, 0.013, s) +
    g(0.72, 0.080, twave)
  );
}

/* Devuelve una señal ECG simulada con N latidos, devuelve string path */
function buildECGPath({
  width = 900, height = 200,
  beats = 6,
  noise = 0.02,
  amp = 1,
  shape = "nsr",          // nsr, brady, tachy, afib, pvc, vt, aflut, sb
  seed = 1,
  pvcIdx = -1
} = {}) {
  const rand = (() => { let s = seed * 9301 + 49297; return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; }; })();
  const N = width * 2; // samples
  const pts = [];
  // Decide intervalos RR según shape
  const baseRR = { nsr: 1.0, brady: 1.35, sb: 1.35, tachy: 0.62, afib: 0.7, aflut: 0.65, pvc: 1.0, vt: 0.42 }[shape] ?? 1.0;
  const rrVar = { nsr: 0.02, brady: 0.04, sb: 0.04, tachy: 0.03, afib: 0.22, aflut: 0.04, pvc: 0.05, vt: 0.04 }[shape] ?? 0.02;
  // Marca para latido central (resaltado)
  const totalT = beats * baseRR;
  const dt = totalT / N;
  let nextBeatT = 0;
  let beatI = 0;
  let curRR = baseRR;
  let cycleStart = 0;
  const beatTs = [];
  for (let i = 0; i < N; i++) {
    const T = i * dt;
    if (T >= nextBeatT) {
      beatTs.push(T);
      const jitter = (rand() - 0.5) * 2 * rrVar;
      curRR = Math.max(0.25, baseRR + jitter * baseRR);
      // Asystole-like skip would lengthen further; not used here
      cycleStart = nextBeatT;
      nextBeatT += curRR;
      beatI++;
    }
    const phase = Math.min(1, Math.max(0, (T - cycleStart) / curRR));
    let opts = {};
    if (shape === "pvc" && (beatI - 1) === pvcIdx) {
      opts = { p: 0.0, q: -0.05, r: -0.7, s: 0.4, twave: -0.4 };
    }
    if (shape === "vt") {
      // Sin onda P, R ancho, T invertida
      opts = { p: 0.0, q: -0.05, r: 0.9, s: -0.5, twave: -0.35 };
    }
    if (shape === "aflut") {
      // baseline con ondas F (sawtooth)
      opts = { p: 0.06 * Math.sin(2 * Math.PI * 4 * (T - cycleStart) / curRR) };
    }
    if (shape === "afib") {
      // Sin onda P, baseline irregular
      opts = { p: 0.03 * (rand() - 0.5) * 2 };
    }
    let y = pqrst(phase, opts) * amp;
    // noise
    y += (rand() - 0.5) * 2 * noise;
    pts.push(y);
  }
  // Normalize to SVG coords
  const yMid = height * 0.55;
  const yScale = height * 0.38;
  let d = "";
  for (let i = 0; i < pts.length; i++) {
    const x = (i / (pts.length - 1)) * width;
    const y = yMid - pts[i] * yScale;
    d += (i === 0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2) + " ";
  }
  // Centro X del latido central
  const centerBeatIdx = pvcIdx >= 0 ? pvcIdx : Math.floor(beatTs.length / 2);
  const centerT = beatTs[centerBeatIdx] ?? totalT / 2;
  const centerX = (centerT / totalT) * width;
  return { d, centerX, totalT, beatTs };
}

/* ---------- Paper grid background ---------- */
const GridBg = ({ width = 900, height = 200, color = "#1a2540" }) => (
  <g>
    <defs>
      <pattern id="grid-sm" width="10" height="10" patternUnits="userSpaceOnUse">
        <path d="M10 0 L 0 0 0 10" fill="none" stroke={color} strokeWidth="0.5" opacity=".5" />
      </pattern>
      <pattern id="grid-lg" width="50" height="50" patternUnits="userSpaceOnUse">
        <path d="M50 0 L 0 0 0 50" fill="none" stroke={color} strokeWidth="1" opacity=".8" />
      </pattern>
    </defs>
    <rect width={width} height={height} fill="url(#grid-sm)" />
    <rect width={width} height={height} fill="url(#grid-lg)" />
  </g>
);

/* ---------- ECGSignal big ---------- */
const ECGSignal = ({
  width = 980, height = 240,
  shape = "nsr", beats = 7, noise = 0.02, seed = 1,
  highlightWindow = true,   // resalta ventana centrada
  windowWidthPct = 0.20,
  showLabels = true,
  realLabel, predLabel, correct = true,
  pvcIdx = -1,
}) => {
  const { d, centerX, totalT } = buildECGPath({ width, height, beats, noise, shape, seed, pvcIdx });
  const winW = width * windowWidthPct;
  const winX = centerX - winW / 2;
  const stroke = correct ? "var(--teal)" : "var(--err)";
  return (
    <svg className="ecg-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <rect x="0" y="0" width={width} height={height} fill="#0b1426" rx="8" />
      <GridBg width={width} height={height} />
      {/* baseline */}
      <line x1="0" y1={height * 0.55} x2={width} y2={height * 0.55} stroke="#1f3157" strokeWidth="1" strokeDasharray="3 4" />
      {/* window highlight */}
      {highlightWindow && (
        <rect x={winX} y="4" width={winW} height={height - 8}
              fill="rgba(74,140,255,.08)"
              stroke="rgba(74,140,255,.4)" strokeDasharray="4 3" strokeWidth="1" rx="4" />
      )}
      {/* signal */}
      <path d={d} fill="none" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
            style={{ filter: `drop-shadow(0 0 4px ${correct ? "rgba(45,212,191,.45)" : "rgba(248,113,113,.45)"})` }} />
      {/* center beat marker */}
      <line x1={centerX} y1="4" x2={centerX} y2={height - 4} stroke="#fbbf24" strokeWidth="1" strokeDasharray="2 3" opacity=".75" />
      <circle cx={centerX} cy={height * 0.20} r="4" fill="#fbbf24" />
      {/* time axis labels */}
      <g fontFamily="IBM Plex Mono" fontSize="9" fill="#5d6c8c">
        {[0,1,2,3,4,5,6].map(i => {
          const x = (i/6) * width;
          return <text key={i} x={x+3} y={height-6}>{(i/6*totalT).toFixed(1)}s</text>;
        })}
        <text x={width-44} y={14}>mV · 25 mm/s</text>
      </g>
      {/* labels */}
      {showLabels && realLabel && (
        <g>
          <rect x={winX} y={height - 26} width="100" height="20" rx="4" fill="rgba(74,140,255,.15)" stroke="rgba(74,140,255,.4)" />
          <text x={winX + 8} y={height - 12} fontFamily="IBM Plex Mono" fontSize="11" fill="#cfe1ff">real: {realLabel}</text>
        </g>
      )}
      {showLabels && predLabel && (
        <g>
          <rect x={winX + 110} y={height - 26} width="120" height="20" rx="4"
            fill={correct ? "rgba(52,211,153,.15)" : "rgba(248,113,113,.15)"}
            stroke={correct ? "rgba(52,211,153,.4)" : "rgba(248,113,113,.4)"} />
          <text x={winX + 118} y={height - 12} fontFamily="IBM Plex Mono" fontSize="11"
            fill={correct ? "#a7f3d0" : "#fecaca"}>pred: {predLabel}</text>
        </g>
      )}
    </svg>
  );
};

/* ---------- Tiny ECG strip (for hero / cards) ---------- */
const ECGStrip = ({ width = 460, height = 88, shape = "nsr", beats = 5, color = "var(--teal)" }) => {
  const { d } = buildECGPath({ width, height, beats, noise: 0.015, shape, seed: 7 });
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} preserveAspectRatio="none">
      <GridBg width={width} height={height} color="#152038" />
      <path d={d} fill="none" stroke={color} strokeWidth="1.4" strokeLinecap="round"
            style={{ filter: "drop-shadow(0 0 4px rgba(45,212,191,.45))" }} />
    </svg>
  );
};

/* ---------- BarChart horizontal ---------- */
const BarChartH = ({ data, maxValue, color = "var(--blue)", valueFormat = v => v.toFixed(3), labelKey = "label", valueKey = "value", height = 28, accentTop = true }) => {
  const max = maxValue ?? Math.max(...data.map(d => d[valueKey]));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {data.map((d, i) => {
        const pct = (d[valueKey] / max) * 100;
        const isTop = i === 0 && accentTop;
        return (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "140px 1fr 60px", alignItems: "center", gap: 12 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--fg-1)" }}>{d[labelKey]}</div>
            <div style={{ height, background: "var(--bg-1)", border: "1px solid var(--line-1)", borderRadius: 4, position: "relative", overflow: "hidden" }}>
              <div style={{
                position: "absolute", left: 0, top: 0, bottom: 0, width: pct + "%",
                background: isTop
                  ? "linear-gradient(90deg, rgba(45,212,191,.85), rgba(74,140,255,.85))"
                  : "linear-gradient(90deg, rgba(74,140,255,.55), rgba(74,140,255,.85))",
                borderRadius: 3,
                transition: "width .6s cubic-bezier(.2,.8,.2,1)"
              }} />
              <div style={{
                position: "absolute", left: 8, top: "50%", transform: "translateY(-50%)",
                fontFamily: "var(--mono)", fontSize: 11, color: "rgba(255,255,255,.8)"
              }}>{d.note}</div>
            </div>
            <div style={{ textAlign: "right", fontFamily: "var(--mono)", fontSize: 12, color: "var(--fg-0)" }}>
              {valueFormat(d[valueKey])}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/* ---------- BarChart vertical (grouped) ---------- */
const BarChartV = ({ data, height = 220, colors = ["var(--blue)", "var(--teal)", "var(--warn)"], keys = ["a"], labels = ["A"], valueFormat = v => v.toFixed(2), yMax }) => {
  const max = yMax ?? Math.max(...data.flatMap(d => keys.map(k => d[k])));
  const W = 900, H = height, pad = { l: 36, r: 12, t: 14, b: 36 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const groupW = innerW / data.length;
  const barW = (groupW - 14) / keys.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
      {/* y grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
        const y = pad.t + (1 - p) * innerH;
        return (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={y} y2={y} stroke="#1c2740" strokeDasharray="2 3" />
            <text x={pad.l - 6} y={y + 3} textAnchor="end" fontFamily="IBM Plex Mono" fontSize="10" fill="#5d6c8c">
              {(p * max).toFixed(2)}
            </text>
          </g>
        );
      })}
      {/* bars */}
      {data.map((d, i) => {
        const gx = pad.l + i * groupW + 7;
        return (
          <g key={i}>
            {keys.map((k, j) => {
              const v = d[k];
              const h = (v / max) * innerH;
              const x = gx + j * barW;
              const y = pad.t + innerH - h;
              return (
                <g key={k}>
                  <rect x={x} y={y} width={barW - 4} height={h} fill={colors[j]} opacity=".85" rx="2">
                    <title>{d.label} · {labels[j]}: {valueFormat(v)}</title>
                  </rect>
                  <text x={x + (barW - 4)/2} y={y - 4} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="9" fill="#c8d2e5">
                    {v >= max * 0.15 ? valueFormat(v) : ""}
                  </text>
                </g>
              );
            })}
            <text x={gx + (groupW - 14) / 2} y={H - pad.b + 16} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="11" fill="#8a98b5">
              {d.label}
            </text>
          </g>
        );
      })}
      {/* legend */}
      {labels.length > 1 && (
        <g transform={`translate(${pad.l},${H - 6})`}>
          {labels.map((l, i) => (
            <g key={l} transform={`translate(${i * 90},0)`}>
              <rect width="10" height="10" y="-10" fill={colors[i]} rx="2" />
              <text x="14" y="-1" fontFamily="IBM Plex Mono" fontSize="10" fill="#8a98b5">{l}</text>
            </g>
          ))}
        </g>
      )}
    </svg>
  );
};

/* ---------- Histogram (simple) ---------- */
const Histogram = ({ bins, height = 180, color = "var(--blue)" }) => {
  const W = 900, H = height, pad = { l: 30, r: 10, t: 10, b: 28 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const max = Math.max(...bins.map(b => b.n));
  const bw = innerW / bins.length - 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
      {[0, .5, 1].map((p, i) => {
        const y = pad.t + (1 - p) * innerH;
        return <line key={i} x1={pad.l} x2={W - pad.r} y1={y} y2={y} stroke="#1c2740" strokeDasharray="2 3" />;
      })}
      {bins.map((b, i) => {
        const h = (b.n / max) * innerH;
        const x = pad.l + i * (innerW / bins.length) + 1;
        return (
          <g key={i}>
            <rect x={x} y={pad.t + innerH - h} width={bw} height={h} fill={color} opacity=".75" rx="1.5" />
            {(i % Math.ceil(bins.length / 8) === 0) &&
              <text x={x + bw/2} y={H - pad.b + 14} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="10" fill="#5d6c8c">{b.label}</text>}
          </g>
        );
      })}
    </svg>
  );
};

/* ---------- Confusion matrix ---------- */
/* dataset: 10x10 values + class ids */
function buildConfusionData(classes, normalized = false) {
  // Cells: diagonal high, plus seeded cross-cell errors
  const ids = classes.map(c => c.id);
  const N = ids.length;
  const mat = Array.from({ length: N }, () => Array(N).fill(0));
  classes.forEach((c, i) => {
    // diagonal proportional to recall * support
    mat[i][i] = Math.round(c.support * c.rec);
    let remain = c.support - mat[i][i];
    // distribute errors with a deterministic rule
    for (let j = 0; j < N && remain > 0; j++) {
      if (j === i) continue;
      const delta = Math.round(remain * (1 / (1 + Math.abs(i - j) * 0.7)) * 0.35);
      mat[i][j] += Math.min(delta, remain);
      remain -= Math.min(delta, remain);
    }
    if (remain > 0) mat[i][(i+1) % N] += remain;
  });
  if (normalized) {
    return mat.map(row => {
      const s = row.reduce((a,b) => a+b, 0) || 1;
      return row.map(v => v / s);
    });
  }
  return mat;
}

const ConfusionMatrix = ({ classes, normalized = false }) => {
  const mat = buildConfusionData(classes, normalized);
  // color scale function
  const colorFor = (val, rowMax) => {
    const t = Math.max(0, Math.min(1, val / rowMax));
    // dark blue -> blue -> teal
    const stops = [
      [14, 30, 52],
      [26, 53, 104],
      [45, 97, 201],
      [74, 140, 255],
      [45, 212, 191],
    ];
    const seg = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
    const local = t * (stops.length - 1) - seg;
    const c = stops[seg].map((v, i) => Math.round(v + (stops[seg + 1][i] - v) * local));
    return `rgb(${c.join(",")})`;
  };
  const rowMaxes = mat.map(r => Math.max(...r));

  return (
    <div className="cm-wrap">
      <table className="cm">
        <thead>
          <tr>
            <th className="corner"></th>
            <th className="corner" colSpan={classes.length} style={{ textAlign: "center", color: "var(--fg-3)" }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 10, letterSpacing: ".12em", textTransform: "uppercase" }}>Predicted →</span>
            </th>
          </tr>
          <tr>
            <th className="corner"></th>
            {classes.map(c => <th key={c.id}>{c.id}</th>)}
          </tr>
        </thead>
        <tbody>
          {classes.map((c, i) => (
            <tr key={c.id}>
              <th style={{ textAlign: "right", color: "var(--fg-2)" }}>{c.id}</th>
              {classes.map((cj, j) => {
                const v = mat[i][j];
                const display = normalized ? (v * 100).toFixed(1) + "%" : v >= 1000 ? Math.round(v/100)/10 + "k" : v;
                return (
                  <td key={cj.id} className={`cell ${i===j?"diag":""}`}
                      style={{ background: colorFor(v, rowMaxes[i]), color: (v / rowMaxes[i]) > 0.35 ? "#fff" : "rgba(255,255,255,.5)" }}>
                    {display}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

/* ---------- Probabilities mini bar ---------- */
const ProbBars = ({ probs, correct = true }) => {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {entries.map(([k, v], i) => (
        <div key={k} style={{ display: "grid", gridTemplateColumns: "70px 1fr 50px", alignItems: "center", gap: 10 }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--fg-1)" }}>{k}</div>
          <div style={{ height: 10, background: "var(--bg-1)", border: "1px solid var(--line-1)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              width: (v * 100) + "%", height: "100%",
              background: i === 0
                ? (correct ? "linear-gradient(90deg, var(--teal), var(--blue))" : "linear-gradient(90deg, var(--err), #f97316)")
                : "var(--line-3)",
            }} />
          </div>
          <div style={{ textAlign: "right", fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--fg-0)" }}>
            {(v * 100).toFixed(0)}%
          </div>
        </div>
      ))}
    </div>
  );
};

Object.assign(window, { ECGSignal, ECGStrip, BarChartH, BarChartV, Histogram, ConfusionMatrix, ProbBars });
