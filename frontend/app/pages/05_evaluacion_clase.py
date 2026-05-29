import streamlit as st

import pandas as pd

from components.layout import page_header
from components.cards import metric_card, callout, section_title, kv_table
from components.badges import badge, badge_row
from components.charts import class_f1_bar, support_vs_f1_scatter, bar_chart_h
from components.tables import class_report_table
from utils.loaders import load_classification_report, load_model_metadata

# ── bootstrap ─────────────────────────────────────────────────────────────────
meta = load_model_metadata()

page_header(
    "Evaluación por clase",
    "Desempeño del modelo ganador en cada tipo de arritmia. "
    "F1-score, soporte (ventanas de test) y análisis de clases difíciles.",
    badge_html=badge_row(badge("LinearSVC", "winner"), badge("Análisis por clase", "info")),
)

# ── load data ─────────────────────────────────────────────────────────────────
df_raw = load_classification_report()

SUMMARY_ROWS = {"accuracy", "macro avg", "weighted avg"}

if df_raw is None:
    callout(
        "warn",
        "Datos no disponibles",
        "El archivo `reports/tables/best_model_classification_report.csv` no fue encontrado. "
        "Ejecuta el pipeline para generarlo.",
    )
    st.stop()

df_classes = df_raw[~df_raw.index.isin(SUMMARY_ROWS)].copy()

for col in ["precision", "recall", "f1-score", "support"]:
    if col in df_classes.columns:
        df_classes[col] = pd.to_numeric(df_classes[col], errors="coerce")

# ── derived values ─────────────────────────────────────────────────────────────
n_classes = len(df_classes)
f1_col = "f1-score" if "f1-score" in df_classes.columns else None
sup_col = "support" if "support" in df_classes.columns else None

if f1_col:
    best_row  = df_classes[f1_col].idxmax()
    worst_row = df_classes[f1_col].idxmin()
    best_f1   = float(df_classes.loc[best_row, f1_col])
    worst_f1  = float(df_classes.loc[worst_row, f1_col])
    avg_f1    = float(df_classes[f1_col].mean())
    n_good    = int((df_classes[f1_col] >= 0.70).sum())
    n_medium  = int(((df_classes[f1_col] >= 0.30) & (df_classes[f1_col] < 0.70)).sum())
    n_hard    = int((df_classes[f1_col] < 0.30).sum())
    tier_good   = df_classes[df_classes[f1_col] >= 0.70].index.tolist()
    tier_medium = df_classes[(df_classes[f1_col] >= 0.30) & (df_classes[f1_col] < 0.70)].index.tolist()
    tier_hard   = df_classes[df_classes[f1_col] < 0.30].index.tolist()
else:
    best_row = worst_row = "—"
    best_f1 = worst_f1 = avg_f1 = 0.0
    n_good = n_medium = n_hard = 0
    tier_good = tier_medium = tier_hard = []

total_support = int(df_classes[sup_col].sum()) if sup_col else 0
min_sup_row   = df_classes[sup_col].idxmin() if sup_col else "—"
min_sup_val   = int(df_classes.loc[min_sup_row, sup_col]) if sup_col else 0

# ── callout metodológico ──────────────────────────────────────────────────────
callout(
    "info",
    "Nota metodológica",
    "El reporte usa **test set independiente** (97 grupos de pacientes). "
    "F1-score macro = promedio no ponderado entre clases — "
    "clases con muy poco soporte impactan desproporcionadamente la métrica global.",
)

st.write("")

# ── KPI row ───────────────────────────────────────────────────────────────────
section_title("Resumen de clases")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Clases evaluadas", str(n_classes), "tipos de arritmia", accent="blue")
with c2:
    metric_card("Mejor clase", f"{best_f1:.3f}", best_row, accent="teal", helper_kind="ok")
with c3:
    metric_card("Peor clase", f"{worst_f1:.3f}", worst_row, accent="err", helper_kind="warn")
with c4:
    metric_card("F1 promedio", f"{avg_f1:.3f}", "media aritmética entre clases", accent="blue")
with c5:
    metric_card("Ventanas de test", f"{total_support:,}", "total en todas las clases", accent="muted")
with c6:
    metric_card("Clase mas pequeña", f"{min_sup_val:,}", min_sup_row, accent="err", helper_kind="warn")

st.write("")

# ── tier cards ────────────────────────────────────────────────────────────────
section_title("Clasificación por tier de desempeño")

col_good, col_med, col_hard = st.columns(3)

with col_good:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:8px">{badge(f"F1 >= 0.70 · {n_good} clases", "ok")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Bien clasificadas**")
        if tier_good:
            for cls in tier_good:
                f1v  = float(df_classes.loc[cls, f1_col]) if f1_col else 0.0
                supv = int(df_classes.loc[cls, sup_col])  if sup_col else 0
                st.markdown(
                    f"· **{cls}** — {badge(f'F1 {f1v:.3f}', 'ok')} "
                    f"<span style='color:#8b9cbd;font-size:0.85em'>{supv:,} ventanas</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Ninguna clase alcanza este umbral.")

with col_med:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:8px">{badge(f"0.30 <= F1 < 0.70 · {n_medium} clases", "info")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Rendimiento moderado**")
        if tier_medium:
            for cls in tier_medium:
                f1v  = float(df_classes.loc[cls, f1_col]) if f1_col else 0.0
                supv = int(df_classes.loc[cls, sup_col])  if sup_col else 0
                st.markdown(
                    f"· **{cls}** — {badge(f'F1 {f1v:.3f}', 'info')} "
                    f"<span style='color:#8b9cbd;font-size:0.85em'>{supv:,} ventanas</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Ninguna clase en este rango.")

with col_hard:
    with st.container(border=True):
        st.markdown(
            f'<div style="margin-bottom:8px">{badge(f"F1 < 0.30 · {n_hard} clases", "err")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Clases difíciles**")
        if tier_hard:
            for cls in tier_hard:
                f1v  = float(df_classes.loc[cls, f1_col]) if f1_col else 0.0
                supv = int(df_classes.loc[cls, sup_col])  if sup_col else 0
                kind = "err" if supv < 500 else "warn"
                st.markdown(
                    f"· **{cls}** — {badge(f'F1 {f1v:.3f}', kind)} "
                    f"<span style='color:#8b9cbd;font-size:0.85em'>{supv:,} ventanas</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Ninguna clase en este rango.")

st.write("")

# ── charts ────────────────────────────────────────────────────────────────────
section_title("Visualizaciones")

labels     = df_classes.index.tolist()
f1_values  = df_classes[f1_col].tolist() if f1_col else []
sup_values = [float(v) for v in df_classes[sup_col].tolist()] if sup_col else []

chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    with st.container(border=True):
        st.markdown("**F1-score por clase**")
        if f1_values:
            st.plotly_chart(class_f1_bar(labels, f1_values), use_container_width=True)
        else:
            st.caption("Sin datos de F1.")

with chart_col2:
    with st.container(border=True):
        st.markdown("**Soporte por clase** (ventanas de test, escala log)")
        if sup_values:
            st.plotly_chart(
                bar_chart_h(
                    labels=labels,
                    values=sup_values,
                    title="",
                    color="#4a8cff",
                    accent_top=False,
                    value_fmt=".0f",
                    height=340,
                    log_x=True,
                ),
                use_container_width=True,
            )
        else:
            st.caption("Sin datos de soporte.")

st.write("")

with st.container(border=True):
    st.markdown("**Soporte vs F1-score** — relación entre tamaño de clase y rendimiento")
    if f1_values and sup_values:
        st.plotly_chart(
            support_vs_f1_scatter(labels, sup_values, f1_values),
            use_container_width=True,
        )
    else:
        st.caption("Sin datos suficientes para el scatter.")

st.write("")

# ── sortable table ─────────────────────────────────────────────────────────────
section_title("Tabla detallada por clase")

col_sort, col_filter, _ = st.columns([2, 2, 4])
with col_sort:
    sort_options = {
        "F1-score": "f1-score",
        "Precision": "precision",
        "Recall": "recall",
        "Soporte": "support",
    }
    sort_label = st.selectbox("Ordenar por", list(sort_options.keys()), index=0)
    sort_col_key = sort_options[sort_label]

with col_filter:
    show_hard_only = st.checkbox("Solo clases difíciles (F1 < 0.30)", value=False)

df_table = df_classes.copy()
if show_hard_only and f1_col:
    df_table = df_table[df_table[f1_col] < 0.30]

class_report_table(df_table, sort_col=sort_col_key, ascending=(sort_label == "Soporte"))

st.write("")

# ── aggregated metrics expander ────────────────────────────────────────────────
with st.expander("Metricas agregadas (macro / weighted avg)", expanded=False):
    summary_rows_df = df_raw[df_raw.index.isin(SUMMARY_ROWS)]
    if not summary_rows_df.empty:
        rows_kv = []
        for idx in summary_rows_df.index:
            row  = summary_rows_df.loc[idx]
            f1v  = row.get("f1-score", float("nan"))
            prec = row.get("precision", float("nan"))
            rec  = row.get("recall", float("nan"))
            sup  = row.get("support", float("nan"))
            f1v  = float(f1v)  if pd.notna(f1v)  else None
            prec = float(prec) if pd.notna(prec) else None
            rec  = float(rec)  if pd.notna(rec)  else None
            parts = []
            if f1v  is not None: parts.append(f"F1={f1v:.3f}")
            if prec is not None: parts.append(f"Prec={prec:.3f}")
            if rec  is not None: parts.append(f"Rec={rec:.3f}")
            if pd.notna(sup):    parts.append(f"N={int(float(sup)):,}")
            rows_kv.append((idx.title(), "  ·  ".join(parts)))
        kv_table(rows_kv)
    else:
        st.caption("No se encontraron filas de resumen en el reporte.")

st.write("")

# ── interpretation callout ────────────────────────────────────────────────────
if tier_good and sup_col:
    best_cls_sup = int(df_classes.loc[tier_good[0], sup_col])
    interp_body = (
        f"El modelo LinearSVC clasifica bien las dos clases más frecuentes "
        f"(**{tier_good[0]}** F1={best_f1:.3f}, {best_cls_sup:,} ventanas), "
        f"pero tiene dificultades con clases poco representadas. "
        f"Las **{n_hard} clases difíciles** (F1 < 0.30) incluyen algunas con muy pocas ventanas de test "
        f"(p.ej. *{min_sup_row}*: {min_sup_val} ventanas), lo que limita la capacidad del modelo. "
        f"El F1-macro de **{avg_f1:.3f}** refleja este desequilibrio de clases. "
        f"Estrategias como SMOTE o un clasificador 1D-CNN podrían mejorar las clases difíciles."
    )
else:
    interp_body = (
        f"Se evaluaron {n_classes} clases. F1-macro promedio = {avg_f1:.3f}. "
        f"El modelo tiene {n_good} clases bien clasificadas, {n_medium} moderadas y {n_hard} difíciles."
    )

callout("info", "Interpretacion", interp_body)
