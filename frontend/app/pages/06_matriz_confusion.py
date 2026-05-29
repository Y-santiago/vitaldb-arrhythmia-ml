import streamlit as st

import pandas as pd

from components.layout import page_header
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from components.charts import class_f1_bar
from components.tables import class_report_table
from utils.loaders import (
    load_classification_report,
    load_confusion_matrix_csv,
    load_model_metadata,
    confusion_matrix_figure_path,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
meta = load_model_metadata()
winner = meta.get("winner_model", "—").replace("_", " ").title() if meta else "—"
f1_str = f"{meta.get('winner_test_f1_macro', 0):.3f}" if meta else "—"
page_header(
    "Matriz de confusión",
    "Análisis visual de aciertos y errores del modelo ganador entre clases de ritmo.",
    badge_html=badge_row(badge("LinearSVC", "winner"), badge("Errores del modelo", "info")),
)

# ── Load data ─────────────────────────────────────────────────────────────────
df_cls_raw   = load_classification_report()
df_cm_csv    = load_confusion_matrix_csv()
cm_png_path  = confusion_matrix_figure_path()

SUMMARY_ROWS = {"accuracy", "macro avg", "weighted avg"}

df_classes = None
if df_cls_raw is not None:
    df_classes = df_cls_raw[~df_cls_raw.index.isin(SUMMARY_ROWS)].copy()
    for col in ["precision", "recall", "f1-score", "support"]:
        if col in df_classes.columns:
            df_classes[col] = pd.to_numeric(df_classes[col], errors="coerce")

has_csv = df_cm_csv is not None
has_png = cm_png_path is not None
has_cls = df_classes is not None and not df_classes.empty

# ── Callout metodológico ──────────────────────────────────────────────────────
callout(
    "info",
    "Cómo leer una matriz de confusión",
    "<b>Filas</b> = clase real &nbsp;·&nbsp; "
    "<b>Columnas</b> = clase predicha &nbsp;·&nbsp; "
    "<b>Diagonal</b> = predicciones correctas (aciertos) &nbsp;·&nbsp; "
    "<b>Fuera de diagonal</b> = confusiones (errores). "
    "En datasets desbalanceados, la <i>accuracy</i> puede parecer alta aunque el modelo "
    "falle completamente en clases minoritarias — la matriz revela estos errores ocultos. "
    "El <b>recall</b> de cada clase = aciertos en esa clase / total real de esa clase = "
    "valor en la diagonal dividido por la suma de la fila.",
)

st.write("")

# ── KPIs ─────────────────────────────────────────────────────────────────────
section_title("Estado de los datos")

n_classes    = len(df_classes) if has_cls else 0
total_sup    = int(df_classes["support"].sum()) if has_cls and "support" in df_classes.columns else 0
best_rec_cls = df_classes["recall"].idxmax() if has_cls and "recall" in df_classes.columns else "—"
worst_rec_cls= df_classes["recall"].idxmin() if has_cls and "recall" in df_classes.columns else "—"
best_rec_val = float(df_classes.loc[best_rec_cls, "recall"]) if has_cls and best_rec_cls != "—" else 0.0
worst_rec_val= float(df_classes.loc[worst_rec_cls, "recall"]) if has_cls and worst_rec_cls != "—" else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Clases evaluadas", str(n_classes), "tipos de ritmo", accent="blue")
with c2:
    metric_card("Test windows", f"{total_sup:,}", "ventanas evaluadas", accent="blue")
with c3:
    metric_card(
        "Mejor recall",
        f"{best_rec_val:.3f}",
        best_rec_cls,
        accent="teal",
        helper_kind="ok",
    )
with c4:
    metric_card(
        "Peor recall",
        f"{worst_rec_val:.3f}",
        worst_rec_cls,
        accent="err",
        helper_kind="warn",
    )
with c5:
    if has_png:
        metric_card("Matriz PNG", "disponible", "reports/figures/", accent="ok", helper_kind="ok")
    else:
        metric_card("Matriz PNG", "no encontrada", "reports/figures/", accent="err", helper_kind="warn")
with c6:
    if has_csv:
        metric_card("Matriz CSV", "disponible", "heatmap interactivo", accent="teal", helper_kind="ok")
    else:
        metric_card("Matriz CSV", "pendiente", "reports/tables/", accent="warn", helper_kind="warn")

st.write("")

# ── Main visualization ────────────────────────────────────────────────────────
section_title("Visualización de la matriz")

if has_csv:
    # ── CASE A: Interactive Plotly heatmap ────────────────────────────────────
    import numpy as np
    import plotly.graph_objects as go

    # Build square matrix from long-format CSV
    required_cols = {"real_label", "predicted_label", "count"}
    has_required  = required_cols.issubset(set(df_cm_csv.columns))

    if has_required:
        labels_all = sorted(
            set(df_cm_csv["real_label"]).union(set(df_cm_csv["predicted_label"]))
        )
        n = len(labels_all)
        idx_map = {lbl: i for i, lbl in enumerate(labels_all)}
        matrix = np.zeros((n, n), dtype=float)
        for _, row in df_cm_csv.iterrows():
            r = idx_map.get(str(row["real_label"]))
            c = idx_map.get(str(row["predicted_label"]))
            if r is not None and c is not None:
                matrix[r, c] = float(row["count"])

        col_view, _ = st.columns([3, 1])
        with col_view:
            view = st.selectbox(
                "Vista",
                ["Conteos absolutos", "Normalizado por fila (recall %)"],
                index=0,
            )

        display_matrix = matrix.copy()
        fmt_str        = ".0f"
        if view == "Normalizado por fila (recall %)":
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            display_matrix = matrix / row_sums
            fmt_str = ".2%"

        text_matrix = [[f"{v:{fmt_str}}" for v in row] for row in display_matrix]

        fig = go.Figure(go.Heatmap(
            z=display_matrix.tolist(),
            x=labels_all,
            y=labels_all,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=9, family="IBM Plex Mono, monospace"),
            colorscale="Teal",
            colorbar=dict(
                title="",
                tickfont=dict(size=9, family="IBM Plex Mono, monospace"),
            ),
            hovertemplate=(
                "Real: <b>%{y}</b><br>"
                "Predicho: <b>%{x}</b><br>"
                "Valor: %{z}<extra></extra>"
            ),
        ))
        fig.update_layout(
            plot_bgcolor="#121a2b",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono, monospace", color="#8a98b5", size=10),
            height=560,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(
                title="Clase predicha",
                tickfont=dict(size=9),
                side="bottom",
            ),
            yaxis=dict(
                title="Clase real",
                tickfont=dict(size=9),
                autorange="reversed",
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        callout(
            "warn",
            "Formato CSV inesperado",
            f"Se esperan columnas <code>real_label</code>, <code>predicted_label</code>, "
            f"<code>count</code>. Columnas encontradas: {list(df_cm_csv.columns)}",
        )
        if has_png:
            st.image(str(cm_png_path), use_container_width=True)

else:
    # ── CASE B: PNG fallback ──────────────────────────────────────────────────
    if has_png:
        with st.container(border=True):
            card_header(
                "Matriz de confusión",
                "imagen exportada por el pipeline",
                right_html=badge("PNG", "ok"),
            )
            st.image(str(cm_png_path), use_container_width=True)

        callout(
            "info",
            "Heatmap interactivo no disponible",
            "Por ahora se muestra la matriz exportada como imagen. "
            "Para habilitar filtros interactivos y normalización por fila, "
            "exporta <code>reports/tables/confusion_matrix.csv</code> desde el notebook de evaluación "
            "(ver sección de recomendación técnica al final de esta página).",
        )
    else:
        callout(
            "warn",
            "Ninguna visualización disponible",
            "No se encontró ni el PNG ni el CSV de la matriz de confusión. "
            "Ejecuta el pipeline de evaluación para generarlos.",
        )

st.write("")

# ── Error table (only if CSV) ─────────────────────────────────────────────────
section_title("Errores principales fuera de diagonal")

if has_csv and "real_label" in df_cm_csv.columns and "predicted_label" in df_cm_csv.columns:
    df_errors = df_cm_csv[
        df_cm_csv["real_label"] != df_cm_csv["predicted_label"]
    ].copy()
    if not df_errors.empty:
        df_errors = df_errors.sort_values("count", ascending=False).head(15)
        if has_cls and "support" in df_classes.columns:
            def _pct(row):
                cls = row["real_label"]
                if cls in df_classes.index:
                    sup = df_classes.loc[cls, "support"]
                    if pd.notna(sup) and sup > 0:
                        return f"{row['count'] / sup:.1%}"
                return "—"
            df_errors["% de la clase"] = df_errors.apply(_pct, axis=1)

        df_errors = df_errors.rename(columns={
            "real_label":      "Clase real",
            "predicted_label": "Clase predicha",
            "count":           "Errores",
        })
        st.dataframe(df_errors, use_container_width=True, hide_index=True)
    else:
        st.info("No se encontraron errores fuera de diagonal en el CSV.")
else:
    st.html(
        '<div class="placeholder-block" style="min-height:120px;padding:24px">'
        '<div class="ph-mono">tabla pendiente</div>'
        '<div class="ph-title">Top errores de confusión</div>'
        '<div class="ph-desc">Disponible cuando se exporte <code>reports/tables/confusion_matrix.csv</code>.</div>'
        '</div>'
    )

st.write("")

# ── Per-class recall / interpretation ────────────────────────────────────────
if has_cls:
    section_title("Interpretación por clase — Recall")

    st.write(
        "El **recall** de una clase indica qué fracción de los ejemplos reales fueron "
        "identificados correctamente. Recall bajo = el modelo frecuentemente predice "
        "otra clase cuando el ritmo real era este."
    )

    col_chart, col_tbl = st.columns([1.4, 1])

    with col_chart:
        with st.container(border=True):
            st.markdown("**Recall por clase** (ordenado ascendente)")
            labels   = df_classes.index.tolist()
            recall_v = df_classes["recall"].tolist() if "recall" in df_classes.columns else []
            if recall_v:
                st.plotly_chart(
                    class_f1_bar(labels, recall_v, thresholds=(0.70, 0.30), height=340),
                    use_container_width=True,
                )

    with col_tbl:
        with st.container(border=True):
            st.markdown("**Clases ordenadas por recall (peor primero)**")
            if "recall" in df_classes.columns:
                class_report_table(df_classes, sort_col="recall", ascending=True)
            else:
                st.caption("Columna recall no encontrada.")

    st.write("")

    # ── Most confused (low recall) analysis ──────────────────────────────────
    if "recall" in df_classes.columns:
        df_hard = df_classes[df_classes["recall"] < 0.30].sort_values("recall")
        if not df_hard.empty:
            section_title("Clases con recall < 0.30 — alta tasa de confusión")
            with st.container(border=True):
                card_header(
                    "Clases difíciles de detectar",
                    f"{len(df_hard)} clases con recall < 0.30",
                    right_html=badge(f"{len(df_hard)} clases", "err"),
                )
                rows_kv = []
                for cls_name in df_hard.index:
                    rec  = float(df_hard.loc[cls_name, "recall"])
                    sup  = int(df_hard.loc[cls_name, "support"]) if "support" in df_hard.columns else 0
                    prec = float(df_hard.loc[cls_name, "precision"]) if "precision" in df_hard.columns else 0.0
                    rec_badge  = badge(f"recall {rec:.3f}", "err" if rec < 0.10 else "warn")
                    prec_badge = badge(f"prec {prec:.3f}", "muted")
                    sup_badge  = badge(f"{sup:,} ventanas", "muted")
                    rows_kv.append((
                        cls_name,
                        f"{rec_badge} {prec_badge} {sup_badge}",
                    ))
                kv_table(rows_kv)

                callout(
                    "warn",
                    "Por qué estas clases son difíciles",
                    "Las clases con recall bajo son frecuentemente <b>predichas como la clase dominante</b> "
                    "(N o AFIB/AFL). Causas probables: "
                    "(1) muy pocas ventanas de entrenamiento, "
                    "(2) features discriminativas insuficientes para este ritmo, "
                    "(3) solapamiento en el espacio de features con clases más frecuentes. "
                    "Posibles mejoras: SMOTE, pesos de clase, modelo 1D-CNN con "
                    "representaciones aprendidas.",
                )

st.write("")

# ── Technical recommendation ──────────────────────────────────────────────────
section_title("Recomendación técnica")

with st.container(border=True):
    card_header("Para habilitar el heatmap interactivo", "exportar CSV desde pipeline")
    kv_table([
        ("Archivo objetivo",  "reports/tables/confusion_matrix.csv"),
        ("Formato esperado",  "real_label, predicted_label, count"),
        ("Cómo generarlo",    "Desde el notebook de evaluación, después de predict() en test set"),
        ("Código de ejemplo", ""),
    ])
    st.code(
        "from sklearn.metrics import confusion_matrix\n"
        "import itertools, pandas as pd\n\n"
        "cm = confusion_matrix(y_true, y_pred, labels=class_names)\n"
        "rows = [(r, c, cm[i, j])\n"
        "        for (i, r), (j, c) in\n"
        "        itertools.product(enumerate(class_names), repeat=2)]\n"
        "pd.DataFrame(rows, columns=['real_label','predicted_label','count'])\\\n"
        "  .to_csv('reports/tables/confusion_matrix.csv', index=False)",
        language="python",
    )
    callout(
        "info",
        "Sin regresiones",
        "Este cambio solo requiere agregar 5 líneas al notebook de evaluación y no "
        "modifica ningún modelo entrenado ni los resultados existentes.",
    )
