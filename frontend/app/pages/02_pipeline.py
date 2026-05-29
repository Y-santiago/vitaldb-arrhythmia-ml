"""Página 02 — Pipeline metodológico."""

import streamlit as st

from components.badges import badge, badge_row
from components.cards import callout, section_title
from components.charts import mini_ecg_placeholder
from components.layout import page_header


# ── helpers ───────────────────────────────────────────────────────────────────

def _badge(text: str, color: str) -> str:
    return f'<span class="pm-badge pm-{color}">{text}</span>'


def _stage(num, title, badge_text, badge_color, desc, status="done",
           substeps=None, params=None, models=None, metrics=None,
           inputs=None, outputs=None, last=False):
    line = "" if last else '<div class="pm-line"></div>'

    substeps_html = ""
    if substeps:
        parts = []
        for i, s in enumerate(substeps):
            parts.append(f'<span class="pm-substep">{s}</span>')
            if i < len(substeps) - 1:
                parts.append('<span class="pm-arrow">&rarr;</span>')
        substeps_html = f'<div class="pm-substeps">{"".join(parts)}</div>'

    params_html = ""
    if params:
        items = "".join(f'<span class="pm-param">{p}</span>' for p in params)
        params_html = f'<div class="pm-params">{items}</div>'

    models_html = ""
    if models:
        items = "".join(
            f'<span class="pm-model{"  pm-win" if m.startswith("★") else ""}">'
            f'{m.lstrip("★ ")}</span>'
            for m in models
        )
        models_html = f'<div class="pm-models">{items}</div>'

    metrics_html = ""
    if metrics:
        chips = "".join(
            f'<div class="pm-mchip">'
            f'<div class="pm-mk">{k}</div>'
            f'<div class="pm-mv pm-{c}">{v}</div>'
            f'</div>'
            for k, v, c in metrics
        )
        metrics_html = f'<div class="pm-mrow">{chips}</div>'

    io_html = ""
    if inputs or outputs:
        in_html = ""
        if inputs:
            lis = "".join(f"<li>{x}</li>" for x in inputs)
            in_html = (
                f'<div class="pm-iobox pm-in">'
                f'<div class="pm-ioh"><span class="pm-dot"></span>Input</div>'
                f'<ul>{lis}</ul></div>'
            )
        out_html = ""
        if outputs:
            lis = "".join(f"<li>{x}</li>" for x in outputs)
            out_html = (
                f'<div class="pm-iobox pm-out">'
                f'<div class="pm-ioh"><span class="pm-dot"></span>Output</div>'
                f'<ul>{lis}</ul></div>'
            )
        io_html = f'<div class="pm-io">{in_html}{out_html}</div>'

    extras = substeps_html + params_html + models_html + metrics_html + io_html

    return (
        f'<div class="pm-stage pm-{status}">'
        f'<div class="pm-spine"><div class="pm-num">{num}</div>{line}</div>'
        f'<div class="pm-body"><div class="pm-card">'
        f'<div class="pm-top"><h4>{title}</h4>{_badge(badge_text, badge_color)}</div>'
        f'<p class="pm-desc">{desc}</p>'
        f'{extras}'
        f'</div></div></div>'
    )


def _node(label, sub="", start=False, end=False):
    extra = " pm-node-start" if start else (" pm-node-end" if end else "")
    sub_html = f"<small>{sub}</small>" if sub else ""
    return f'<div class="pm-node{extra}"><span class="pm-nname">{label}{sub_html}</span></div>'


def _conn():
    return '<div class="pm-conn"></div>'


# ── 1 · Header ────────────────────────────────────────────────────────────────

page_header(
    "Pipeline metodológico",
    "De señales ECG intraoperatorias a evaluación multiclase de ritmos cardíacos.",
    badge_html=badge("Metodología", "info"),
)

st.markdown(
    badge_row(
        badge("VitalDB", "info"),
        badge("ECG", "info"),
        badge("Machine Learning", "info"),
        badge("Demo académica · No uso clínico", "warn"),
    ),
    unsafe_allow_html=True,
)

st.plotly_chart(
    mini_ecg_placeholder(height=70, n_beats=12),
    use_container_width=True,
    config={"displayModeBar": False},
)

# ── 2 · Flujo metodológico ────────────────────────────────────────────────────

section_title("Flujo metodológico end-to-end")

st.markdown(
    badge_row(
        badge("9 completadas", "ok"),
        badge("1 en curso", "info"),
        badge("1 pendiente", "muted"),
    ),
    unsafe_allow_html=True,
)

stages = [
    _stage(1, "Análisis del problema", "Problem definition", "blue",
           "Se define el reto de clasificar ritmos cardíacos intraoperatorios a partir de señales ECG, "
           "considerando ruido, artefactos y desbalance de clases.",
           inputs=["Contexto clínico", "ECG intraoperatorio", "Etiquetas de ritmo"],
           outputs=["Objetivo de clasificación multiclase"]),
    _stage(2, "Exploración de datos", "EDA", "blue",
           "Se revisan metadatos, anotaciones por latido, distribución de clases, "
           "calidad de señal y presencia de datos faltantes.",
           inputs=["metadata.csv", "Archivos de anotaciones", "Señales ECG"],
           outputs=["Conteos por clase", "Revisión de NaN", "Revisión de calidad", "Diagnóstico de desbalance"]),
    _stage(3, "Preprocesamiento ECG", "Signal processing", "teal",
           "Las señales ECG crudas se transforman para hacerlas compatibles con el flujo del modelo.",
           substeps=["Interpolación de NaN", "Remuestreo a 500 Hz", "Filtro pasa banda 0.5–40 Hz", "Normalización"],
           params=["TARGET_FS = <b>500 Hz</b>", "LOWCUT_HZ = <b>0.5</b>", "HIGHCUT_HZ = <b>40.0</b>"],
           inputs=["ECG crudo"], outputs=["ECG procesado"]),
    _stage(4, "Limpieza y preparación del dataset", "Data preparation", "blue",
           "Se construye un dataset tabular final, definiendo variables predictoras, "
           "variable objetivo y columnas prohibidas para evitar leakage.",
           params=["target = <b>rhythm_label</b>", "group = <b>case_id</b>",
                   "beat_type = <b style='color:var(--err)'>descriptivo · no predictor</b>"],
           outputs=["Dataset tabular para modelado"]),
    _stage(5, "Feature engineering", "Features", "blue",
           "Se extraen características numéricas asociadas a ritmo, intervalos RR, amplitud, energía y morfología.",
           models=["case_rr_std", "case_rr_rmssd", "rr_prev", "rr_mean_local", "std", "var", "energy"],
           outputs=["Vector de features"]),
    _stage(6, "Split train/test por case_id", "Leakage control", "amber",
           "Los datos se dividen en entrenamiento y prueba separando por caso/paciente para reducir leakage.",
           params=["split = <b>80 / 20</b>", "separación por <b>case_id</b>",
                   "sin mezclar ventanas del mismo caso"]),
    _stage(7, "Modelado", "Model training", "blue",
           "Se entrenan y comparan modelos supervisados de clasificación multiclase.",
           models=["Logistic Regression", "SVM / LinearSVC", "Decision Tree", "Random Forest", "XGBoost", "MLP"]),
    _stage(8, "Búsqueda de hiperparámetros", "Tuning", "blue",
           "Se aplica búsqueda de hiperparámetros para mejorar la generalización y reducir sobreajuste.",
           params=["método = <b>RandomizedSearchCV</b>"],
           outputs=["Mejores configuraciones por modelo"]),
    _stage(9, "Evaluación y selección del modelo", "Evaluation", "teal",
           "Los modelos se comparan usando métricas globales y por clase.",
           status="active",
           models=["Precision", "Recall", "F1-score", "F1-macro", "Accuracy", "Matriz de confusión"],
           metrics=[("Mejor modelo", "LinearSVC", "green"), ("F1-macro", "≈ 0.3439", ""), ("Accuracy", "≈ 0.8061", "blue")]),
    _stage(10, "Interpretabilidad", "Interpretability", "blue",
           "Se analizan las features más importantes para entender qué información usa el modelo.",
           status="todo",
           models=["case_rr_std", "case_rr_rmssd", "std", "var", "energy"]),
    _stage(11, "Despliegue en Streamlit", "Streamlit app", "green",
           "La app permite explorar el proyecto, revisar resultados, visualizar señales ECG y probar el preprocesamiento con casos demo.",
           status="todo",
           params=["estado = <b>demo académica</b>",
                   "<b style='color:var(--warn)'>no uso clínico</b>",
                   "predicción final pendiente de conexión completa"],
           last=True),
]

st.html(f'<div class="pm-flow">{"".join(stages)}</div>')

# ── 3 · Dos pipelines conectados ──────────────────────────────────────────────

section_title("Dos niveles: procesamiento de señal y aprendizaje")

pipeline_a_nodes = "".join([
    _node("ECG crudo", "raw signal · VitalDB", start=True), _conn(),
    _node("Interpolación de NaN"), _conn(),
    _node("Resampling", "TARGET_FS = 500 Hz"), _conn(),
    _node("Filtro pasa banda", "0.5 – 40 Hz"), _conn(),
    _node("Normalización"), _conn(),
    _node("ECG procesado", "signal ready", end=True),
])

pipeline_b_nodes = "".join([
    _node("ECG procesado / anotaciones", start=True), _conn(),
    _node("Ventanas"), _conn(),
    _node("Features"), _conn(),
    _node("Dataset tabular"), _conn(),
    _node("Modelos"), _conn(),
    _node("Evaluación"), _conn(),
    _node("App", "demo académica", end=True),
])

rails_html = (
    '<div class="pm-rails">'
    '<div class="pm-rail pm-rail-a">'
    '<div class="pm-rh"><span class="pm-rtag">PIPELINE A</span>'
    '<span class="pm-rname">Señal ECG</span></div>'
    '<div class="pm-rsub">Transforma la señal cruda en una entrada limpia y estandarizada. '
    '<b style="color:var(--fg-0)">No es el modelo</b>: es el paso previo.</div>'
    f'<div class="pm-chain">{pipeline_a_nodes}</div>'
    '</div>'
    '<div class="pm-rail pm-rail-b">'
    '<div class="pm-rh"><span class="pm-rtag">PIPELINE B</span>'
    '<span class="pm-rname">Machine Learning</span></div>'
    '<div class="pm-rsub">Convierte la señal procesada y las anotaciones en un dataset tabular, modelos y evaluación.</div>'
    f'<div class="pm-chain">{pipeline_b_nodes}</div>'
    '</div>'
    '</div>'
    '<div class="pm-bridge">'
    '&rarr;&nbsp;'
    'El <b>ECG procesado</b> del Pipeline A es la entrada del Pipeline B — '
    'el preprocesamiento de señal no es el modelo, es el puente hacia él.'
    '</div>'
)

st.html(rails_html)

# ── 4 · Decisiones metodológicas ─────────────────────────────────────────────

section_title("Decisiones metodológicas")

decisions_html = (
    '<div class="pm-decisions">'

    '<div class="pm-dec pm-dec-amber">'
    '<div class="pm-decic">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
    '<path d="M3 3v18h18"/><path d="M7 15l3-4 3 2 4-6"/></svg>'
    '</div>'
    '<h4>F1-macro como métrica principal</h4>'
    '<p>Se prioriza F1-macro porque el dataset presenta fuerte desbalance entre clases.</p>'
    '</div>'

    '<div class="pm-dec pm-dec-green">'
    '<div class="pm-decic">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
    '<path d="M16 3h5v5M21 3l-7 7M8 21H3v-5M3 21l7-7"/></svg>'
    '</div>'
    '<h4>Split por case_id</h4>'
    '<p>Se evita que señales del mismo paciente aparezcan en entrenamiento y prueba.</p>'
    '</div>'

    '<div class="pm-dec pm-dec-red">'
    '<div class="pm-decic">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
    '<rect x="4" y="11" width="16" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>'
    '</div>'
    '<h4>beat_type bloqueado</h4>'
    '<p>beat_type se usa solo como información descriptiva, nunca como predictor, para evitar leakage.</p>'
    '</div>'

    '<div class="pm-dec">'
    '<div class="pm-decic">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>'
    '</div>'
    '<h4>No uso clínico</h4>'
    '<p>La app es una demo académica y no reemplaza el juicio médico.</p>'
    '</div>'

    '</div>'
)

st.html(decisions_html)

# ── 5 · Artefactos generados ──────────────────────────────────────────────────

section_title("Artefactos generados")

def _chip(name: str, ext: str) -> str:
    return (
        f'<span class="pm-chip">'
        f'<span class="pm-ext pm-{ext.lower()}">{ext}</span>'
        f'{name}'
        f'</span>'
    )

artifacts_html = (
    '<div class="pm-outputs">'
    + _chip("model_comparison.csv", "CSV")
    + _chip("best_model_classification_report.csv", "CSV")
    + _chip("best_model_confusion_matrix.png", "PNG")
    + _chip("best_model_feature_importance.csv", "CSV")
    + _chip("best_model_pipeline.joblib", "JOBLIB")
    + _chip("feature_columns.json", "JSON")
    + _chip("demo_cases_metadata.csv", "CSV")
    + '</div>'
)

st.html(artifacts_html)

# ── 6 · Callout final ─────────────────────────────────────────────────────────

callout(
    "warn",
    "Lectura de la metodología",
    "La metodología conecta procesamiento de señal, construcción de features, "
    "comparación de modelos y evaluación por clase. El desempeño global debe interpretarse "
    "con cautela debido al fuerte desbalance de ritmos intraoperatorios.",
)
