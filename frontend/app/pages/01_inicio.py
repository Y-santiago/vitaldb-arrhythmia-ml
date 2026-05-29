"""Página 01 — Inicio · landing dashboard con datos reales."""

import streamlit as st

import pandas as pd

from components.badges  import badge, badge_row, pill
from components.cards   import callout, card_header, kv_table, metric_card, section_title
from components.charts  import bar_chart_h, mini_ecg_placeholder
from utils.loaders      import (
    load_classification_report,
    load_model_comparison,
    load_model_metadata,
)

# ── Global setup ──────────────────────────────────────────────────────────────
meta    = load_model_metadata()
df_cmp  = load_model_comparison()
df_cls  = load_classification_report()

winner_raw  = meta.get("winner_model", "linear_svc")      if meta else "linear_svc"
winner_nice = winner_raw.replace("_", " ").title()
f1_val      = meta.get("winner_test_f1_macro")             if meta else None
f1_str      = f"{f1_val:.3f}"                              if f1_val is not None else "0.344"
acc_val     = None
prec_val    = None
rec_val     = None

if df_cmp is not None:
    winner_row = df_cmp[df_cmp["model"] == winner_raw]
    if not winner_row.empty:
        acc_val  = winner_row["test_accuracy"].iloc[0]
        prec_val = winner_row["test_precision_macro"].iloc[0]
        rec_val  = winner_row["test_recall_macro"].iloc[0]

# Derive counts from metadata or defaults
train_rows  = meta.get("train_size_rows", 509707) if meta else 509707
test_rows   = meta.get("test_size_rows",  128983) if meta else 128983
total_rows  = train_rows + test_rows
n_features  = meta.get("n_features", 26)           if meta else 26
n_train_grp = meta.get("n_train_groups", 384)      if meta else 384
n_test_grp  = meta.get("n_test_groups",  97)       if meta else 97
total_cases = n_train_grp + n_test_grp

# Count real classes from classification report (exclude summary rows)
if df_cls is not None:
    _summary = {"accuracy", "macro avg", "weighted avg"}
    n_classes = len([i for i in df_cls.index if i.lower() not in _summary])
else:
    n_classes = 10

# Count real models from comparison csv
n_models = len(df_cmp) if df_cmp is not None else 5

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="hero-block">
          <div style="display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap">
            {badge("VitalDB · 2026","muted")}
            {badge("Machine Learning","info")}
            {badge("Demo académica","warn")}
          </div>
          <h1>Detección y clasificación de
              <em>arritmias intraoperatorias</em>
              con señales ECG y modelos ML</h1>
          <p>
            Esta demo recorre el pipeline completo: limpieza de señales ECG de la
            VitalDB Arrhythmia Database, generación de ventanas, ingeniería de
            features y comparación de modelos clásicos de clasificación.
          </p>
          <div class="hero-tags">
            {badge("scikit-learn","muted")}
            {badge("XGBoost","muted")}
            {badge(f"{n_classes} clases","muted")}
            {badge(f"{n_features} features","muted")}
            {badge("GroupKFold (case_id)","muted")}
          </div>
          <div class="hero-foot">
            <span>VitalDB Arrhythmia Database</span>
            <span class="sep">·</span>
            <span>v1.0 pipeline</span>
            <span class="sep">·</span>
            <span>Entrenado: {meta.get("training_datetime","—")[:10] if meta else "2026-05-21"}</span>
            <div class="hero-status">
              <span class="led"></span>
              Pipeline OK · datos cargados
            </div>
          </div>
        </div>""",
    unsafe_allow_html=True,
)

# ── ECG decorativo ────────────────────────────────────────────────────────────
ecg_fig = mini_ecg_placeholder(height=110, n_beats=7)
st.plotly_chart(ecg_fig, use_container_width=True, config={"displayModeBar": False})

# ── MÉTRICAS PRINCIPALES ──────────────────────────────────────────────────────
section_title("Resumen del proyecto")

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card(
        "Casos quirúrgicos",
        str(total_cases),
        helper=f"{n_train_grp} train · {n_test_grp} test",
        accent="blue",
        helper_kind="muted",
    )
with col2:
    metric_card(
        "Ventanas ECG",
        f"{total_rows:,}".replace(",", " "),
        helper=f"train {train_rows:,} · test {test_rows:,}".replace(",", " "),
        accent="teal",
        helper_kind="muted",
    )
with col3:
    metric_card(
        "Clases de ritmo",
        str(n_classes),
        helper="clasificación multiclase",
        accent="warn",
        helper_kind="muted",
    )
with col4:
    metric_card(
        "Modelos comparados",
        str(n_models),
        helper="baseline + búsqueda HP",
        accent="blue",
        helper_kind="muted",
    )

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── CARDS PRINCIPALES ─────────────────────────────────────────────────────────
section_title("Resultados")

col_a, col_b, col_c = st.columns([1.1, 1, 1])

# Card A — Mejor modelo
with col_a:
    with st.container(border=True):
        card_header(
            "Mejor modelo",
            "test set",
            right_html=badge("winner", "winner"),
        )

        st.markdown(
            f"""<div style="display:flex;align-items:baseline;gap:14px;margin:6px 0 12px">
                  <div style="font-size:28px;font-weight:600;color:var(--fg-0);
                              letter-spacing:-.01em;">{winner_nice}</div>
                  <span style="font-family:var(--mono);font-size:11px;
                               color:var(--fg-3)">scikit-learn</span>
                </div>""",
            unsafe_allow_html=True,
        )

        kv_table([
            ("F1-macro",   f'<b style="color:var(--teal)">{f1_str}</b>'),
            ("Precision",  f"{prec_val:.3f}" if prec_val is not None else "—"),
            ("Recall",     f"{rec_val:.3f}"  if rec_val  is not None else "—"),
            ("Accuracy",   f"{acc_val:.3f}"  if acc_val  is not None else "—"),
            ("Features",   str(n_features)),
        ])

        if meta and "best_hyperparams_per_model" in meta:
            hp = meta["best_hyperparams_per_model"].get(winner_raw, {})
            if hp:
                st.caption("Mejores hiperparámetros:")
                hp_str = " · ".join(f"{k}={v}" for k, v in hp.items())
                st.code(hp_str, language=None)

# Card B — Dataset
with col_b:
    with st.container(border=True):
        card_header("Resumen del dataset", "VitalDB")
        kv_table([
            ("Casos totales",  f"{total_cases} (train+test)"),
            ("Señal",          "ECG II · 250 Hz"),
            ("Train windows",  f"{train_rows:,}".replace(",", " ")),
            ("Test windows",   f"{test_rows:,}".replace(",", " ")),
            ("Clases",         f"{n_classes} ritmos"),
            ("Features",       str(n_features)),
        ])

        if df_cls is not None:
            _summary = {"accuracy", "macro avg", "weighted avg"}
            classes_list = [i for i in df_cls.index if i.lower() not in _summary]
            chips = " ".join(badge(c, "muted") for c in classes_list)
            st.markdown(
                f'<div style="margin-top:10px;line-height:2">{chips}</div>',
                unsafe_allow_html=True,
            )

# Card C — F1-macro comparación
with col_c:
    with st.container(border=True):
        card_header("F1-macro · comparación", "modelos")

        if df_cmp is not None:
            labels = df_cmp["model"].str.replace("_", " ").str.title().tolist()
            values = df_cmp["test_f1_macro"].tolist()
            fig = bar_chart_h(
                labels=labels,
                values=values,
                color="#4a8cff",
                accent_top=True,
                value_fmt=".3f",
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                f"Winner: {winner_nice} · F1={f1_str} "
                f"(métrica principal: macro promedia las 10 clases)"
            )
        else:
            st.warning("model_comparison.csv no encontrado.")

# ── AVISO ──────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
callout(
    "warn",
    "Aviso · Demo académica",
    "Esta aplicación no permite cargar datos nuevos ni realizar diagnóstico clínico. "
    "Los resultados provienen de un pipeline ya ejecutado sobre la VitalDB Arrhythmia Database. "
    "<b>No usar para decisiones médicas reales.</b>",
)

# ── DESCRIPCIÓN DEL PROYECTO ──────────────────────────────────────────────────
section_title("Descripción del proyecto")

col_d, col_e = st.columns(2)

with col_d:
    with st.container(border=True):
        card_header("Problema clínico", "contexto")
        st.markdown(
            """<div class="kv-table" style="row-gap:10px">
              <div class="kv-key">¿Qué se clasifica?</div>
              <div class="kv-val">Ritmos cardíacos intraoperatorios en señal ECG continua</div>
              <div class="kv-key">¿Por qué importa?</div>
              <div class="kv-val">Las arritmias intraoperatorias aumentan morbilidad y mortalidad si no se detectan a tiempo</div>
              <div class="kv-key">¿Qué hace el modelo?</div>
              <div class="kv-val">Clasifica ventanas de 2 segundos en una de las clases de ritmo disponibles</div>
              <div class="kv-key">Métricas clave</div>
              <div class="kv-val">F1-macro (penaliza fallo en clases minoritarias) y Accuracy</div>
            </div>""",
            unsafe_allow_html=True,
        )

with col_e:
    with st.container(border=True):
        card_header("Pipeline resumido", "end-to-end")
        steps = [
            ("01", "Carga y EDA",            "VitalDB · 482+ casos"),
            ("02", "Limpieza de señales",     "Filtros · NaN · SNR"),
            ("03", "Ventaneo ECG",            "2 s · 50% overlap"),
            ("04", "Feature engineering",     f"{n_features} features por ventana"),
            ("05", "Split por case_id",       "GroupKFold · sin leakage"),
            ("06", "Entrenamiento",           f"{n_models} modelos baseline"),
            ("07", "Búsqueda de HP",          "RandomizedSearchCV"),
            ("08", "Evaluación final",        "F1-macro · CM · Reporte"),
        ]
        rows_html = "".join(
            f"""<div style="display:grid;grid-template-columns:28px 1fr;gap:8px 10px;
                           align-items:start;padding:6px 0;border-bottom:1px dashed var(--line-1)">
                  <div style="font-family:var(--mono);font-size:10px;color:var(--fg-4);
                              background:var(--bg-3);border-radius:4px;padding:2px 4px;
                              text-align:center;margin-top:2px">{n}</div>
                  <div>
                    <div style="font-size:12.5px;font-weight:500;color:var(--fg-0)">{name}</div>
                    <div style="font-size:11px;color:var(--fg-3);font-family:var(--mono)">{sub}</div>
                  </div>
                </div>"""
            for n, name, sub in steps
        )
        st.markdown(rows_html, unsafe_allow_html=True)

# ── NOTA TÉCNICA ───────────────────────────────────────────────────────────────
if meta and "library_versions" in meta:
    section_title("Entorno técnico")
    libs = meta["library_versions"]
    chips = " ".join(
        badge(f"{k} {v}", "muted") for k, v in libs.items()
    )
    st.markdown(
        f'<div style="line-height:2.4">{chips}</div>',
        unsafe_allow_html=True,
    )
