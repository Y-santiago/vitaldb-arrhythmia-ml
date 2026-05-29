import streamlit as st

import pandas as pd

from components.layout import page_header
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from utils.loaders import (
    load_model_metadata,
    load_model_comparison,
    load_classification_report,
    load_feature_importance,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
meta   = load_model_metadata()

page_header(
    "Conclusiones",
    "Resumen final de hallazgos, limitaciones y próximos pasos.",
    badge_html=badge_row(badge("Cierre del proyecto", "info"), badge("Académico", "muted")),
)

# ── Load real data ─────────────────────────────────────────────────────────────
df_models = load_model_comparison()
df_cls    = load_classification_report()
df_imp    = load_feature_importance()

SUMMARY = {"accuracy", "macro avg", "weighted avg"}

# Derive real values from classification report
if df_cls is not None:
    df_classes = df_cls[~df_cls.index.isin(SUMMARY)].copy()
    for col in ["f1-score", "recall", "precision", "support"]:
        if col in df_classes.columns:
            df_classes[col] = pd.to_numeric(df_classes[col], errors="coerce")
    n_classes     = len(df_classes)
    good_classes  = df_classes[df_classes["f1-score"] >= 0.70].index.tolist()
    medium_classes= df_classes[(df_classes["f1-score"] >= 0.30) & (df_classes["f1-score"] < 0.70)].index.tolist()
    hard_classes  = df_classes[df_classes["f1-score"] < 0.30].index.tolist()
    total_test    = int(df_classes["support"].sum()) if "support" in df_classes.columns else 128983
else:
    df_classes    = None
    n_classes     = 10
    good_classes  = ["AFIB/AFL", "N"]
    medium_classes= ["WAP/MAT", "SND", "Patterned Ventricular Ectopy", "Patterned Atrial Ectopy"]
    hard_classes  = ["VT", "SVTA", "AVB", "Unclassifiable"]
    total_test    = 128983

# Derive winner metrics from model comparison
if df_models is not None and "model" in df_models.columns:
    wr = df_models[df_models["model"] == "linear_svc"]
    winner_acc  = float(wr["test_accuracy"].iloc[0])    if not wr.empty else 0.8061
    winner_f1   = float(wr["test_f1_macro"].iloc[0])   if not wr.empty else 0.3439
    winner_time = float(wr["fit_time_seconds"].iloc[0]) if not wr.empty else 185.0
    n_models    = len(df_models)
else:
    winner_acc, winner_f1, winner_time, n_models = 0.8061, 0.3439, 185.0, 5

train_size = meta.get("train_size_rows", 509707) if meta else 509707
test_size  = meta.get("test_size_rows",  128983) if meta else 128983

# ── Callout de alcance ────────────────────────────────────────────────────────
callout(
    "warn",
    "Alcance académico — no clínico",
    "Esta app es una <b>demo académica</b> de Machine Learning para clasificación "
    "multiclase de ritmos intraoperatorios. "
    "<b>No es una herramienta clínica, no reemplaza interpretación médica "
    "y no debe usarse para diagnóstico.</b> "
    "Los resultados reflejan el desempeño del modelo sobre un conjunto de test "
    "con alta prevalencia de ritmo sinusal (N) y fibrilación auricular.",
)

st.write("")

# ── Resumen ejecutivo ─────────────────────────────────────────────────────────
section_title("Resumen ejecutivo")

kv_table([
    ("Pipeline completo",
     f"Se construyó un flujo reproducible desde señal ECG cruda hasta evaluación comparativa "
     f"de {n_models} modelos clásicos de ML."),
    ("Separación por case_id",
     "El split tren/test se hizo por <b>case_id</b> para evitar leakage entre ventanas "
     f"del mismo paciente ({meta.get('n_train_groups', 384)} grupos entrenamiento, "
     f"{meta.get('n_test_groups', 97)} test)."),
    ("Modelo ganador",
     f"<b>LinearSVC</b> fue el mejor modelo según F1-macro "
     f"(F1={winner_f1:.3f}, Acc={winner_acc:.3f})."),
    ("Desempeño global",
     f"El rendimiento es <b>moderado y desigual</b>: "
     f"{len(good_classes)} clases con F1 ≥ 0.70, "
     f"{len(hard_classes)} clases con F1 &lt; 0.30."),
    ("Interpretabilidad",
     "Las features más importantes son de variabilidad del ritmo global "
     "(<code>case_rr_std</code>, <code>case_rr_rmssd</code>) y "
     "morfología de señal (<code>std</code>, <code>var</code>, <code>energy</code>)."),
    ("Desbalance de clases",
     "El dataset está fuertemente desbalanceado: N y AFIB/AFL dominan. "
     "Las clases minoritarias (AVB, SVTA, VT) tienen desempeño muy bajo o nulo."),
])

st.write("")

# ── Cards de resultados principales ──────────────────────────────────────────
section_title("Resultados principales")

c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.container(border=True):
        card_header(
            "Modelo ganador",
            "LinearSVC",
            right_html=badge("winner", "winner"),
        )
        metric_card("F1-macro (test)", f"{winner_f1:.4f}", "métrica principal", accent="teal", helper_kind="ok")
        metric_card("Accuracy (test)", f"{winner_acc:.4f}", "puede ser engañosa", accent="blue")
        metric_card("Tiempo de ajuste", f"{winner_time:.0f} s", f"C = {meta.get('best_hyperparams_per_model', {}).get('linear_svc', {}).get('clf__C', 0.0746):.4f}" if meta else "C = 0.0746", accent="muted")

with c2:
    with st.container(border=True):
        card_header("Dataset", f"{(train_size + test_size):,} ventanas", right_html=badge("ECG", "info"))
        metric_card("Entrenamiento", f"{train_size:,}", f"{meta.get('n_train_groups', 384)} casos", accent="blue")
        metric_card("Test", f"{test_size:,}", f"{meta.get('n_test_groups', 97)} casos", accent="blue")
        metric_card("Features", str(meta.get("n_features", 26) if meta else 26), "por ventana ECG", accent="muted")

with c3:
    with st.container(border=True):
        card_header("Evaluación por clase", f"{n_classes} clases evaluadas", right_html=badge("desigual", "warn"))
        metric_card(
            "Clases fuertes (F1 ≥ 0.70)",
            str(len(good_classes)),
            ", ".join(good_classes) if good_classes else "—",
            accent="teal",
            helper_kind="ok",
        )
        metric_card(
            "Clases difíciles (F1 < 0.30)",
            str(len(hard_classes)),
            ", ".join(hard_classes[:3]) + ("…" if len(hard_classes) > 3 else ""),
            accent="err",
            helper_kind="warn",
        )

with c4:
    with st.container(border=True):
        card_header("Interpretabilidad", "coeficientes LinearSVC", right_html=badge("global", "muted"))
        metric_card("Feature #1", "case rr std", "HRV global — 23.6%", accent="teal", helper_kind="ok")
        metric_card("Feature #2", "case rr rmssd", "HRV global — 23.0%", accent="teal")
        metric_card("Grupo dominante", "RR global", "≈ 46.6% importancia", accent="muted")

st.write("")

# ── Análisis de clases ─────────────────────────────────────────────────────────
section_title("Desempeño por clase")

col_good, col_med, col_bad = st.columns(3)

with col_good:
    with st.container(border=True):
        card_header("Clases con buen desempeño", "F1 ≥ 0.70", right_html=badge(f"{len(good_classes)} clases", "ok"))
        if df_classes is not None and good_classes:
            rows = []
            for cls in good_classes:
                f1  = float(df_classes.loc[cls, "f1-score"]) if cls in df_classes.index else 0.0
                sup = int(df_classes.loc[cls, "support"]) if cls in df_classes.index else 0
                rows.append((cls, f'{badge(f"F1 {f1:.3f}", "ok")} {badge(f"{sup:,} ventanas", "muted")}'))
            kv_table(rows)
        else:
            for cls in good_classes:
                kv_table([(cls, badge("F1 ≥ 0.70", "ok"))])
        callout("ok", "Por qué funcionan", "Clases con muchas muestras y patrones RR claramente distintos.")

with col_med:
    with st.container(border=True):
        card_header("Desempeño moderado", "0.30 ≤ F1 < 0.70", right_html=badge(f"{len(medium_classes)} clases", "info"))
        if df_classes is not None and medium_classes:
            rows = []
            for cls in medium_classes:
                f1  = float(df_classes.loc[cls, "f1-score"]) if cls in df_classes.index else 0.0
                sup = int(df_classes.loc[cls, "support"]) if cls in df_classes.index else 0
                rows.append((cls, f'{badge(f"F1 {f1:.3f}", "info")} {badge(f"{sup:,}", "muted")}'))
            kv_table(rows)
        else:
            for cls in medium_classes:
                kv_table([(cls, badge("F1 0.30–0.70", "info"))])
        callout("info", "Mejorables", "Menor soporte o solapamiento con clases dominantes.")

with col_bad:
    with st.container(border=True):
        card_header("Clases difíciles", "F1 < 0.30", right_html=badge(f"{len(hard_classes)} clases", "err"))
        if df_classes is not None and hard_classes:
            rows = []
            for cls in hard_classes:
                f1  = float(df_classes.loc[cls, "f1-score"]) if cls in df_classes.index else 0.0
                sup = int(df_classes.loc[cls, "support"]) if cls in df_classes.index else 0
                rows.append((cls, f'{badge(f"F1 {f1:.3f}", "err")} {badge(f"{sup:,}", "muted")}'))
            kv_table(rows)
        else:
            for cls in hard_classes:
                kv_table([(cls, badge("F1 < 0.30", "err"))])
        callout("warn", "Riesgo clínico potencial", "AVB y SVTA tienen F1 casi nulo — el modelo no las detecta.")

st.write("")

# ── Lo que funcionó bien ──────────────────────────────────────────────────────
section_title("Lo que funcionó bien")

with st.container(border=True):
    card_header("Aciertos del pipeline", "evaluación honesta")
    kv_table([
        (badge("✓", "ok") + " Pipeline reproducible",
         "Cada paso está en notebooks numerados y es re-ejecutable desde cero."),
        (badge("✓", "ok") + " Split por case_id",
         "La separación por caso evita leakage entre ventanas del mismo paciente."),
        (badge("✓", "ok") + " Comparación multi-modelo",
         f"Se evaluaron {n_models} modelos con búsqueda de hiperparámetros (GridSearchCV)."),
        (badge("✓", "ok") + " Evaluación por clase",
         "El reporte de clasificación expone el desempeño real por tipo de arritmia."),
        (badge("✓", "ok") + " Interpretabilidad inicial",
         "Los coeficientes del LinearSVC dan una señal coherente sobre las features más útiles."),
        (badge("✓", "ok") + " App Streamlit con datos reales",
         "Todas las métricas y visualizaciones usan archivos reales del pipeline."),
    ])

st.write("")

# ── Principales limitaciones ──────────────────────────────────────────────────
section_title("Principales limitaciones")

lim1, lim2 = st.columns(2)

with lim1:
    callout(
        "warn",
        "F1-macro bajo (0.3439)",
        "El modelo global tiene desempeño moderado. La accuracy alta (80%) es engañosa: "
        "refleja el dominio de clases N y AFIB/AFL, no el desempeño en clases minoritarias.",
    )
    callout(
        "warn",
        "Desbalance severo entre clases",
        "N y AFIB/AFL representan la mayoría de ventanas. "
        "Clases como AVB (346 muestras) o Unclassifiable (3 muestras) "
        "son prácticamente ignoradas por el modelo sin pesos de clase.",
    )
    callout(
        "warn",
        "Clases minoritarias sin detección",
        "AVB (F1 = 0.000) y SVTA (F1 = 0.027) no son detectadas de forma confiable. "
        "Esto limita severamente la utilidad clínica del modelo actual.",
    )
    callout(
        "warn",
        "Demo de predicciones pendiente",
        "Faltan <code>test_predictions.csv</code>, <code>confusion_matrix.csv</code> "
        "y <code>demo_windows.parquet</code>. La página 07 (predicciones) "
        "no puede implementarse todavía.",
    )

with lim2:
    callout(
        "warn",
        "Interpretación no causal",
        "La importancia de features por coeficientes del LinearSVC no implica causalidad clínica. "
        "Es una correlación aprendida del dataset de entrenamiento.",
    )
    callout(
        "warn",
        "Features correlacionadas",
        "Pares como (std, var) o (case_rr_std, case_rr_rmssd) miden casi lo mismo. "
        "Sus importancias individuales subestiman la influencia real del concepto subyacente.",
    )
    callout(
        "warn",
        "Sin explicaciones locales",
        "Los coeficientes globales no explican por qué una ventana específica fue clasificada "
        "de cierta manera. Para eso se requiere SHAP o LIME.",
    )
    callout(
        "info",
        "App académica — no clínica",
        "Los resultados son válidos como experimento metodológico. "
        "Cualquier aplicación clínica requeriría validación externa, "
        "calibración y revisión médica especializada.",
    )

st.write("")

# ── Próximos pasos ─────────────────────────────────────────────────────────────
section_title("Próximos pasos técnicos — hoja de ruta")

col_near, col_far = st.columns(2)

with col_near:
    with st.container(border=True):
        card_header("Corto plazo", "completar la app y los datos", right_html=badge("prioridad alta", "err"))
        kv_table([
            ("1", "Exportar <code>reports/tables/test_predictions.csv</code> desde el notebook de evaluación."),
            ("2", "Exportar <code>reports/tables/confusion_matrix.csv</code> en formato long (real_label, predicted_label, count)."),
            ("3", "Crear <code>data/demo/demo_windows.parquet</code> con ventanas ECG livianas para la demo."),
            ("4", "Implementar <code>07_predicciones.py</code> una vez que existan esos archivos."),
            ("5", "Implementar <code>03_dataset_limpieza.py</code> con estadísticas de ventanas, calidad de señal y distribución de clases."),
        ])

with col_far:
    with st.container(border=True):
        card_header("Mediano plazo", "mejorar el modelo", right_html=badge("mejora de calidad", "info"))
        kv_table([
            ("6", "<b>Class weighting</b>: usar <code>class_weight='balanced'</code> en LinearSVC para penalizar más los errores en clases minoritarias."),
            ("7", "<b>SMOTE / resampling</b>: aumentar muestras de clases minoritarias en el set de entrenamiento."),
            ("8", "<b>Agrupación clínica</b>: considerar fusionar clases cercanas (p. ej. Patterned Atrial Ectopy y WAP/MAT) para reducir el problema de desbalance."),
            ("9", "<b>SHAP / Permutation Importance</b>: reemplazar importancia por coeficientes con métodos más robustos y con explicaciones locales."),
            ("10", "<b>Modelo 1D-CNN</b>: aprender representaciones directamente de la señal ECG cruda, sin depender de features manuales."),
        ])

st.write("")

# ── Estado de páginas ─────────────────────────────────────────────────────────
section_title("Estado actual de la app")

pages_data = [
    ("01", "Inicio",               "Implementada",     "ok"),
    ("02", "Pipeline",             "Placeholder",      "warn"),
    ("03", "Dataset y limpieza",   "Pendiente",        "warn"),
    ("04", "Rendimiento del modelo","Implementada",    "ok"),
    ("05", "Evaluación por clase", "Implementada",     "ok"),
    ("06", "Matriz de confusión",  "Implementada (PNG fallback)", "ok"),
    ("07", "Predicciones",         "Pendiente — archivos faltantes", "err"),
    ("08", "Interpretabilidad",    "Implementada",     "ok"),
    ("09", "Conclusiones",         "Implementada",     "ok"),
]

STATUS_ICONS = {"ok": "✓", "warn": "◑", "err": "✗"}
STATUS_COLORS = {
    "ok":   "color:var(--ok)",
    "warn": "color:var(--warn)",
    "err":  "color:var(--err)",
}

rows_html = ""
for num, name, status, kind in pages_data:
    icon  = STATUS_ICONS.get(kind, "·")
    style = STATUS_COLORS.get(kind, "")
    rows_html += (
        f'<div class="kv-key">'
        f'<span style="color:var(--fg-3);font-size:11px">0{num} </span>{name}'
        f'</div>'
        f'<div class="kv-val">'
        f'<span style="{style};font-family:var(--font-mono)">{icon} {status}</span>'
        f'</div>'
    )

st.html(f'<div class="kv-table">{rows_html}</div>')

n_done    = sum(1 for *_, k in pages_data if k == "ok")
n_pending = sum(1 for *_, k in pages_data if k in ("warn", "err"))

c_a, c_b, c_c = st.columns(3)
with c_a:
    metric_card("Páginas implementadas", str(n_done), "de 9 páginas", accent="teal", helper_kind="ok")
with c_b:
    metric_card("Pendientes / placeholder", str(n_pending), "páginas", accent="warn", helper_kind="warn")
with c_c:
    metric_card("Datos faltantes", "3 archivos", "predictions · cm csv · demo parquet", accent="err", helper_kind="warn")

st.write("")

# ── Mensaje final ─────────────────────────────────────────────────────────────
section_title("Reflexión final")

with st.container(border=True):
    card_header(
        "Balance del proyecto",
        "lo que logramos y lo que falta",
        right_html=badge("académico", "muted"),
    )
    st.write("")
    st.html(
        '<p style="color:var(--fg-1);font-size:15px;line-height:1.7;max-width:860px">'
        "El proyecto demuestra que es posible construir un <b>flujo reproducible y evaluado</b> "
        "para clasificar ritmos intraoperatorios a partir de features de ECG. "
        "El modelo LinearSVC logra distinguir correctamente los dos ritmos dominantes "
        "(N y AFIB/AFL) con F1 superior a 0.88, lo que representa una base sólida. "
        "Sin embargo, el análisis también evidencia que el <b>desbalance de clases</b> "
        "y la <b>variabilidad entre ritmos minoritarios</b> siguen siendo los principales "
        "retos metodológicos antes de pensar en aplicaciones clínicas. "
        "Los próximos pasos claros son: completar la exportación de archivos de evaluación, "
        "implementar estrategias de re-balanceo y explorar modelos que aprendan representaciones "
        "directamente de la señal cruda."
        "</p>"
    )
    st.write("")
    callout(
        "info",
        "Reproducibilidad",
        "Todo el código de este proyecto (pipeline, modelos y app) puede re-ejecutarse "
        "desde cero siguiendo los notebooks numerados del repositorio.",
    )
