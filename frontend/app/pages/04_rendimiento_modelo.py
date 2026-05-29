"""Página 04 — Rendimiento del modelo.

Comparación de clasificadores sobre el conjunto de test (GroupKFold por case_id).
Métrica principal: F1-macro.
"""

import json

import pandas as pd
import streamlit as st

from components.badges import badge
from components.cards  import callout, card_header, kv_table, metric_card, section_title
from components.charts import fit_time_bar, model_metrics_bar
from components.layout import page_header
from components.tables import model_comparison_table
from utils.loaders     import load_model_comparison, load_model_metadata

# ── Setup ─────────────────────────────────────────────────────────────────────
meta   = load_model_metadata()
df_raw = load_model_comparison()

_winner_id  = meta.get("winner_model", "") if meta else ""
_f1_meta    = meta.get("winner_test_f1_macro") if meta else None
page_header(
    "Rendimiento del modelo",
    "Comparación de clasificadores entrenados para la detección multiclase "
    "de ritmos intraoperatorios.",
    badge_html=badge("Evaluación final", "info"),
)

# ── Early exit ────────────────────────────────────────────────────────────────
if df_raw is None:
    callout(
        "err",
        "Datos no disponibles",
        "No se encontró <code>reports/tables/model_comparison.csv</code>. "
        "Ejecuta el pipeline para generarlo.",
    )
    st.stop()

# ── Column detection helpers ──────────────────────────────────────────────────

def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy: any column whose name contains the candidate substring
    for c in candidates:
        matches = [col for col in df.columns if c in col.lower()]
        if matches:
            return matches[0]
    return None


COL_MODEL  = _find(df_raw, ["model"])
COL_STATUS = _find(df_raw, ["status"])
COL_F1     = _find(df_raw, ["test_f1_macro",        "f1_macro",     "f1"])
COL_PREC   = _find(df_raw, ["test_precision_macro",  "precision_macro", "precision"])
COL_REC    = _find(df_raw, ["test_recall_macro",     "recall_macro", "recall"])
COL_ACC    = _find(df_raw, ["test_accuracy",         "accuracy"])
COL_CV     = _find(df_raw, ["cv_f1_macro",           "cv_f1"])
COL_TIME   = _find(df_raw, ["fit_time_seconds",      "fit_time",     "training_time", "elapsed"])
COL_PARAMS = _find(df_raw, ["best_params",           "params",       "best_hyperparameters"])
COL_ERROR  = _find(df_raw, ["error_message",         "error"])

# ── Split ok / failed ─────────────────────────────────────────────────────────
if COL_STATUS:
    df_ok   = df_raw[df_raw[COL_STATUS].astype(str).str.lower() == "ok"].copy()
    df_fail = df_raw[df_raw[COL_STATUS].astype(str).str.lower() != "ok"].copy()
else:
    df_ok   = df_raw.copy()
    df_fail = pd.DataFrame()

if COL_F1 and not df_ok.empty:
    df_ok = df_ok.sort_values(COL_F1, ascending=False).reset_index(drop=True)

# ── Identify winner ───────────────────────────────────────────────────────────
_has_ok = not df_ok.empty and COL_MODEL

winner_row      = df_ok.iloc[0]       if _has_ok else None
winner_id       = str(winner_row[COL_MODEL]) if winner_row is not None else ""
winner_nice     = winner_id.replace("_", " ").title()

def _val(row, col, fmt=None, fallback="—"):
    """Safely get a formatted value from a Series row."""
    if row is None or col is None or col not in row.index:
        return fallback
    v = row[col]
    if pd.isna(v):
        return fallback
    if fmt:
        return format(v, fmt)
    return v

def _fmt_model(name: str) -> str:
    return str(name).replace("_", " ").title()

def _fmt_params(raw) -> str:
    """Parse best_params JSON string into a readable single line."""
    if pd.isna(raw) or raw == "":
        return "No disponible"
    try:
        params = json.loads(str(raw))
        cleaned = {
            k.replace("clf__", "").replace("__", "_"): v
            for k, v in params.items()
        }
        return " · ".join(f"{k} = {v}" for k, v in cleaned.items())
    except Exception:
        s = str(raw)
        return s[:120] + "…" if len(s) > 120 else s


# ── Lookup: model-specific visual chips ──────────────────────────────────────
_MODEL_CHIPS: dict[str, list[tuple[str, str]]] = {
    "linear_svc":    [("lineal", "info"), ("L2 hinge", "muted"), ("balanced", "ok")],
    "random_forest": [("ensemble", "info"), ("bagging", "muted"), ("balanced", "ok")],
    "xgboost":       [("boosting", "info"), ("tree-based", "muted"), ("grad. descent", "muted")],
    "mlp":           [("neural net", "info"), ("relu", "muted"), ("adam", "muted")],
    "decision_tree": [("árbol", "info"), ("entropy", "muted")],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Callout metodológico
# ═══════════════════════════════════════════════════════════════════════════════
callout(
    "info",
    "Metodología de evaluación",
    "Se compararon <b>"
    + str(len(df_ok))
    + " modelos</b> clásicos de clasificación multiclase. "
    "La métrica principal es <b>F1-macro</b>: promedia el F1 de cada clase con "
    "peso uniforme, penalizando al modelo cuando falla en clases minoritarias "
    "(especialmente relevante aquí, donde la clase NSR concentra ~65% de las ventanas). "
    "El split se realizó por <code>case_id</code> (GroupKFold, 5 folds) para garantizar "
    "que ventanas del mismo paciente nunca aparezcan simultáneamente en train y test. "
    "Estos resultados son <b>académicos</b> — no para uso clínico.",
)

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Modelo ganador + KPIs
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Modelo seleccionado")

col_win, col_kpis = st.columns([1.35, 1])

# ── Winner card ──────────────────────────────────────────────────────────────
with col_win:
    with st.container(border=True):
        card_header(
            "Modelo ganador",
            "mejor F1-macro en test",
            right_html=badge("winner", "winner"),
        )

        # Model name + chips
        chips = _MODEL_CHIPS.get(winner_id, [])
        chips_html = " ".join(badge(t, k) for t, k in chips)
        st.markdown(
            f"""<div style="display:flex;align-items:baseline;gap:14px;margin:6px 0 10px">
                  <div style="font-size:28px;font-weight:600;color:var(--fg-0);
                              letter-spacing:-.01em">{winner_nice}</div>
                  <span style="font-family:var(--mono);font-size:11px;color:var(--fg-3)">
                    scikit-learn
                  </span>
                </div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
                  {chips_html}
                </div>""",
            unsafe_allow_html=True,
        )

        # Key-value metrics
        kv_rows = []
        if COL_F1:
            kv_rows.append(("F1-macro (test)",
                            f'<b style="color:var(--teal)">{_val(winner_row, COL_F1, ".3f")}</b>'))
        if COL_PREC:
            kv_rows.append(("Precision (macro)", _val(winner_row, COL_PREC, ".3f")))
        if COL_REC:
            kv_rows.append(("Recall (macro)",    _val(winner_row, COL_REC, ".3f")))
        if COL_ACC:
            kv_rows.append(("Accuracy",           _val(winner_row, COL_ACC, ".3f")))
        if COL_CV:
            kv_rows.append(("CV F1-macro (mean)", _val(winner_row, COL_CV, ".3f")))
        if COL_TIME:
            t = winner_row[COL_TIME] if winner_row is not None and COL_TIME else None
            if t is not None and not pd.isna(t):
                kv_rows.append(("Tiempo entreno",
                                f"{t:.0f} s ({t/60:.1f} min)" if t < 3600 else f"{t/3600:.1f} h"))

        kv_table(kv_rows)

        # Hyperparameters
        if COL_PARAMS and winner_row is not None:
            params_str = _fmt_params(winner_row[COL_PARAMS])
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            st.caption("Mejores hiperparámetros:")
            st.code(params_str, language=None)

# ── KPI metrics ──────────────────────────────────────────────────────────────
with col_kpis:
    r1, r2 = st.columns(2)
    r3, r4 = st.columns(2)
    r5, r6 = st.columns(2)

    with r1:
        metric_card("Mejor modelo", winner_nice, accent="teal")
    with r2:
        metric_card(
            "F1-macro test",
            _val(winner_row, COL_F1, ".3f"),
            helper=f"CV: {_val(winner_row, COL_CV, '.3f')}",
            accent="teal",
            helper_kind="muted",
        )
    with r3:
        metric_card(
            "Accuracy test",
            _val(winner_row, COL_ACC, ".3f"),
            accent="blue",
        )
    with r4:
        metric_card(
            "Precision macro",
            _val(winner_row, COL_PREC, ".3f"),
            accent="blue",
        )
    with r5:
        metric_card(
            "Modelos comparados",
            str(len(df_raw)),
            helper=f"{len(df_ok)} exitosos · {len(df_fail)} fallidos",
            accent="warn",
            helper_kind="muted",
        )
    with r6:
        t_winner = (
            winner_row[COL_TIME]
            if winner_row is not None and COL_TIME and not pd.isna(winner_row[COL_TIME])
            else None
        )
        if t_winner:
            time_display = f"{t_winner:.0f} s"
        else:
            time_display = "—"
        metric_card("Tiempo entreno", time_display, accent="warn")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Gráficos comparativos
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Comparación entre modelos")

col_g1, col_g2 = st.columns([1.5, 1])

# ── Grouped metrics bar ────────────────────────────────────────────────────────
with col_g1:
    with st.container(border=True):
        card_header("Métricas por modelo", "test set · F1 · Precisión · Recall · Accuracy")

        # Build metrics dict from available columns
        model_labels = [_fmt_model(m) for m in df_ok[COL_MODEL]] if COL_MODEL else []
        winner_label_nice = winner_nice

        metrics_dict: dict[str, list[float]] = {}
        for col_key, display_name in [
            (COL_F1,   "F1-macro"),
            (COL_PREC, "Precision"),
            (COL_REC,  "Recall"),
            (COL_ACC,  "Accuracy"),
        ]:
            if col_key and not df_ok.empty:
                metrics_dict[display_name] = df_ok[col_key].tolist()

        if metrics_dict and model_labels:
            fig_metrics = model_metrics_bar(
                labels=model_labels,
                metrics=metrics_dict,
                winner_label=winner_label_nice,
                height=310,
            )
            st.plotly_chart(
                fig_metrics, use_container_width=True, config={"displayModeBar": False}
            )
            st.caption(
                f"El modelo ganador ({winner_nice}) está destacado con fondo teal. "
                "Todos los valores están en escala 0–1."
            )
        else:
            st.warning("No hay métricas suficientes para graficar.")

# ── Training time bar ─────────────────────────────────────────────────────────
with col_g2:
    with st.container(border=True):
        card_header("Tiempo de entrenamiento", "segundos · menor es mejor")

        if COL_TIME and not df_ok.empty and COL_MODEL:
            time_labels = [_fmt_model(m) for m in df_ok[COL_MODEL]]
            time_values = df_ok[COL_TIME].tolist()

            fig_time = fit_time_bar(
                labels=time_labels,
                times=time_values,
                winner_label=winner_label_nice,
                height=310,
            )
            st.plotly_chart(
                fig_time, use_container_width=True, config={"displayModeBar": False}
            )

            # Trade-off callout
            if winner_row is not None and COL_TIME:
                t_w = winner_row[COL_TIME]
                t_max = df_ok[COL_TIME].max()
                t_max_model = _fmt_model(
                    df_ok.loc[df_ok[COL_TIME].idxmax(), COL_MODEL]
                )
                if not pd.isna(t_w) and not pd.isna(t_max):
                    callout(
                        "info",
                        "Trade-off velocidad / calidad",
                        f"{winner_nice} entrena en <b>{t_w:.0f} s</b> y obtiene el mejor "
                        f"F1-macro, frente a {t_max_model} que tarda "
                        f"<b>{t_max:.0f} s ({t_max/60:.0f} min)</b>.",
                    )
        else:
            st.warning("Columna de tiempo de entrenamiento no disponible.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Tabla comparativa completa
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Tabla comparativa completa")

with st.container(border=True):
    card_header(
        "Todos los modelos",
        "ordenados por F1-macro · test set",
        right_html=badge(f"{len(df_ok)} modelos exitosos", "ok"),
    )
    model_comparison_table(df_ok, winner_model_id=winner_id)

    st.caption(
        "La fila destacada corresponde al modelo ganador seleccionado para la demo. "
        "CV F1-macro es el promedio en validación cruzada (no en test set)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Hiperparámetros (expander)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Mejores hiperparámetros por modelo", expanded=False):
    if COL_PARAMS and COL_MODEL and not df_ok.empty:
        for _, row in df_ok.iterrows():
            model_name = _fmt_model(row[COL_MODEL])
            is_winner  = str(row[COL_MODEL]) == winner_id

            label_html = (
                f'{model_name} &nbsp; {badge("winner", "winner")}'
                if is_winner
                else model_name
            )
            params_str = _fmt_params(row[COL_PARAMS])

            st.markdown(
                f"""<div style="display:grid;grid-template-columns:160px 1fr;
                               gap:10px;align-items:start;
                               padding:10px 14px;margin-bottom:6px;
                               border:1px solid var(--line-1);border-radius:8px;
                               background:var(--bg-1);">
                      <div style="font-weight:600;color:var(--fg-0);font-size:13px">
                        {label_html}
                      </div>
                      <code style="font-family:var(--mono);font-size:11.5px;
                                   color:var(--fg-1);word-break:break-all">
                        {params_str}
                      </code>
                    </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No se encontraron columnas de hiperparámetros en el CSV.")

    # Strategy details from metadata
    if meta:
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        card_header("Estrategia de búsqueda", "metodología")
        kv_table([
            ("Método",           "RandomizedSearchCV"),
            ("Split",            "StratifiedGroupKFold · 5-fold · group=case_id"),
            ("Scoring",          "f1_macro"),
            ("Train groups",     str(meta.get("n_train_groups", "—"))),
            ("Test groups",      str(meta.get("n_test_groups", "—"))),
            ("Features",         str(meta.get("n_features", "—"))),
            ("Seed / modo",      f"fast_mode={meta.get('fast_mode','—')}"),
            ("Fecha entreno",    str(meta.get("training_datetime", "—"))[:19]),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — Modelos fallidos (si existen)
# ═══════════════════════════════════════════════════════════════════════════════
if not df_fail.empty:
    with st.expander(
        f"Modelos con error o no evaluados ({len(df_fail)})", expanded=False
    ):
        callout(
            "warn",
            "Modelos no evaluados",
            "Los siguientes modelos no completaron la evaluación. "
            "Se conservan en el informe por transparencia metodológica.",
        )
        for _, row in df_fail.iterrows():
            model_name = _fmt_model(row.get(COL_MODEL, "—"))
            status_val = str(row.get(COL_STATUS, "—"))
            error_val  = str(row.get(COL_ERROR, ""))  if COL_ERROR else "—"
            st.markdown(
                f"""<div style="padding:10px 14px;margin-bottom:6px;
                               border:1px solid rgba(248,113,113,.25);
                               border-radius:8px;background:var(--err-bg);">
                      <div style="font-weight:600;color:var(--err)">{model_name}
                        &nbsp; {badge(status_val, "err")}
                      </div>
                      <div style="font-size:12px;color:var(--fg-2);margin-top:4px">
                        {error_val if error_val and error_val != 'nan' else "Sin detalle disponible"}
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
else:
    # All models succeeded — show a quiet confirmation
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — Nota interpretativa
# ═══════════════════════════════════════════════════════════════════════════════
section_title("Interpretación de resultados")

# Build interpretation from real data
f1_winner_str = _val(winner_row, COL_F1, ".3f")
acc_winner_str = _val(winner_row, COL_ACC, ".3f")

# Second best model
second_str = ""
if len(df_ok) > 1 and COL_MODEL and COL_F1:
    second_row  = df_ok.iloc[1]
    second_name = _fmt_model(second_row[COL_MODEL])
    second_f1   = _val(second_row, COL_F1, ".3f")
    second_str  = (
        f" El segundo mejor modelo fue <b>{second_name}</b> con F1-macro = {second_f1}."
    )

with st.container(border=True):
    card_header("Conclusión de esta sección", "lectura cualitativa")
    st.markdown(
        f"""<div style="font-size:13.5px;color:var(--fg-1);line-height:1.75">
              <p>
                El modelo <b style="color:var(--teal)">{winner_nice}</b> obtuvo el mejor
                F1-macro en el conjunto de test: <b>{f1_winner_str}</b> con una accuracy de
                <b>{acc_winner_str}</b>.{second_str}
              </p>
              <p>
                La <b>accuracy ({acc_winner_str})</b> puede resultar engañosamente alta en este
                problema: dado el fuerte desbalance de clases (la clase dominante concentra la
                mayoría de ventanas), un clasificador trivial que prediga siempre la clase más
                frecuente también obtendría alta accuracy. Por eso el <b>F1-macro ({f1_winner_str})</b>
                es la métrica principal — pondera por igual a todas las clases, incluyendo las
                minoritarias clínicamente relevantes.
              </p>
              <p>
                El valor de F1-macro obtenido indica que el modelo tiene dificultades con varias
                clases minoritarias. Los detalles de rendimiento por clase se exploran en la
                siguiente sección: <b>Evaluación por clase</b>.
              </p>
            </div>""",
        unsafe_allow_html=True,
    )

    callout(
        "warn",
        "Limitación metodológica",
        "Estos resultados corresponden a un pipeline académico en modo <code>fast_mode</code>, "
        "con búsqueda de hiperparámetros reducida. Los valores de F1-macro podrían mejorar "
        "con más iteraciones de búsqueda, augmentación de clases minoritarias (SMOTE) o "
        "modelos basados en representaciones aprendidas (1D-CNN).",
    )
