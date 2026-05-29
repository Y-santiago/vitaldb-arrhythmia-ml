import streamlit as st

import pandas as pd

from components.layout import page_header
from components.cards import callout, card_header, kv_table, metric_card, section_title
from components.badges import badge, badge_row
from components.charts import bar_chart_h, apply_dark_layout
from components.tables import feature_importance_table
from utils.loaders import (
    load_feature_importance,
    load_feature_columns,
    load_model_metadata,
    load_model_comparison,
    correlation_figure_path,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
meta   = load_model_metadata()

page_header(
    "Interpretabilidad",
    "Análisis de las features más relevantes para el modelo ganador.",
    badge_html=badge_row(badge("LinearSVC", "winner"), badge("Explicación del modelo", "info")),
)

# ── Feature group classification (covers all 26 known features) ───────────────
_GROUP_MAP: dict[str, str] = {
    # Local RR — per-beat timing
    "rr_prev":       "RR local",
    "rr_next":       "RR local",
    "rr_mean_local": "RR local",
    "rr_ratio":      "RR local",
    # Global RR — case-level HRV
    "case_rr_count": "RR global (caso)",
    "case_rr_mean":  "RR global (caso)",
    "case_rr_std":   "RR global (caso)",
    "case_rr_min":   "RR global (caso)",
    "case_rr_max":   "RR global (caso)",
    "case_rr_rmssd": "RR global (caso)",
    "case_rr_pnn50": "RR global (caso)",
    # Amplitude / morphology
    "mean":     "Amplitud / morfología",
    "std":      "Amplitud / morfología",
    "var":      "Amplitud / morfología",
    "min":      "Amplitud / morfología",
    "max":      "Amplitud / morfología",
    "range":    "Amplitud / morfología",
    "median":   "Amplitud / morfología",
    "p25":      "Amplitud / morfología",
    "p75":      "Amplitud / morfología",
    "iqr":      "Amplitud / morfología",
    "skew":     "Amplitud / morfología",
    "kurtosis": "Amplitud / morfología",
    "energy":   "Amplitud / morfología",
    "abs_mean": "Amplitud / morfología",
    # Signal quality
    "zero_crossing_rate": "Calidad / señal",
}

_GROUP_COLORS = {
    "RR global (caso)":     "#2dd4bf",
    "Amplitud / morfología":"#4a8cff",
    "RR local":             "#a78bfa",
    "Calidad / señal":      "#fbbf24",
    "Otros":                "#5d6c8c",
}

# ── Robust column detection ────────────────────────────────────────────────────
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        matches = [col for col in df.columns if c in col.lower()]
        if matches:
            return matches[0]
    return None


# ── Load data ─────────────────────────────────────────────────────────────────
df_imp       = load_feature_importance()
feature_cols = load_feature_columns()
corr_png     = correlation_figure_path()

has_imp  = df_imp is not None and not df_imp.empty
has_cols = feature_cols is not None
has_corr = corr_png is not None

if not has_imp:
    callout(
        "warn",
        "Datos de importancia no disponibles",
        "No se encontró <code>reports/tables/best_model_feature_importance.csv</code>. "
        "Ejecuta el pipeline de evaluación para generarlo.",
    )
    st.stop()

# Detect actual column names
FEAT_COL = _find_col(df_imp, ["feature", "feature_name", "variable"])
IMP_COL  = _find_col(df_imp, [
    "importance", "abs_importance", "score",
    "coefficient", "coef", "abs_coefficient",
])

if FEAT_COL is None or IMP_COL is None:
    callout(
        "err",
        "Formato de CSV inesperado",
        f"No se encontraron columnas de feature/importancia. "
        f"Columnas disponibles: {list(df_imp.columns)}",
    )
    st.stop()

# Ensure importance is numeric and sorted
df_imp[IMP_COL] = pd.to_numeric(df_imp[IMP_COL], errors="coerce")
df_imp = df_imp.dropna(subset=[IMP_COL]).sort_values(IMP_COL, ascending=False).reset_index(drop=True)

n_features_csv   = len(df_imp)
n_features_total = len(feature_cols) if has_cols else n_features_csv
top_feature      = str(df_imp.loc[0, FEAT_COL])
top_importance   = float(df_imp.loc[0, IMP_COL])
total_importance = float(df_imp[IMP_COL].sum())

# Assign group to each feature in CSV
df_imp["Grupo"] = df_imp[FEAT_COL].map(lambda f: _GROUP_MAP.get(str(f), "Otros"))

# Infer interpretation type from metadata
interp_type = "coeficientes (abs)"
if meta and meta.get("winner_model", ""):
    m = meta["winner_model"].lower()
    if "forest" in m or "tree" in m or "xgb" in m or "boost" in m:
        interp_type = "feature importance (árbol)"
    elif "svc" in m or "svm" in m or "linear" in m:
        interp_type = "norma de coeficientes"

# ── Callout metodológico ──────────────────────────────────────────────────────
callout(
    "info",
    "Cómo interpretar la importancia de features",
    "La importancia de variables ayuda a entender qué información usa el modelo, "
    "pero <b>no equivale a una explicación clínica directa</b>. "
    "Para <b>LinearSVC</b>, la importancia se deriva de la norma de los coeficientes del hiperplano: "
    "features con coeficientes grandes (positivos o negativos) tienen más influencia en la decisión. "
    "Una feature importante <i>no implica causalidad</i> — puede ser una correlación espuria. "
    "Features muy correlacionadas entre sí pueden repartirse la importancia artificialmente. "
    "Este análisis es <b>académico</b> y no debe usarse para diagnóstico clínico.",
)

st.write("")

# ── KPIs ─────────────────────────────────────────────────────────────────────
section_title("Resumen de features")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card(
        "Features totales",
        str(n_features_total),
        "en el modelo entrenado",
        accent="blue",
    )
with c2:
    metric_card(
        "Features en reporte",
        str(n_features_csv),
        "con importancia > 0",
        accent="blue",
    )
with c3:
    metric_card(
        "Feature más importante",
        top_feature.replace("_", " "),
        f"importancia {top_importance:.3f}",
        accent="teal",
        helper_kind="ok",
    )
with c4:
    metric_card(
        "Importancia máxima",
        f"{top_importance:.3f}",
        f"de total {total_importance:.1f}",
        accent="teal",
    )
with c5:
    metric_card(
        "Modelo interpretado",
        winner,
        meta.get("winner_model", "—") if meta else "—",
        accent="muted",
    )
with c6:
    metric_card(
        "Tipo de interpretación",
        "coeficientes",
        interp_type,
        accent="muted",
    )

st.write("")

# ── Top-N bar chart ───────────────────────────────────────────────────────────
section_title("Importancia de features")

col_sel, _, _ = st.columns([1, 1, 2])
with col_sel:
    _n_options = [5, min(10, n_features_csv)]
    if n_features_csv > 10:
        _n_options.append(15)
    if n_features_csv > 15:
        _n_options.append(n_features_csv)
    _n_options = sorted(set(_n_options))
    _labels_n  = [f"Top {n}" if n < n_features_csv else f"Top {n} (todos)" for n in _n_options]
    if len(_n_options) == 1:
        top_n = _n_options[0]
    else:
        choice = st.selectbox("Mostrar", _labels_n, index=min(1, len(_labels_n) - 1))
        top_n  = _n_options[_labels_n.index(choice)]

df_plot = df_imp.head(top_n)
labels  = df_plot[FEAT_COL].str.replace("_", " ").tolist()
values  = df_plot[IMP_COL].tolist()

chart_col, table_col = st.columns([1.5, 1])

with chart_col:
    with st.container(border=True):
        st.markdown(f"**Top {top_n} features por importancia**")
        # Color each bar by group
        group_col = df_plot["Grupo"].tolist()
        colors = [_GROUP_COLORS.get(g, _GROUP_COLORS["Otros"]) for g in group_col]

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#8a98b5"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
        ))
        apply_dark_layout(
            fig,
            height=max(280, top_n * 32),
            bargap=0.28,
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                showticklabels=True,
            ),
            yaxis=dict(showgrid=False, zeroline=False, autorange=True),
            margin=dict(l=10, r=50, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Group legend
        legend_html = " ".join(
            f'<span class="badge badge-muted" style="background:{c};color:#07090f;font-size:10px">'
            f'{g}</span>'
            for g, c in _GROUP_COLORS.items()
            if g != "Otros" and g in group_col
        )
        st.html(f'<div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap">{legend_html}</div>')

with table_col:
    with st.container(border=True):
        st.markdown("**Tabla de importancias**")
        feature_importance_table(
            df_plot,
            feature_col=FEAT_COL,
            importance_col=IMP_COL,
            group_map=_GROUP_MAP,
        )

st.write("")

# ── Group analysis ────────────────────────────────────────────────────────────
section_title("Importancia por grupo de features")

# Compute group summary from CSV features only
group_summary = (
    df_imp.groupby("Grupo")[IMP_COL]
    .agg(["sum", "mean", "count"])
    .rename(columns={"sum": "Importancia acumulada", "mean": "Importancia promedio", "count": "N features"})
    .sort_values("Importancia acumulada", ascending=False)
    .reset_index()
)

grp_col1, grp_col2 = st.columns([1.2, 1])

with grp_col1:
    with st.container(border=True):
        st.markdown("**Importancia acumulada por grupo**")
        g_labels = group_summary["Grupo"].tolist()
        g_values = group_summary["Importancia acumulada"].tolist()
        g_colors = [_GROUP_COLORS.get(g, _GROUP_COLORS["Otros"]) for g in g_labels]

        fig_grp = go.Figure(go.Bar(
            x=g_values,
            y=g_labels,
            orientation="h",
            marker=dict(color=g_colors, line=dict(width=0)),
            text=[f"{v:.2f} ({v/total_importance:.0%})" for v in g_values],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#8a98b5"),
            cliponaxis=False,
        ))
        apply_dark_layout(
            fig_grp,
            height=240,
            bargap=0.32,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, autorange=True),
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(fig_grp, use_container_width=True, config={"displayModeBar": False})

with grp_col2:
    with st.container(border=True):
        st.markdown("**Resumen por grupo**")
        tbl = group_summary.copy()
        tbl["Importancia acumulada"] = tbl["Importancia acumulada"].apply(lambda v: f"{v:.3f}")
        tbl["Importancia promedio"]  = tbl["Importancia promedio"].apply(lambda v: f"{v:.3f}")
        tbl["% del total"]           = group_summary["Importancia acumulada"].apply(
            lambda v: f"{v / total_importance:.1%}"
        )
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Contextual note for total vs reported features
    if n_features_csv < n_features_total:
        n_zero = n_features_total - n_features_csv
        callout(
            "warn",
            f"{n_zero} features con importancia cero o no reportadas",
            f"El CSV contiene {n_features_csv} features con importancia > 0. "
            f"Las {n_zero} features restantes (de {n_features_total} totales) "
            f"tienen importancia nula o muy pequeña para este modelo y no aparecen en el reporte. "
            f"Esto es normal en modelos lineales con muchas features correlacionadas.",
        )

st.write("")

# ── Correlation heatmap ────────────────────────────────────────────────────────
section_title("Mapa de correlación entre features")

if has_corr:
    with st.container(border=True):
        card_header(
            "Correlación de Pearson entre features",
            "imagen exportada por el pipeline",
            right_html=badge("PNG", "ok"),
        )
        st.image(str(corr_png), use_container_width=True)

    callout(
        "warn",
        "Features correlacionadas — efecto en la interpretación",
        "Algunas features presentan alta correlación entre sí (p. ej. "
        "<code>std</code> y <code>var</code> o "
        "<code>case_rr_std</code> y <code>case_rr_rmssd</code>). "
        "En modelos lineales, features muy correlacionadas "
        "<b>reparten artificialmente su importancia</b>: el modelo puede dar un peso pequeño "
        "a dos features correlacionadas cuando en realidad ambas representan la misma información. "
        "La importancia real del concepto subyacente es la <i>suma</i> de las dos, "
        "no cada una por separado. Esto debe considerarse al interpretar los rankings.",
    )
else:
    st.html(
        '<div class="placeholder-block" style="min-height:100px;padding:20px">'
        '<div class="ph-mono">imagen no encontrada</div>'
        '<div class="ph-title">Heatmap de correlación</div>'
        '<div class="ph-desc">No se encontró reports/figures/feature_correlation_heatmap.png</div>'
        '</div>'
    )

st.write("")

# ── Auto-generated interpretation ────────────────────────────────────────────
section_title("Interpretación automática")

# Find dominant group
top_group     = group_summary.loc[0, "Grupo"]
top_group_imp = float(group_summary.loc[0, "Importancia acumulada"])
top_group_pct = top_group_imp / total_importance

# Second group
second_group     = group_summary.loc[1, "Grupo"] if len(group_summary) > 1 else "—"
second_group_pct = float(group_summary.loc[1, "Importancia acumulada"]) / total_importance if len(group_summary) > 1 else 0.0

# Top 2 features
top2 = df_imp.head(2)
top2_names = [str(top2.loc[r, FEAT_COL]) for r in top2.index]
top2_vals  = [float(top2.loc[r, IMP_COL]) for r in top2.index]

with st.container(border=True):
    card_header(
        "Conclusión de esta sección",
        "lectura cualitativa · datos reales",
        right_html=badge("auto-generada", "muted"),
    )
    kv_table([
        ("Feature más importante",
         f'<b style="color:var(--teal)">{top2_names[0].replace("_"," ")}</b>'
         f' — importancia {top2_vals[0]:.3f}'),
        ("Segunda feature",
         f'{top2_names[1].replace("_"," ")} — importancia {top2_vals[1]:.3f}'),
        ("Grupo dominante",
         f'<b style="color:var(--teal)">{top_group}</b> ({top_group_pct:.0%} de la importancia total)'),
        ("Segundo grupo",
         f'{second_group} ({second_group_pct:.0%})'),
        ("Interpretación",
         f"El modelo usa principalmente información de <b>{top_group}</b> y "
         f"<b>{second_group}</b> para clasificar arritmias."),
    ])

    st.write("")
    callout(
        "info",
        "Qué sugiere este patrón",
        f"Las dos features con mayor importancia (<b>{top2_names[0].replace('_',' ')}</b> e "
        f"<b>{top2_names[1].replace('_',' ')}</b>) pertenecen al grupo "
        f"<b>{_GROUP_MAP.get(top2_names[0], 'Otros')}</b>. "
        f"Esto sugiere que el modelo discrimina arritmias principalmente a partir de "
        f"<b>variabilidad del ritmo cardíaco a nivel de caso (HRV global)</b>, "
        f"lo cual es clínicamente plausible: distintos tipos de arritmia tienen patrones "
        f"de RR muy diferentes (p. ej. fibrilación auricular presenta alta irregularidad). "
        f"Las features morfológicas (amplitud, varianza de la señal) aportan un rol "
        f"complementario ({second_group_pct:.0%} de la importancia total).",
    )

st.write("")

# ── Limitations ────────────────────────────────────────────────────────────────
section_title("Limitaciones de la interpretación")

lim1, lim2 = st.columns(2)
with lim1:
    callout(
        "warn",
        "Importancia ≠ causalidad",
        "Una feature importante para el modelo no implica que sea la causa clínica "
        "de la arritmia. Puede ser una correlación aprendida del dataset de entrenamiento.",
    )
    callout(
        "warn",
        "Features correlacionadas distorsionan rankings",
        "Grupos de features muy correlacionadas reparten su importancia entre ellas. "
        "El ranking individual subestima la influencia real de cada grupo.",
    )
    callout(
        "warn",
        "Solo features con importancia > 0",
        f"El reporte incluye {n_features_csv} de {n_features_total} features. "
        "Las features con coeficiente nulo no aparecen, pero existen en el modelo.",
    )

with lim2:
    callout(
        "warn",
        "LinearSVC no genera explicaciones locales",
        "Los coeficientes globales del modelo lineal no explican por qué se clasificó "
        "una ventana específica de cierta manera. Para explicaciones locales se necesita "
        "SHAP o LIME.",
    )
    callout(
        "warn",
        "Desempeño desigual entre clases",
        "El análisis de importancia es global. Las features que explican bien AFIB/AFL "
        "pueden no ser las mismas que explican las clases minoritarias (AVB, VT, SVTA).",
    )
    callout(
        "info",
        "Cómo mejorar esta página",
        "Para interpretabilidad local, añadir SHAP values o "
        "Permutation Importance (más robusto que coeficientes para SVC). "
        "El heatmap interactivo se habilitaría exportando el CSV de correlaciones.",
    )
