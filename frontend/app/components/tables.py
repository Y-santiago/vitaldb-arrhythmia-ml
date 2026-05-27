"""Table rendering helpers."""

import pandas as pd
import streamlit as st


def safe_dataframe(df: pd.DataFrame | None, title: str = "") -> None:
    """Render a DataFrame or a graceful missing-data notice."""
    if df is None:
        st.html(
            '<div class="callout-block callout-warn">'
            '<span class="callout-title">⚠ Datos no disponibles</span>'
            'El archivo de datos requerido no fue encontrado. '
            'Ejecuta el pipeline para generarlo.'
            '</div>'
        )
        return
    if title:
        st.caption(title)
    st.dataframe(df, use_container_width=True)


def format_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format model_comparison.csv for display."""
    cols_rename = {
        "model":                 "Modelo",
        "test_f1_macro":         "F1-macro",
        "test_precision_macro":  "Precision",
        "test_recall_macro":     "Recall",
        "test_accuracy":         "Accuracy",
        "fit_time_seconds":      "Tiempo (s)",
        "cv_f1_macro":           "CV F1-macro",
        "status":                "Estado",
    }
    display_cols = [c for c in cols_rename if c in df.columns]
    out = df[display_cols].rename(columns=cols_rename).copy()

    float_cols = ["F1-macro", "Precision", "Recall", "Accuracy", "CV F1-macro"]
    for col in float_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—"
            )

    if "Tiempo (s)" in out.columns:
        out["Tiempo (s)"] = out["Tiempo (s)"].apply(
            lambda v: f"{v:.0f} s" if pd.notna(v) else "—"
        )

    return out


def style_model_table(df: pd.DataFrame, winner_model: str = ""):
    """Apply background highlighting to the winner row."""

    def highlight_winner(row):
        is_winner = str(row.get("Modelo", "")).lower() == winner_model.lower()
        return ["background-color: rgba(45,212,191,0.08); color: #f1f5fb" if is_winner else "" for _ in row]

    return df.style.apply(highlight_winner, axis=1)


def model_comparison_table(
    df: pd.DataFrame,
    winner_model_id: str = "",
) -> None:
    """Format and render the full model comparison table with winner highlighted.

    Args:
        df: raw model_comparison.csv DataFrame.
        winner_model_id: raw model id (e.g. 'linear_svc') used for highlighting.
    """
    if df is None or df.empty:
        safe_dataframe(None)
        return

    formatted = format_model_table(df)

    # Add Status badge column using raw data if available
    if "status" in df.columns:
        def _status_cell(s):
            s = str(s).lower()
            if s == "ok":
                return "✓ ok"
            return f"✗ {s}"

        formatted.insert(1, "Estado", df["status"].apply(_status_cell))

    # Normalise winner name for comparison
    winner_nice = winner_model_id.replace("_", " ").title()

    def _styler(row):
        is_winner = str(row.get("Modelo", "")).lower() == winner_nice.lower()
        if is_winner:
            return [
                "background-color: rgba(45,212,191,0.08); "
                "color: #f1f5fb; font-weight: 600"
            ] * len(row)
        return [""] * len(row)

    styled = formatted.style.apply(_styler, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def class_report_table(
    df: pd.DataFrame,
    sort_col: str = "f1-score",
    ascending: bool = False,
) -> None:
    """Render best_model_classification_report.csv with F1-coded row colors.

    df must have index = class name and columns precision, recall, f1-score, support.
    Summary rows (accuracy, macro avg, weighted avg) are excluded.
    """
    if df is None or df.empty:
        safe_dataframe(None)
        return

    SUMMARY = {"accuracy", "macro avg", "weighted avg"}
    df_classes = df[~df.index.isin(SUMMARY)].copy()

    # Sort
    valid_cols = [c for c in [sort_col] if c in df_classes.columns]
    if valid_cols:
        df_classes = df_classes.sort_values(valid_cols[0], ascending=ascending)

    # Build display frame
    out = df_classes.copy()
    out.index.name = "Clase"

    for col in ["precision", "recall", "f1-score"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    if "support" in out.columns:
        out["support"] = out["support"].apply(
            lambda v: f"{int(v):,}" if pd.notna(v) else "—"
        )

    out = out.rename(columns={
        "precision": "Precision",
        "recall":    "Recall",
        "f1-score":  "F1",
        "support":   "Soporte",
    })

    def _row_color(row):
        try:
            f1_val = float(df_classes.loc[
                df_classes.index[list(out.index).index(row.name)], "f1-score"
            ])
        except Exception:
            return [""] * len(row)
        if f1_val >= 0.70:
            color = "rgba(52,211,153,0.08)"
        elif f1_val >= 0.30:
            color = "rgba(74,140,255,0.06)"
        else:
            color = "rgba(248,113,113,0.08)"
        return [f"background-color:{color}"] * len(row)

    styled = out.style.apply(_row_color, axis=1)
    st.dataframe(styled, use_container_width=True)


def feature_importance_table(
    df: pd.DataFrame,
    feature_col: str,
    importance_col: str,
    group_map: dict | None = None,
) -> None:
    """Render feature importance with rank, optional group column, and importance-coded colors.

    Top 2 rows get teal highlight; next 3 get blue; rest are neutral.
    """
    if df is None or df.empty:
        safe_dataframe(None)
        return

    out = df[[feature_col, importance_col]].copy().reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))

    if group_map:
        out["Grupo"] = out[feature_col].map(lambda f: group_map.get(f, "Otros"))

    out = out.rename(columns={feature_col: "Feature", importance_col: "Importancia"})
    imp_raw = out["Importancia"].copy()
    out["Importancia"] = out["Importancia"].apply(
        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
    )

    def _row_color(row):
        rank = int(row["Rank"])
        if rank <= 2:
            return ["background-color:rgba(45,212,191,0.10)"] * len(row)
        if rank <= 5:
            return ["background-color:rgba(74,140,255,0.07)"] * len(row)
        return [""] * len(row)

    styled = out.style.apply(_row_color, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
