import streamlit as st

st.set_page_config(page_title="ECG Arrhythmia ML · Pipeline", page_icon="🫀", layout="wide")

from components.layout import inject_css, sidebar_branding, placeholder_page
from utils.loaders import load_model_metadata

inject_css()
meta = load_model_metadata()
winner = meta.get("winner_model","—").replace("_"," ").title() if meta else "—"
f1_str = f"{meta.get('winner_test_f1_macro',0):.3f}" if meta else "—"
sidebar_branding(winner_model=winner, winner_f1=f1_str, pipeline_ok=meta is not None)

placeholder_page(
    title="Pipeline del proyecto",
    description=(
        "Recorrido completo desde los registros crudos de VitalDB hasta la evaluación final. "
        "11 etapas: carga, EDA, limpieza, ventaneo, features, split, entrenamiento, "
        "búsqueda de hiperparámetros, evaluación y demo."
    ),
    files_needed=[
        "models/model_artifacts_metadata.json",
        "reports/tables/model_comparison.csv",
    ],
    note="Se conectará en la siguiente fase con diagrama de nodos interactivo (Mermaid / Plotly Sankey).",
)
