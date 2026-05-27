import streamlit as st

st.set_page_config(page_title="ECG Arrhythmia ML · Dataset", page_icon="🫀", layout="wide")

from components.layout import inject_css, sidebar_branding, placeholder_page
from utils.loaders import load_model_metadata

inject_css()
meta = load_model_metadata()
winner = meta.get("winner_model","—").replace("_"," ").title() if meta else "—"
f1_str = f"{meta.get('winner_test_f1_macro',0):.3f}" if meta else "—"
sidebar_branding(winner_model=winner, winner_f1=f1_str, pipeline_ok=meta is not None)

placeholder_page(
    title="Dataset y limpieza",
    description=(
        "Auditoría de calidad sobre la VitalDB Arrhythmia Database. "
        "Estadísticas de señales auditadas, ventanas generadas, ventanas válidas, "
        "razones de descarte (NaN, saturación, SNR bajo, pulso ausente) y "
        "distribución de clases."
    ),
    files_needed=[
        "reports/tables/best_model_classification_report.csv  (distribución de clases)",
        "data/demo/demo_windows.parquet  (opcional — para histogramas reales)",
    ],
    note="Los histogramas de duración y SNR se generarán desde metadata; la distribución de clases usa el classification report.",
)
