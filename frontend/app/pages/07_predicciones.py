import streamlit as st

st.set_page_config(page_title="ECG Arrhythmia ML · Predicciones", page_icon="🫀", layout="wide")

from components.layout import inject_css, sidebar_branding, placeholder_page
from utils.loaders import load_model_metadata

inject_css()
meta = load_model_metadata()
winner = meta.get("winner_model","—").replace("_"," ").title() if meta else "—"
f1_str = f"{meta.get('winner_test_f1_macro',0):.3f}" if meta else "—"
sidebar_branding(winner_model=winner, winner_f1=f1_str, pipeline_ok=meta is not None)

placeholder_page(
    title="Jugar con predicciones",
    description=(
        "Casos demo preseleccionados que ilustran el comportamiento del modelo. "
        "Selector por categoría (bien clasificado / error / clase minoritaria). "
        "Visualización de la ventana ECG, probabilidades por clase y features del vector de entrada."
    ),
    files_needed=[
        "reports/tables/test_predictions.csv  (FALTANTE — predicciones del pipeline)",
        "models/best_model_pipeline.joblib  (disponible)",
        "models/feature_columns.json  (disponible)",
        "data/demo/demo_windows.parquet  (FALTANTE — señales de demostración)",
    ],
    note=(
        "Sin test_predictions.csv se usarán 6 casos demo hardcoded del diseño de referencia. "
        "El visor de ECG usará señales sintéticas hasta que existan waveforms reales."
    ),
)
