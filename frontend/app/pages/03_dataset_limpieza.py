import streamlit as st

from components.layout import placeholder_page

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
