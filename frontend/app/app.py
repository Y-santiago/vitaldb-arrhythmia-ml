"""ECG Arrhythmia ML — Streamlit App entry point.

Run with:
    streamlit run frontend/app/app.py
    python -m streamlit run frontend/app/app.py  (if pyvital not found)
"""

import streamlit as st

st.set_page_config(
    page_title="ECG Arrhythmia ML",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.navigation MUST be called before any other Streamlit command (besides set_page_config).
pages = [
    st.Page("pages/07_predicciones.py", title="Probar ECG",              icon="🫀"),
    st.Page("pages/01_inicio.py",        title="Inicio",                  icon="🏠"),
    st.Page("pages/02_pipeline.py",      title="Pipeline del proyecto",   icon="🧭"),
    st.Page("pages/03_dataset_limpieza.py", title="Dataset y limpieza",   icon="🧹"),
    st.Page("pages/04_rendimiento_modelo.py", title="Rendimiento del modelo", icon="📊"),
    st.Page("pages/05_evaluacion_clase.py",   title="Evaluación por clase",   icon="🧬"),
    st.Page("pages/06_matriz_confusion.py",   title="Matriz de confusión",    icon="🎯"),
    st.Page("pages/08_interpretabilidad.py",  title="Interpretabilidad",      icon="🔎"),
    st.Page("pages/09_conclusiones.py",       title="Conclusiones",           icon="✅"),
]
pg = st.navigation(pages)

# ── Shared setup (runs on every page) ────────────────────────────────────────
from components.layout import inject_css, sidebar_branding
from utils.loaders import load_model_metadata

inject_css()

_meta   = load_model_metadata()
_winner = _meta.get("winner_model", "—").replace("_", " ").title() if _meta else "—"
_f1     = f"{_meta.get('winner_test_f1_macro', 0):.3f}" if _meta else "—"
sidebar_branding(winner_model=_winner, winner_f1=_f1, pipeline_ok=_meta is not None)

pg.run()
