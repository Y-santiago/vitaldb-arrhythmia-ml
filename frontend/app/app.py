"""ECG Arrhythmia ML — Streamlit App entry point.

Run with:
    streamlit run frontend/app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ECG Arrhythmia ML",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.layout import inject_css, sidebar_branding
from utils.loaders import load_model_metadata

inject_css()

# Load metadata for sidebar status (best-effort)
meta = load_model_metadata()
winner  = meta.get("winner_model", "—").replace("_", " ").title() if meta else "—"
f1_val  = meta.get("winner_test_f1_macro") if meta else None
f1_str  = f"{f1_val:.3f}" if f1_val is not None else "—"

sidebar_branding(winner_model=winner, winner_f1=f1_str, pipeline_ok=meta is not None)

# ── Portada ──────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:60vh;text-align:center;
                padding:48px 24px;">

      <div style="width:56px;height:56px;border-radius:14px;margin-bottom:24px;
                  background:radial-gradient(circle at 30% 30%,rgba(45,212,191,.35),transparent 60%),
                             linear-gradient(135deg,#0e2238,#15324f);
                  border:1px solid #243150;display:grid;place-items:center;">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
             stroke="#2dd4bf" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M2 12 H6 L8 8 L11 16 L14 6 L16 12 H22"/>
        </svg>
      </div>

      <h1 style="font-size:32px;font-weight:600;color:#f1f5fb;
                 letter-spacing:-.015em;margin:0 0 12px;line-height:1.2;
                 font-family:'IBM Plex Sans',sans-serif;">
        ECG Arrhythmia ML
      </h1>

      <p style="color:#8a98b5;font-size:15px;max-width:520px;margin:0 0 28px;line-height:1.6;">
        Clasificación multiclase de ritmos intraoperatorios
        usando señales ECG de la VitalDB Arrhythmia Database.
      </p>

      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-bottom:32px;">
        <span class="badge badge-muted">VitalDB · 2026</span>
        <span class="badge badge-info">Machine Learning</span>
        <span class="badge badge-muted">scikit-learn</span>
        <span class="badge badge-muted">10 clases</span>
        <span class="badge badge-warn">Demo académica</span>
      </div>

      <p style="color:#5d6c8c;font-size:13px;font-family:'IBM Plex Mono',monospace;">
        Usa el menú lateral para navegar entre secciones →
      </p>

    </div>
    """,
    unsafe_allow_html=True,
)

# Quick stats row if metadata loaded
if meta:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modelo ganador", winner)
    with col2:
        st.metric("F1-macro (test)", f1_str)
    with col3:
        n_feat = meta.get("n_features", "—")
        st.metric("Features", str(n_feat))
    with col4:
        n_groups = meta.get("n_train_groups", 0) + meta.get("n_test_groups", 0)
        st.metric("Casos totales", str(n_groups))
