"""Página 07 — Demo de preprocesamiento ECG."""

import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.paths import PROJECT_ROOT, DEMO_DATA_DIR
from components.layout import page_header
from components.cards import callout, section_title
from components.charts import apply_dark_layout

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Pipeline backend detection ─────────────────────────────────────────────────

try:
    from src.pipeline import _HAS_PYVITAL
except Exception:
    _HAS_PYVITAL = False

if not _HAS_PYVITAL:
    st.html(
        '<div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.25);'
        'border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:12px;color:#fbbf24">'
        "<b>pyvital no encontrado</b> — el pipeline usará scipy como alternativa "
        "(resultados equivalentes).<br>"
        "Para usar pyvital, ejecuta en tu entorno activo:<br>"
        "<code style='font-size:11px'>"
        "python -m pip install pyvital<br>"
        "python -m streamlit run frontend/app/app.py"
        "</code>"
        "</div>"
    )

page_header(
    "Demo de preprocesamiento ECG",
    "Carga una señal ECG real y pásala por el pipeline para visualizar los efectos del preprocesamiento.",
)

# ── Demo loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def _load_demo_meta() -> dict:
    path = DEMO_DATA_DIR / "metadata.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _load_demo_signal(filename: str) -> np.ndarray:
    return pd.read_csv(DEMO_DATA_DIR / filename)["signal"].to_numpy(dtype=float)


_demo_meta = _load_demo_meta()

# ── Input ──────────────────────────────────────────────────────────────────────

section_title("1 · Fuente de datos ECG")

input_mode = st.radio(
    "Modo de entrada",
    ["Caso demo preestablecido", "Subir archivo CSV propio"],
    horizontal=True,
    label_visibility="collapsed",
)

signal_raw = None
original_fs = 500
source_label = ""

if input_mode == "Caso demo preestablecido":
    if not _demo_meta:
        st.error(
            f"No se encontraron archivos demo en {DEMO_DATA_DIR}. "
            "Ejecuta el script de generación para crearlos."
        )
    else:
        options = {
            f"{info['display_label']}  ·  caso {info['case_id']}": key
            for key, info in _demo_meta.items()
        }
        col_sel, col_dl = st.columns([4, 1])
        with col_sel:
            selected_opt = st.selectbox(
                "Caso demo",
                list(options.keys()),
                label_visibility="collapsed",
            )
        key = options[selected_opt]
        info = _demo_meta[key]
        signal_raw = _load_demo_signal(info["filename"])
        original_fs = info["fs"]
        source_label = info["display_label"]

        st.html(
            '<div style="font-size:11px;color:var(--fg-3);margin:4px 0 8px;font-family:var(--mono)">'
            f"ritmo: <b>{info['rhythm_label']}</b>"
            f"&nbsp;&nbsp;case_id: <b>{info['case_id']}</b>"
            f"&nbsp;&nbsp;fs: <b>{info['fs']} Hz</b>"
            f"&nbsp;&nbsp;duracion: <b>{info['n_samples'] / info['fs']:.0f} s</b>"
            f"&nbsp;&nbsp;origen: <b>VitalDB (señal real)</b>"
            "</div>"
        )

        csv_content = "signal\n" + "\n".join(f"{v:.6f}" for v in signal_raw)
        with col_dl:
            st.download_button(
                "Descargar CSV",
                data=csv_content,
                file_name=f"ecg_{key}_case{info['case_id']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    st.html(
        '<div style="font-size:12px;color:var(--fg-3);margin:6px 0 10px">'
        "Formato esperado: CSV con columna <code>signal</code>, un valor por fila. "
        "Descarga el archivo de ejemplo para ver el formato correcto."
        "</div>"
    )
    col_up, col_fs, col_dl = st.columns([3, 1, 1])
    with col_up:
        uploaded = st.file_uploader("Archivo ECG (CSV)", type=["csv"], label_visibility="collapsed")
    with col_fs:
        original_fs = int(st.number_input("fs (Hz)", min_value=50, max_value=5000, value=500, step=50))

    nsr_path = DEMO_DATA_DIR / "nsr.csv"
    with col_dl:
        if nsr_path.exists():
            example_data = nsr_path.read_bytes()
            example_name = "ecg_ejemplo_nsr_real.csv"
        else:
            example_data = b"signal\n"
            example_name = "ecg_ejemplo.csv"
        st.download_button(
            "Descargar ejemplo",
            data=example_data,
            file_name=example_name,
            mime="text/csv",
            use_container_width=True,
        )

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "signal" not in df_up.columns:
                st.error("El CSV debe contener una columna llamada 'signal'.")
            else:
                signal_raw = df_up["signal"].to_numpy(dtype=float)
                source_label = uploaded.name
                duration_s = len(signal_raw) / original_fs
                st.success(
                    f"{len(signal_raw):,} muestras cargadas · "
                    f"fs = {original_fs} Hz · "
                    f"duracion ≈ {duration_s:.1f} s"
                )
        except Exception as exc:
            st.error(f"Error al leer el archivo: {exc}")


# Limpia el cache del pipeline cuando cambia la señal
if signal_raw is not None:
    sig_hash = hash(signal_raw.tobytes())
    if st.session_state.get("_sig_hash") != sig_hash:
        st.session_state["_sig_hash"] = sig_hash
        st.session_state.pop("_processed", None)
        st.session_state.pop("_proc_error", None)


# ── Plot helper ────────────────────────────────────────────────────────────────

def _ecg_fig(
    signal: np.ndarray,
    fs: float,
    title: str,
    line_color: str = "#2dd4bf",
    fill_color: str = "rgba(45,212,191,0.04)",
    max_seconds: float = 10.0,
) -> go.Figure:
    n = min(len(signal), int(max_seconds * fs))
    t_axis = np.arange(n) / fs
    fig = go.Figure(go.Scatter(
        x=t_axis,
        y=signal[:n],
        mode="lines",
        line=dict(color=line_color, width=1.2),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="t=%{x:.3f} s  amp=%{y:.4f}<extra></extra>",
    ))
    apply_dark_layout(
        fig,
        title=dict(text=title, font=dict(size=12, color="#c8d2e5")),
        height=230,
        margin=dict(l=44, r=10, t=36, b=32),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            title=dict(text="tiempo (s)", font=dict(size=10)),
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            title=dict(text="amplitud", font=dict(size=10)),
            tickfont=dict(size=9),
        ),
    )
    return fig


# ── Raw signal ─────────────────────────────────────────────────────────────────

if signal_raw is not None:

    section_title("2 · Señal cruda")
    st.plotly_chart(
        _ecg_fig(
            signal_raw,
            original_fs,
            f"Señal cruda  ·  {source_label}  ·  fs = {original_fs} Hz",
        ),
        use_container_width=True,
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    section_title("3 · Preprocesamiento")

    backend_note = "pyvital" if _HAS_PYVITAL else "scipy (fallback)"
    st.html(
        '<div style="font-size:12px;color:var(--fg-3);padding:4px 0 10px">'
        "Pasos: <b>interpolacion NaN</b> &rarr; "
        "<b>resampleo a 500 Hz</b> &rarr; "
        "<b>filtro pasa banda 0.5&ndash;40 Hz</b> &rarr; "
        "<b>normalizacion Z-score</b>"
        f"&nbsp;&nbsp;<span style='color:var(--fg-4)'>backend: {backend_note}</span>"
        "</div>"
    )

    if st.button("Preprocesar ECG", type="primary"):
        with st.spinner("Ejecutando pipeline..."):
            try:
                from src.pipeline import preprocess_ecg
                result = preprocess_ecg(signal_raw, original_fs=original_fs)
                st.session_state["_processed"] = np.asarray(result, dtype=float)
                st.session_state["_proc_error"] = None
            except ImportError as exc:
                st.session_state["_proc_error"] = (
                    f"Dependencia no disponible: {exc}\n\n"
                    "Ejecuta en tu entorno:\n"
                    "  python -m pip install pyvital scipy scikit-learn\n"
                    "  python -m streamlit run frontend/app/app.py"
                )
                st.session_state["_processed"] = None
            except Exception as exc:
                st.session_state["_proc_error"] = f"Error en el pipeline: {exc}"
                st.session_state["_processed"] = None

    proc_error = st.session_state.get("_proc_error")
    processed = st.session_state.get("_processed")

    if proc_error:
        st.error(proc_error)
    elif processed is not None:
        st.plotly_chart(
            _ecg_fig(
                processed,
                500,
                "Señal procesada  ·  500 Hz  ·  filtrada  ·  normalizada",
                line_color="#4a8cff",
                fill_color="rgba(74,140,255,0.04)",
            ),
            use_container_width=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Muestras (cruda)", f"{len(signal_raw):,}")
        c2.metric("fs entrada", f"{original_fs} Hz")
        c3.metric("Muestras (procesada)", f"{len(processed):,}")
        c4.metric("fs salida", "500 Hz")
    else:
        st.html(
            '<div style="font-size:12px;color:var(--fg-3);padding:8px 0">'
            'Haz clic en "Preprocesar ECG" para ver el resultado del pipeline.'
            "</div>"
        )

    # ── Prediction placeholder ─────────────────────────────────────────────────

    section_title("4 · Prediccion de arritmia")

    callout(
        "warn",
        "Modelo pendiente",
        "El modelo de clasificacion de arritmias esta en desarrollo. "
        "Una vez disponible, recibira la señal procesada del paso anterior "
        "y mostrara el tipo de ritmo predicho con las probabilidades por clase.",
    )
