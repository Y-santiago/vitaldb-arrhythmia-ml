# app.py

import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Detección de Arritmias con IA",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: #0f172a;
}

.hero {
    background: linear-gradient(
        135deg,
        #2563eb 0%,
        #06b6d4 100%
    );
    padding: 3rem;
    border-radius: 24px;
    color: white;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
}

.result-box {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

.status-badge {
    background: #dcfce7;
    color: #166534;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MOCK BACKEND
# =========================================================

def predict_ecg(signal):

    """Modelo DUMMY temporal
    """

    time.sleep(2)

    prediction = random.choice([
        "Normal",
        "Arrhythmia"
    ])

    confidence = round(
        random.uniform(0.80, 0.99),
        2
    )

    return {
        "prediction": prediction,
        "confidence": confidence
    }

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    # ❤️ VitalDB Arrhythmia ML
    """)

    st.markdown("""
    Panel de análisis ECG impulsado por IA
    para detección de arritmias.
    """)

    st.markdown("---")

    selected = option_menu(
        menu_title=None,
        options=[
            "Inicio",
            "Prediccion",
            "Visualizacion"
        ],
        icons=[
            "house",
            "activity",
            "graph-up",
            "bar-chart",
            "info-circle"
        ],
        default_index=0
    )

    st.markdown("---")

    st.markdown("""
    <div class="status-badge">
        Modelo Dummy Activo
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.info("Integración futura lista")

# =========================================================
# HOME PAGE
# =========================================================

if selected == "Inicio":

    st.markdown("""
    <div class="hero">

    <h1>
        VitalDB Arrhythmia ML
    </h1>

    <p style="font-size:18px;">
        Panel moderno de inteligencia artificial médica
        para análisis ECG y clasificación de arritmias.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "98.2%")

    with col2:
        st.metric("Recall", "96.4%")

    with col3:
        st.metric("Classes", "5")

    with col4:
        st.metric("Dataset", "MIT-BIH")

    st.write("")
    st.write("")

    st.subheader("🧠 Pipeline de Machine Learning")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("📈 ECG Signal")

    with c2:
        st.info("⚙️ Preprocesamiento")

    with c3:
        st.info("🤖 Modelo")

    with c4:
        st.info("✅ Predicción")

    st.write("")
    st.write("")

    st.subheader("🏥 Descripción")

    st.write("""
    Esta plataforma proporciona análisis ECG para 
    detección de arritmias usando machine learning.
    """)

# =========================================================
# PREDICTION PAGE
# =========================================================

elif selected == "Prediccion":

    st.title("❤️ ECG Prediccion")

    uploaded_file = st.file_uploader(
        "Subir archivo ECG CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("📄 Vista previa de datos")

            st.dataframe(df.head())

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(numeric_columns) == 0:

                st.error(
                    "No se encontraron columnas numéricas ECG."
                )

            else:

                selected_column = st.selectbox(
                    "Seleccionar columna ECG",
                    numeric_columns
                )

                if st.button("Ejecutar Inferencia"):

                    signal = df[selected_column].values

                    with st.spinner(
                        "Analizando señal ECG..."
                    ):

                        result = predict_ecg(signal)

                    st.success(
                        "Análisis completado exitosamente."
                    )

                    st.write("")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown("""
                        <div class="result-box">
                        <h3>Prediccion</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        st.metric(
                            "Diagnosis",
                            result["prediction"]
                        )

                    with col2:

                        st.markdown("""
                        <div class="result-box">
                        <h3>Confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        st.metric(
                            "Probabilidad",
                            f"{result['confidence'] * 100:.1f}%"
                        )

        except Exception as e:

            st.error(f"Error cargando archivo: {e}")

# =========================================================
# VISUALIZATION PAGE
# =========================================================

elif selected == "Visualizacion":

    st.title("📈 ECG Visualizacion")

    signal = np.sin(
        np.linspace(0, 20, 1000)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=signal,
            mode="lines",
            name="ECG Signal"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Time",
        yaxis_title="Amplitude"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
