import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.paths import (
    MODELS_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
)


def file_exists(path: Path) -> bool:
    return Path(path).exists()


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    path = REPORT_TABLES_DIR / "model_comparison.csv"
    if not file_exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_classification_report() -> pd.DataFrame | None:
    path = REPORT_TABLES_DIR / "best_model_classification_report.csv"
    if not file_exists(path):
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    path = REPORT_TABLES_DIR / "best_model_feature_importance.csv"
    if not file_exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_metadata() -> dict | None:
    path = MODELS_DIR / "model_artifacts_metadata.json"
    if not file_exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_feature_columns() -> list | None:
    path = MODELS_DIR / "feature_columns.json"
    if not file_exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource
def load_model():
    path = MODELS_DIR / "best_model_pipeline.joblib"
    if not file_exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def confusion_matrix_figure_path() -> Path | None:
    path = REPORT_FIGURES_DIR / "best_model_confusion_matrix.png"
    return path if file_exists(path) else None


@st.cache_data
def load_confusion_matrix_csv() -> "pd.DataFrame | None":
    """Load reports/tables/confusion_matrix.csv if it exists.

    Expected format: real_label, predicted_label, count
    Returns None if the file has not been exported yet.
    """
    path = REPORT_TABLES_DIR / "confusion_matrix.csv"
    if not file_exists(path):
        return None
    return pd.read_csv(path)


def correlation_figure_path() -> Path | None:
    path = REPORT_FIGURES_DIR / "feature_correlation_heatmap.png"
    return path if file_exists(path) else None
