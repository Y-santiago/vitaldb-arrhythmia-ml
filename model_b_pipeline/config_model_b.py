"""Configuration for the independent Model B binary rhythm pipeline.

Model B intentionally uses a small, fixed set of original tabular variables
for the binary task `normal_sinus` vs `arrhythmia_or_abnormal`.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

BASE_BINARY_DATASET_FILENAME = "binary_rhythm_modeling_dataset.parquet"
BASE_BINARY_DATASET_PATH = PROCESSED_DIR / BASE_BINARY_DATASET_FILENAME
MODEL_B_DATASET_FILENAME = "model_b_dataset.parquet"
MODEL_B_DATASET_PATH = PROCESSED_DIR / MODEL_B_DATASET_FILENAME

REPORTS_MODEL_B_DIR = PROJECT_ROOT / "reports" / "model_b"
TABLES_DIR = REPORTS_MODEL_B_DIR / "tables"
FIGURES_DIR = REPORTS_MODEL_B_DIR / "figures"
MODELS_MODEL_B_DIR = PROJECT_ROOT / "models" / "model_b"

TARGET_COLUMN = "rhythm_binary"
CASE_ID_COLUMN = "case_id"
POSITIVE_CLASS = "arrhythmia_or_abnormal"
NEGATIVE_CLASS = "normal_sinus"
CLASS_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]

RANDOM_STATE = 42

FEATURES_MODEL_B = [
    # RR y temporales
    "rr_prev",
    "rr_next",
    "hr_inst_from_rr_prev",
    "position_in_case",
    "rr_prev_rolling_mean_5",
    "rr_prev_rolling_std_5",
    "rr_prev_rolling_mean_20",
    "rr_prev_rolling_std_20",
    "rr_rmssd_5",
    "rr_rmssd_20",
    "rr_pnn50_5",
    "rr_pnn50_20",
    "local_hr_mean_5",
    "local_hr_mean_20",

    # Clinicas basicas
    "age",
    "sex",
    "bmi",
    "asa",
    "preop_htn",
    "preop_dm",

    # Laboratorios preoperatorios seleccionados
    "preop_hb",
    "preop_na",
    "preop_k",
    "preop_gluc",
    "preop_cr",
]

FORBIDDEN_COLUMNS_MODEL_B = {
    "rhythm_label",
    "rhythm_binary",
    "beat_type",
    "case_id",
    "caseid",
    "subjectid",
    "rhythm_classes",
    "bad_signal_quality",
    "bad_signal_quality_label",
    "source_file",
    "adm",
    "dis",
    "icu_days",
    "death_inhosp",
    "dx",
    "opname",
}

ADDITIONAL_EXCLUDED_COLUMNS_MODEL_B = {
    "analysis_start_time_sec",
    "analysis_end_time_sec",
    "analyzed_duration_sec",
    "total_beats",
    "caseend",
    "anestart",
    "aneend",
    "opstart",
    "opend",
    "intraop_phe",
    "intraop_eph",
    "intraop_epi",
    "intraop_ftn",
    "intraop_rocu",
    "intraop_crystalloid",
    "intraop_colloid",
    "intraop_rbc",
    "intraop_ffp",
}


def validate_model_b_feature_set(df) -> None:
    """Validate the required Model B schema and feature safety rules.

    Parameters
    ----------
    df:
        A pandas DataFrame containing the base binary rhythm dataset.

    Raises
    ------
    ValueError
        If required columns are missing, a forbidden column is configured as a
        feature, or the feature list is not exactly the expected 25 variables.
    """
    missing_schema = [
        col for col in (CASE_ID_COLUMN, TARGET_COLUMN)
        if col not in df.columns
    ]
    if missing_schema:
        raise ValueError(
            "Model B requires these columns in the dataset: "
            f"{missing_schema}. Expected at least '{CASE_ID_COLUMN}' and "
            f"'{TARGET_COLUMN}'."
        )

    if len(FEATURES_MODEL_B) != 25:
        raise ValueError(
            "FEATURES_MODEL_B must contain exactly 25 variables; "
            f"found {len(FEATURES_MODEL_B)}."
        )

    duplicate_features = sorted(
        {feature for feature in FEATURES_MODEL_B if FEATURES_MODEL_B.count(feature) > 1}
    )
    if duplicate_features:
        raise ValueError(
            "FEATURES_MODEL_B contains duplicate variables: "
            f"{duplicate_features}."
        )

    missing_features = [
        feature for feature in FEATURES_MODEL_B
        if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "The base binary dataset is missing Model B features: "
            f"{missing_features}."
        )

    forbidden_in_features = sorted(
        set(FEATURES_MODEL_B) & FORBIDDEN_COLUMNS_MODEL_B
    )
    if forbidden_in_features:
        raise ValueError(
            "Forbidden columns cannot be used as Model B predictors: "
            f"{forbidden_in_features}."
        )

    excluded_in_features = sorted(
        set(FEATURES_MODEL_B) & ADDITIONAL_EXCLUDED_COLUMNS_MODEL_B
    )
    if excluded_in_features:
        raise ValueError(
            "Administrative or intraoperative columns cannot be used in "
            f"Model B predictors: {excluded_in_features}."
        )
