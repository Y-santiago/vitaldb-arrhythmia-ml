"""Tests for the independent Model B pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_b_pipeline import config_model_b as cfg
from model_b_pipeline import evaluate_model_b, train_model_b
from model_b_pipeline.config_model_b import validate_model_b_feature_set
from model_b_pipeline.utils_model_b import (
    compute_binary_metrics,
    get_numeric_and_categorical_features,
)


def _synthetic_model_b_df(n_cases: int = 10, rows_per_case: int = 6) -> pd.DataFrame:
    rows = []
    rng = np.random.RandomState(7)
    for case_id in range(n_cases):
        for row_i in range(rows_per_case):
            label = cfg.NEGATIVE_CLASS if (case_id + row_i) % 2 == 0 else cfg.POSITIVE_CLASS
            rr = 0.80 + 0.03 * row_i + 0.02 * case_id
            rows.append({
                cfg.CASE_ID_COLUMN: case_id,
                cfg.TARGET_COLUMN: label,
                "rr_prev": rr,
                "rr_next": rr + 0.02,
                "hr_inst_from_rr_prev": 60.0 / rr,
                "position_in_case": row_i / rows_per_case,
                "rr_prev_rolling_mean_5": rr,
                "rr_prev_rolling_std_5": 0.01 + rng.rand() * 0.01,
                "rr_prev_rolling_mean_20": rr,
                "rr_prev_rolling_std_20": 0.02 + rng.rand() * 0.01,
                "rr_rmssd_5": 0.01 + rng.rand() * 0.01,
                "rr_rmssd_20": 0.02 + rng.rand() * 0.01,
                "rr_pnn50_5": float(row_i % 2),
                "rr_pnn50_20": float((row_i + case_id) % 2),
                "local_hr_mean_5": 60.0 / rr,
                "local_hr_mean_20": 60.0 / rr,
                "age": 40 + case_id,
                "sex": "F" if case_id % 2 == 0 else "M",
                "bmi": 22.0 + case_id * 0.1,
                "asa": float(1 + (case_id % 4)),
                "preop_htn": int(case_id % 3 == 0),
                "preop_dm": int(case_id % 4 == 0),
                "preop_hb": 12.0 + rng.rand(),
                "preop_na": 138.0 + rng.rand(),
                "preop_k": 4.0 + rng.rand() * 0.1,
                "preop_gluc": 100.0 + rng.rand() * 5,
                "preop_cr": 0.8 + rng.rand() * 0.1,
            })
    return pd.DataFrame(rows)


def _patch_output_paths(monkeypatch: pytest.MonkeyPatch, tmp_path):
    processed = tmp_path / "data" / "processed"
    reports = tmp_path / "reports" / "model_b"
    models = tmp_path / "models" / "model_b"
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "PROCESSED_DIR", processed)
    monkeypatch.setattr(cfg, "BASE_BINARY_DATASET_PATH", processed / cfg.BASE_BINARY_DATASET_FILENAME)
    monkeypatch.setattr(cfg, "MODEL_B_DATASET_PATH", processed / cfg.MODEL_B_DATASET_FILENAME)
    monkeypatch.setattr(cfg, "REPORTS_MODEL_B_DIR", reports)
    monkeypatch.setattr(cfg, "TABLES_DIR", reports / "tables")
    monkeypatch.setattr(cfg, "FIGURES_DIR", reports / "figures")
    monkeypatch.setattr(cfg, "MODELS_MODEL_B_DIR", models)
    processed.mkdir(parents=True, exist_ok=True)
    return processed, reports, models


def test_features_model_b_has_exactly_25_variables():
    assert len(cfg.FEATURES_MODEL_B) == 25
    assert len(set(cfg.FEATURES_MODEL_B)) == 25


def test_no_forbidden_columns_in_features_model_b():
    assert not (set(cfg.FEATURES_MODEL_B) & cfg.FORBIDDEN_COLUMNS_MODEL_B)


def test_validate_model_b_feature_set_fails_if_feature_is_missing():
    df = _synthetic_model_b_df()
    df = df.drop(columns=[cfg.FEATURES_MODEL_B[0]])
    with pytest.raises(ValueError, match="missing Model B features"):
        validate_model_b_feature_set(df)


def test_group_split_has_no_case_overlap():
    df = _synthetic_model_b_df(n_cases=12, rows_per_case=4)
    train_idx, test_idx, _ = train_model_b.make_group_train_test_split_with_coverage(df)
    train_cases = set(df.iloc[train_idx][cfg.CASE_ID_COLUMN])
    test_cases = set(df.iloc[test_idx][cfg.CASE_ID_COLUMN])
    assert train_cases.isdisjoint(test_cases)


def test_group_split_keeps_both_classes_in_train_and_test():
    df = _synthetic_model_b_df(n_cases=12, rows_per_case=4)
    train_idx, test_idx, _ = train_model_b.make_group_train_test_split_with_coverage(df)
    train_classes = set(df.iloc[train_idx][cfg.TARGET_COLUMN])
    test_classes = set(df.iloc[test_idx][cfg.TARGET_COLUMN])
    assert set(cfg.CLASS_LABELS).issubset(train_classes)
    assert set(cfg.CLASS_LABELS).issubset(test_classes)


def test_preprocessor_handles_numeric_features_and_sex_categorical():
    df = _synthetic_model_b_df()
    preprocessor, numeric, categorical = train_model_b.build_preprocessor(df)
    assert "sex" in categorical
    assert "rr_prev" in numeric
    transformed = preprocessor.fit_transform(df[cfg.FEATURES_MODEL_B], df[cfg.TARGET_COLUMN])
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= len(numeric) + 2


def test_metrics_compute_sensitivity_and_specificity_correctly():
    y_true = [
        cfg.NEGATIVE_CLASS,
        cfg.NEGATIVE_CLASS,
        cfg.POSITIVE_CLASS,
        cfg.POSITIVE_CLASS,
    ]
    y_pred = [
        cfg.NEGATIVE_CLASS,
        cfg.POSITIVE_CLASS,
        cfg.NEGATIVE_CLASS,
        cfg.POSITIVE_CLASS,
    ]
    metrics = compute_binary_metrics(y_true, y_pred)
    assert metrics["recall_abnormal_sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity_normal"] == pytest.approx(0.5)


def test_debug_mode_generates_expected_outputs(monkeypatch, tmp_path):
    _, reports, models = _patch_output_paths(monkeypatch, tmp_path)
    df = _synthetic_model_b_df(n_cases=12, rows_per_case=5)
    df.to_parquet(cfg.MODEL_B_DATASET_PATH, index=False)

    exit_code = train_model_b.main([
        "--debug",
        "--models",
        "dummy_most_frequent,logreg_balanced",
        "--n-jobs",
        "1",
    ])
    assert exit_code == 0
    exit_code = evaluate_model_b.main([])
    assert exit_code == 0

    expected_paths = [
        reports / "tables" / "model_b_train_test_split_summary.csv",
        reports / "tables" / "model_b_class_support_train_test.csv",
        reports / "tables" / "model_b_case_overlap_check.csv",
        reports / "tables" / "model_b_model_comparison_cv.csv",
        reports / "tables" / "model_b_model_comparison_test.csv",
        reports / "tables" / "model_b_best_hyperparameters.csv",
        reports / "tables" / "model_b_best_model_classification_report.csv",
        reports / "tables" / "model_b_confusion_matrix_absolute.csv",
        reports / "tables" / "model_b_confusion_matrix_normalized.csv",
        reports / "tables" / "model_b_threshold_analysis.csv",
        reports / "figures" / "model_b_confusion_matrix_absolute.png",
        reports / "figures" / "model_b_confusion_matrix_normalized.png",
        reports / "figures" / "model_b_roc_curve_best_model.png",
        reports / "figures" / "model_b_precision_recall_curve_best_model.png",
        reports / "MODEL_B_REPORT.md",
        reports / "NEXT_STEPS_MODEL_B.md",
        models / "model_b_best_pipeline.joblib",
        models / "model_b_feature_columns.json",
        models / "model_b_metadata.json",
    ]
    missing = [path for path in expected_paths if not path.exists()]
    assert missing == []
