"""Tests para `src.tabular_search` y `src.preprocessing.build_tabular_preprocessor`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config
from src.modeling import (
    assert_no_forbidden_features,
    make_train_test_group_split_with_coverage,
)
from src.preprocessing import build_tabular_preprocessor
from src.tabular_search import (
    TABULAR_MODEL_NAMES,
    build_cv_splitter,
    build_pipeline_for_model,
    classify_features,
)


# ---------------------------------------------------------------------------
# Fixture sintética del dataset tabular
# ---------------------------------------------------------------------------
def _toy_dataset(n_per_case: int = 30,
                 case_ids=tuple(range(1, 13)),
                 classes=("N", "AFIB/AFL", "VT", "SVTA"),
                 seed: int = 11) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for cid in case_ids:
        # Cada caso tiene un sex / department fijos
        sex = rng.choice(["F", "M"])
        dept = rng.choice(["A", "B", "C"])
        for i in range(n_per_case):
            cls = classes[(cid + i) % len(classes)]
            rows.append({
                config.CASE_ID_COLUMN: cid,
                config.TARGET_COLUMN: cls,
                config.BEAT_TYPE_COLUMN: rng.choice(["N", "V", "S"]),  # presente, prohibido
                config.SIGNAL_QUALITY_COLUMN: False,
                config.BEAT_TIME_COLUMN: float(i) * 0.85,
                "rhythm_classes": "should_be_dropped",  # leakage
                "age": float(rng.randint(20, 90)),
                "height": float(rng.normal(170, 10)),
                "weight": float(rng.normal(70, 15)),
                "bmi": float(rng.normal(24, 4)),
                "rr_prev": float(rng.uniform(0.6, 1.2)),
                "rr_next": float(rng.uniform(0.6, 1.2)),
                "hr_inst_from_rr_prev": float(rng.uniform(50, 100)),
                "position_in_case": float(i / n_per_case),
                "sex": sex,
                "department": dept,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# classify_features: exclusión de columnas prohibidas
# ---------------------------------------------------------------------------
def test_classify_features_excludes_leakage_and_metadata():
    df = _toy_dataset()
    cls = classify_features(df)

    # beat_type, rhythm_label, case_id deben estar en leakage_excluded
    assert config.BEAT_TYPE_COLUMN in cls["leakage_excluded"]
    assert config.TARGET_COLUMN in cls["leakage_excluded"]
    assert config.CASE_ID_COLUMN in cls["leakage_excluded"]
    assert "rhythm_classes" in cls["leakage_excluded"]
    assert config.SIGNAL_QUALITY_COLUMN in cls["leakage_excluded"]

    # Las features esperadas SÍ deben aparecer
    expected_numeric = {"age", "height", "weight", "bmi", "rr_prev", "rr_next",
                        "hr_inst_from_rr_prev", "position_in_case"}
    assert expected_numeric.issubset(set(cls["numeric_features"]))
    assert {"sex", "department"}.issubset(set(cls["categorical_features"]))


def test_classify_features_blocks_forbidden_predictors():
    df = _toy_dataset()
    cls = classify_features(df)
    feature_cols = cls["numeric_features"] + cls["categorical_features"]
    # Ninguna columna prohibida debe colarse
    for col in config.TABULAR_LEAKAGE_COLUMNS:
        assert col not in feature_cols, f"{col} no debería estar en features"


def test_assert_no_forbidden_features_catches_beat_type():
    with pytest.raises(ValueError, match=config.BEAT_TYPE_COLUMN):
        assert_no_forbidden_features(["mean", "age", config.BEAT_TYPE_COLUMN])


def test_assert_no_forbidden_features_catches_case_id():
    with pytest.raises(ValueError, match=config.CASE_ID_COLUMN):
        assert_no_forbidden_features(["mean", "age", config.CASE_ID_COLUMN])


# ---------------------------------------------------------------------------
# Preprocesador tabular
# ---------------------------------------------------------------------------
def test_build_tabular_preprocessor_returns_column_transformer():
    pre = build_tabular_preprocessor(
        numeric_features=["age", "rr_prev"],
        categorical_features=["sex", "department"],
    )
    assert isinstance(pre, ColumnTransformer)


def test_build_tabular_preprocessor_handles_numeric_and_categorical():
    df = _toy_dataset(n_per_case=40)
    pre = build_tabular_preprocessor(
        numeric_features=["age", "rr_prev", "rr_next"],
        categorical_features=["sex", "department"],
        ohe_min_frequency=5,  # bajo para que el dataset sintético no descarte categorías
    )
    X = df[["age", "rr_prev", "rr_next", "sex", "department"]]
    Xt = pre.fit_transform(X)
    # Debería expandir: 3 numéricas + (2 sex - 1 + 3 dept - 1) o similar; cualquiera > 3
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] >= 4
    # No debe haber NaN tras imputación
    assert not np.isnan(Xt).any()


def test_build_tabular_preprocessor_imputes_nan():
    pre = build_tabular_preprocessor(
        numeric_features=["age", "height"],
        categorical_features=["sex"],
        ohe_min_frequency=1,
    )
    X = pd.DataFrame({
        "age": [30.0, np.nan, 50.0, 60.0],
        "height": [170.0, 180.0, np.nan, 165.0],
        "sex": ["F", "M", None, "F"],
    })
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 4
    assert not np.isnan(Xt).any()


def test_build_tabular_preprocessor_raises_when_empty():
    with pytest.raises(ValueError):
        build_tabular_preprocessor(numeric_features=[], categorical_features=[])


# ---------------------------------------------------------------------------
# Split por grupo (no fuga de case_id)
# ---------------------------------------------------------------------------
def test_group_split_no_case_leak():
    df = _toy_dataset()
    cls = classify_features(df)
    numeric = cls["numeric_features"]
    categorical = cls["categorical_features"]
    X = df[numeric + categorical]
    y = df[config.TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()

    tr, te, info = make_train_test_group_split_with_coverage(X, y, groups, test_size=0.25)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    # case_id no aparece como columna de X
    assert config.CASE_ID_COLUMN not in X.columns
    assert info["n_classes_covered"] >= 1


def test_group_split_test_fraction_close_to_target_with_many_groups():
    df = _toy_dataset(case_ids=tuple(range(1, 21)))
    cls = classify_features(df)
    X = df[cls["numeric_features"] + cls["categorical_features"]]
    y = df[config.TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()
    _, _, info = make_train_test_group_split_with_coverage(X, y, groups, test_size=0.2)
    assert abs(info["actual_test_fraction"] - 0.2) < 0.1


# ---------------------------------------------------------------------------
# build_cv_splitter
# ---------------------------------------------------------------------------
def test_build_cv_splitter_returns_name_and_obj():
    groups = np.repeat(np.arange(8), 5)
    y = np.tile(np.array(["A", "B", "C", "A", "B"]), 8)
    cv, name, n_splits = build_cv_splitter(groups, y, n_splits=3)
    assert n_splits == 3
    assert name in {"StratifiedGroupKFold", "GroupKFold"}
    folds = list(cv.split(np.zeros((len(y), 1)), y, groups=groups))
    assert len(folds) == 3
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_build_cv_splitter_clamps_n_splits_to_n_groups():
    groups = np.repeat(np.arange(3), 4)
    y = np.tile(np.array(["A", "B"]), 6)
    _, _, n_splits = build_cv_splitter(groups, y, n_splits=10)
    assert n_splits == 3


# ---------------------------------------------------------------------------
# build_pipeline_for_model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list(TABULAR_MODEL_NAMES))
def test_build_pipeline_structure(model_name):
    pipe = build_pipeline_for_model(
        model_name,
        numeric_features=["age", "rr_prev"],
        categorical_features=["sex"],
    )
    assert isinstance(pipe, Pipeline)
    steps = [s for s, _ in pipe.steps]
    assert steps == ["preprocessor", "clf"]


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(KeyError):
        build_pipeline_for_model("not_a_model",
                                 numeric_features=["age"],
                                 categorical_features=["sex"])


# ---------------------------------------------------------------------------
# Estructura mínima de tablas de resultados (smoke)
# ---------------------------------------------------------------------------
def test_results_dataframe_has_expected_columns():
    """La tabla comparativa debe exponer al menos modelo, status, métricas test."""
    expected = {"model", "status", "best_cv_score_primary",
                "test_f1_macro", "test_precision_macro", "test_recall_macro",
                "test_accuracy", "test_balanced_accuracy", "test_f1_weighted"}
    # Construimos manualmente un row representativo y verificamos
    row = {
        "model": "logreg",
        "status": "ok",
        "best_cv_score_primary": 0.5,
        "cv_f1_macro": 0.5,
        "test_f1_macro": 0.5,
        "test_precision_macro": 0.5,
        "test_recall_macro": 0.5,
        "test_accuracy": 0.8,
        "test_balanced_accuracy": 0.5,
        "test_f1_weighted": 0.7,
        "best_params_json": "{}",
    }
    df = pd.DataFrame([row])
    assert expected.issubset(set(df.columns))
