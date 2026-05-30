"""Tests para la pipeline binaria `normal_sinus` vs `arrhythmia_or_abnormal`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config
from src.binary_search import (
    BINARY_NON_FEATURE_METADATA_COLUMNS,
    BINARY_SCORING_METRICS,
    build_binary_cv_splitter,
    build_binary_model_registry,
    classify_binary_features,
    make_binary_group_train_test_split_with_coverage,
    select_threshold_max_f1,
    select_threshold_youden_j,
)
from src.modeling import assert_no_forbidden_features
from src.preprocessing import build_tabular_preprocessor


# ---------------------------------------------------------------------------
# 1. map_rhythm_label_to_binary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label, expected",
    [
        ("N", config.BINARY_NEGATIVE_CLASS),
        ("AFIB/AFL", config.BINARY_POSITIVE_CLASS),
        ("AVB", config.BINARY_POSITIVE_CLASS),
        ("Patterned Atrial Ectopy", config.BINARY_POSITIVE_CLASS),
        ("Patterned Ventricular Ectopy", config.BINARY_POSITIVE_CLASS),
        ("SND", config.BINARY_POSITIVE_CLASS),
        ("SVTA", config.BINARY_POSITIVE_CLASS),
        ("VT", config.BINARY_POSITIVE_CLASS),
        ("WAP/MAT", config.BINARY_POSITIVE_CLASS),
        ("Noise", None),                # excluida
        ("Unclassifiable", None),       # excluida
        (None, None),
        (np.nan, None),
        ("", None),
        ("nan", None),
        ("None", None),
        ("LabelThatDoesNotExist", None),  # debe quedar excluida, no asignar auto
    ],
)
def test_map_rhythm_label_to_binary(label, expected):
    assert config.map_rhythm_label_to_binary(label) == expected


# ---------------------------------------------------------------------------
# 2. classify_binary_features: bloqueo de columnas prohibidas
# ---------------------------------------------------------------------------
def _toy_binary_df(n_per_case: int = 30,
                   case_ids=tuple(range(1, 13)),
                   seed: int = 11) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for cid in case_ids:
        sex = rng.choice(["F", "M"])
        dept = rng.choice(["A", "B", "C"])
        for i in range(n_per_case):
            # Mezcla de clases dentro del caso
            label = "N" if (cid + i) % 3 == 0 else "AFIB/AFL"
            binary = (config.BINARY_NEGATIVE_CLASS if label == "N"
                      else config.BINARY_POSITIVE_CLASS)
            rows.append({
                config.CASE_ID_COLUMN: cid,
                config.TARGET_COLUMN: label,
                config.BINARY_TARGET_COLUMN: binary,
                config.BEAT_TYPE_COLUMN: "N",  # prohibido
                config.BEAT_TIME_COLUMN: float(i) * 0.85,
                "rhythm_classes": "leak",       # prohibido
                "age": float(rng.randint(20, 90)),
                "rr_prev": float(rng.uniform(0.6, 1.2)),
                "rr_next": float(rng.uniform(0.6, 1.2)),
                "sex": sex,
                "department": dept,
            })
    return pd.DataFrame(rows)


def test_classify_excludes_forbidden_predictors():
    df = _toy_binary_df()
    cls = classify_binary_features(df)
    feature_cols = cls["numeric_features"] + cls["categorical_features"]
    for col in (
        config.BEAT_TYPE_COLUMN,
        config.CASE_ID_COLUMN,
        config.TARGET_COLUMN,
        config.BINARY_TARGET_COLUMN,
        "rhythm_classes",
    ):
        assert col not in feature_cols, f"{col} no debería estar en features"


def test_assert_no_forbidden_features_blocks_beat_type():
    with pytest.raises(ValueError, match=config.BEAT_TYPE_COLUMN):
        assert_no_forbidden_features(
            ["mean", config.BEAT_TYPE_COLUMN, "age"],
            forbidden=config.BINARY_LEAKAGE_COLUMNS,
        )


def test_assert_no_forbidden_features_blocks_case_id():
    with pytest.raises(ValueError, match=config.CASE_ID_COLUMN):
        assert_no_forbidden_features(
            ["mean", config.CASE_ID_COLUMN],
            forbidden=config.BINARY_LEAKAGE_COLUMNS,
        )


def test_assert_no_forbidden_features_blocks_rhythm_binary():
    with pytest.raises(ValueError, match=config.BINARY_TARGET_COLUMN):
        assert_no_forbidden_features(
            ["mean", config.BINARY_TARGET_COLUMN],
            forbidden=config.BINARY_LEAKAGE_COLUMNS,
        )


# ---------------------------------------------------------------------------
# 3. Preprocesador con numéricas y categóricas
# ---------------------------------------------------------------------------
def test_preprocessor_accepts_numeric_and_categorical():
    df = _toy_binary_df(n_per_case=40)
    pre = build_tabular_preprocessor(
        numeric_features=["age", "rr_prev", "rr_next"],
        categorical_features=["sex", "department"],
        ohe_min_frequency=2,
    )
    assert isinstance(pre, ColumnTransformer)
    Xt = pre.fit_transform(df[["age", "rr_prev", "rr_next", "sex", "department"]])
    assert Xt.shape[0] == len(df)
    assert not np.isnan(Xt).any()


# ---------------------------------------------------------------------------
# 4. Split por case_id: no overlap + ambas clases en ambos lados
# ---------------------------------------------------------------------------
def test_split_no_case_overlap():
    df = _toy_binary_df()
    cls = classify_binary_features(df)
    X = df[cls["numeric_features"] + cls["categorical_features"]]
    y = df[config.BINARY_TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()
    tr, te, info = make_binary_group_train_test_split_with_coverage(X, y, groups, test_size=0.25)
    assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_split_both_classes_in_train_and_test():
    df = _toy_binary_df()
    cls = classify_binary_features(df)
    X = df[cls["numeric_features"] + cls["categorical_features"]]
    y = df[config.BINARY_TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()
    _, _, info = make_binary_group_train_test_split_with_coverage(X, y, groups, test_size=0.25)
    assert info["binary_coverage_ok"], info


# ---------------------------------------------------------------------------
# 5. CV builder
# ---------------------------------------------------------------------------
def test_cv_splitter_clamps_n_splits_to_n_groups():
    groups = np.repeat(np.arange(3), 4)
    y = np.tile([config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS], 6)
    _, _, n_splits = build_binary_cv_splitter(groups, y, n_splits=10)
    assert n_splits == 3


def test_cv_splitter_returns_disjoint_folds():
    groups = np.repeat(np.arange(6), 5)
    y = np.tile([config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS], 15)
    cv, _, _ = build_binary_cv_splitter(groups, y, n_splits=3)
    folds = list(cv.split(np.zeros((len(y), 1)), y, groups=groups))
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


# ---------------------------------------------------------------------------
# 6. Registro de modelos
# ---------------------------------------------------------------------------
def test_registry_contains_required_baselines():
    reg = build_binary_model_registry()
    required = {
        "dummy_most_frequent", "logreg_balanced", "sgd_log_loss",
        "linear_svc_balanced", "hist_gradient_boosting",
        "random_forest_balanced", "extra_trees_balanced",
    }
    assert required.issubset(set(reg.keys()))


def test_registry_pipelines_have_preprocessor_and_clf():
    reg = build_binary_model_registry()
    for name, spec in reg.items():
        pipe = spec.pipeline_factory(["age", "rr_prev"], ["sex"])
        assert isinstance(pipe, Pipeline)
        names = [s for s, _ in pipe.steps]
        assert names == ["preprocessor", "clf"], f"{name}: pipeline mal estructurado"


def test_scoring_dict_includes_required_metrics():
    for required in ("balanced_accuracy", "accuracy", "f1", "precision", "recall"):
        assert required in BINARY_SCORING_METRICS


# ---------------------------------------------------------------------------
# 7. Threshold analysis (Youden J / F1)
# ---------------------------------------------------------------------------
def test_youden_j_threshold_recovers_perfect_split():
    # Score perfecto: scores < 0.5 son negativos, scores ≥ 0.5 son positivos.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr, j = select_threshold_youden_j(y_true, scores)
    assert 0.3 < thr <= 0.7
    assert j == pytest.approx(1.0, abs=1e-6)


def test_max_f1_threshold_for_imbalanced_case():
    # Positivos minoritarios; F1 debe encontrar el corte que los recupera.
    y_true = np.array([0, 0, 0, 0, 0, 1, 1])
    scores = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.85, 0.95])
    thr, f1 = select_threshold_max_f1(y_true, scores)
    assert thr > 0.4
    assert f1 == pytest.approx(1.0, abs=1e-6)


def test_specificity_calc_matches_definition():
    """Specificity = TN / (TN + FP), donde positiva = clase anormal."""
    # 4 normales (todos predichos como normales), 2 anormales (1 acertado).
    y_true = np.array([config.BINARY_NEGATIVE_CLASS] * 4 + [config.BINARY_POSITIVE_CLASS] * 2)
    y_pred = np.array([config.BINARY_NEGATIVE_CLASS] * 4 + [config.BINARY_POSITIVE_CLASS,
                                                           config.BINARY_NEGATIVE_CLASS])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
    )
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    assert specificity == 1.0
    assert sensitivity == 0.5
