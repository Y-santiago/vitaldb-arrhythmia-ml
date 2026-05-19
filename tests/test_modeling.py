"""Tests para `src.modeling` (helpers de split y construcción del Pipeline)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.modeling import (
    assert_no_forbidden_features,
    build_logreg_pipeline,
    build_rf_pipeline,
    make_group_kfold,
    safe_n_splits,
)


# ---------------------------------------------------------------------------
# safe_n_splits
# ---------------------------------------------------------------------------
def test_safe_n_splits_trims_to_n_groups():
    groups = np.array([1, 1, 2, 2, 3, 3])
    assert safe_n_splits(5, groups) == 3


def test_safe_n_splits_keeps_value_if_enough_groups():
    groups = np.array([1, 2, 3, 4, 5, 6])
    assert safe_n_splits(3, groups) == 3


def test_safe_n_splits_raises_with_less_than_two_groups():
    with pytest.raises(ValueError):
        safe_n_splits(5, np.array([1, 1, 1]))


# ---------------------------------------------------------------------------
# make_group_kfold (usa safe_n_splits internamente)
# ---------------------------------------------------------------------------
def test_make_group_kfold_disjoint_groups_per_fold():
    X = np.arange(60).reshape(-1, 2)
    y = np.tile([0, 1], 15)
    groups = np.repeat(np.arange(6), 5)

    folds = list(make_group_kfold(X, y, groups, n_splits=3))
    assert len(folds) == 3
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_make_group_kfold_trims_n_splits_to_n_groups():
    X = np.arange(40).reshape(-1, 2)
    y = np.tile([0, 1], 10)
    groups = np.repeat(np.arange(3), 7)[:20]  # 3 grupos
    folds = list(make_group_kfold(X, y, groups, n_splits=5))
    assert len(folds) == 3  # se recorta de 5 a 3


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def test_logreg_pipeline_structure():
    pipe = build_logreg_pipeline()
    names = [step for step, _ in pipe.steps]
    assert names == ["imputer", "scaler", "clf"]
    assert isinstance(pipe.named_steps["imputer"], SimpleImputer)
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["clf"], LogisticRegression)
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_rf_pipeline_structure():
    pipe = build_rf_pipeline()
    names = [step for step, _ in pipe.steps]
    assert names == ["imputer", "clf"]  # RF no requiere scaler


def test_pipeline_handles_nan_in_features():
    """El Imputer debe permitir que el Pipeline corra aunque X tenga NaN."""
    rng = np.random.RandomState(0)
    X = rng.randn(40, 4)
    X[0, 0] = np.nan
    X[5, 2] = np.nan
    y = np.array(["A"] * 20 + ["B"] * 20)

    pipe = build_logreg_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (40,)


def test_assert_no_forbidden_features_message_mentions_offender():
    with pytest.raises(ValueError, match="beat_type"):
        assert_no_forbidden_features(["mean", "beat_type", "std"])
