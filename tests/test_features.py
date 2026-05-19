"""Tests para `src.features` y validación metodológica de `src.modeling`."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    compute_rr_features,
    compute_rr_intervals,
    compute_time_features,
    compute_time_features_batch,
)
from src.modeling import assert_no_forbidden_features


def test_time_features_keys_present():
    win = np.linspace(-1.0, 1.0, num=200)
    feats = compute_time_features(win)
    expected = {
        "mean", "std", "var", "min", "max", "range",
        "median", "p25", "p75", "iqr", "skew", "kurtosis",
        "energy", "zero_crossing_rate", "abs_mean",
    }
    assert expected.issubset(feats.keys())


def test_time_features_constant_window_is_finite():
    win = np.zeros(100)
    feats = compute_time_features(win)
    for v in feats.values():
        assert np.isfinite(v)


def test_time_features_batch_shape():
    arr = np.random.RandomState(0).randn(5, 100)
    df = compute_time_features_batch(arr)
    assert df.shape[0] == 5
    assert df.isna().sum().sum() == 0


def test_rr_intervals_basic():
    rr = compute_rr_intervals(np.array([0.0, 1.0, 2.0, 3.5]))
    np.testing.assert_allclose(rr, [1.0, 1.0, 1.5])


def test_rr_features_handles_empty_input():
    feats = compute_rr_features(np.array([]))
    assert feats["rr_count"] == 0
    assert np.isnan(feats["rr_mean"])


def test_rr_features_basic_metrics():
    beats = np.arange(0, 10, 1.0)  # RR constante de 1s
    feats = compute_rr_features(beats)
    assert feats["rr_count"] == 9
    assert feats["rr_mean"] == pytest.approx(1.0)
    assert feats["rr_std"] == pytest.approx(0.0, abs=1e-12)


def test_assert_no_forbidden_features_passes_with_valid_columns():
    assert_no_forbidden_features(["mean", "std", "rr_mean"])


def test_assert_no_forbidden_features_blocks_beat_type():
    with pytest.raises(ValueError, match="beat_type"):
        assert_no_forbidden_features(["mean", "beat_type"])


def test_assert_no_forbidden_features_blocks_target():
    with pytest.raises(ValueError):
        assert_no_forbidden_features(["rhythm_label", "mean"])


def test_assert_no_forbidden_features_blocks_case_id():
    with pytest.raises(ValueError):
        assert_no_forbidden_features(["case_id", "mean"])
