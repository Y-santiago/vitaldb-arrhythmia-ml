"""Tests para `src.data_loading` y `src.preprocessing`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loading import merge_metadata_and_annotations
from src.preprocessing import (
    apply_basic_filters,
    drop_exact_duplicates,
    exclude_bad_signal_quality,
    exclude_rhythm_labels,
    validate_columns,
)


def _sample_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [1, 1, 2, 2, 3, 3],
            "beat_time": [0.5, 1.2, 0.3, 0.9, 0.7, 1.5],
            "rhythm_label": ["Sinus", "Sinus", "Noise", "AF", "Sinus", "Sinus"],
            "beat_type": ["N", "N", "N", "V", "N", "N"],
            "bad_signal_quality": [False, False, False, True, False, False],
        }
    )


def _sample_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [1, 2, 3],
            "age": [40, 55, 33],
            "sex": ["F", "M", "F"],
        }
    )


def test_validate_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        validate_columns(df, ["a", "b"])


def test_exclude_rhythm_labels_drops_noise():
    df = _sample_annotations()
    out = exclude_rhythm_labels(df)
    assert "Noise" not in out["rhythm_label"].unique()
    assert len(out) == 5


def test_exclude_bad_signal_quality_filters_true_rows():
    df = _sample_annotations()
    out = exclude_bad_signal_quality(df)
    assert out["bad_signal_quality"].any() is False or (~out["bad_signal_quality"]).all()
    assert len(out) == 5


def test_exclude_bad_signal_quality_accepts_string_flags():
    df = pd.DataFrame(
        {"rhythm_label": ["a", "b"], "bad_signal_quality": ["true", "false"]}
    )
    out = exclude_bad_signal_quality(df)
    assert len(out) == 1
    assert out.iloc[0]["rhythm_label"] == "b"


def test_drop_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    out = drop_exact_duplicates(df)
    assert len(out) == 2


def test_apply_basic_filters_excludes_noise_and_bad_quality():
    df = _sample_annotations()
    out = apply_basic_filters(df)
    assert "Noise" not in out["rhythm_label"].unique()
    assert (~out["bad_signal_quality"].astype(bool)).all()


def test_merge_metadata_and_annotations_inner():
    md = _sample_metadata()
    ann = _sample_annotations()
    merged = merge_metadata_and_annotations(md, ann, on="case_id", how="inner")
    assert "age" in merged.columns
    assert set(merged["case_id"].unique()) == {1, 2, 3}


def test_merge_metadata_and_annotations_missing_key_raises():
    md = pd.DataFrame({"foo": [1]})
    ann = _sample_annotations()
    with pytest.raises(KeyError):
        merge_metadata_and_annotations(md, ann, on="case_id")
