"""Tests para `src.data_loading` y `src.preprocessing`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import (
    _case_id_from_filename,
    load_all_annotations,
    load_annotations_for_case,
    merge_metadata_and_annotations,
)
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
            "time_second": [0.5, 1.2, 0.3, 0.9, 0.7, 1.5],
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


# ---------------------------------------------------------------------------
# Extracción de case_id desde el nombre del archivo de anotaciones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, expected",
    [
        ("Annotation_file_1.csv", 1),
        ("Annotation_file_103.csv", 103),
        ("Annotation_file_1001.csv", 1001),
        ("annotation_file_42.csv", 42),
        ("Annotations_file_12.csv", 12),  # variante plural admitida
        ("metadata.csv", None),
        ("Annotation_file_abc.csv", None),
        ("Annotation_file_.csv", None),
    ],
)
def test_case_id_from_filename(name, expected):
    assert _case_id_from_filename(name) == expected


def _write_annotation_csv(directory: Path, case_id: int) -> None:
    df = pd.DataFrame(
        {
            "time_second": [0.5, 1.0],
            "beat_type": ["N", "N"],
            "rhythm_label": ["Sinus", "Sinus"],
            "bad_signal_quality": [False, False],
        }
    )
    df.to_csv(directory / f"Annotation_file_{case_id}.csv", index=False)


def test_load_annotations_for_case_injects_case_id(tmp_path):
    annotations_dir = tmp_path / "Annotation_Files"
    annotations_dir.mkdir()
    _write_annotation_csv(annotations_dir, case_id=42)

    df = load_annotations_for_case(42, physionet_dir=tmp_path)
    assert "case_id" in df.columns
    assert (df["case_id"] == 42).all()
    assert "time_second" in df.columns
    assert "rhythm_label" in df.columns


def test_load_annotations_for_case_disambiguates_by_exact_name(tmp_path):
    """case_id=1 no debe traer el archivo de case_id=10."""
    annotations_dir = tmp_path / "Annotation_Files"
    annotations_dir.mkdir()
    _write_annotation_csv(annotations_dir, case_id=1)
    _write_annotation_csv(annotations_dir, case_id=10)
    _write_annotation_csv(annotations_dir, case_id=100)

    df = load_annotations_for_case(1, physionet_dir=tmp_path)
    assert df["case_id"].unique().tolist() == [1]


def test_load_annotations_for_case_missing_raises(tmp_path):
    (tmp_path / "Annotation_Files").mkdir()
    with pytest.raises(FileNotFoundError):
        load_annotations_for_case(999, physionet_dir=tmp_path)


def test_load_all_annotations_concatenates_and_injects_case_id(tmp_path):
    annotations_dir = tmp_path / "Annotation_Files"
    annotations_dir.mkdir()
    for cid in (1, 10, 1001):
        _write_annotation_csv(annotations_dir, case_id=cid)
    # Archivo basura que no debe entrar al concat.
    (annotations_dir / "metadata.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")

    out = load_all_annotations(physionet_dir=tmp_path)
    assert "case_id" in out.columns
    assert sorted(out["case_id"].unique().tolist()) == [1, 10, 1001]
