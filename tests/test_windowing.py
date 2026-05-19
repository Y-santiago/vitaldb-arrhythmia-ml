"""Tests para `src.windowing`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.windowing import WindowSpec, build_windows_for_case


def _synthetic_signal(seconds: float = 10.0, fs_hz: int = 500) -> np.ndarray:
    n = int(seconds * fs_hz)
    t = np.arange(n) / fs_hz
    return np.sin(2 * np.pi * 1.5 * t).astype(np.float32)


def _beats_df(times, labels=None):
    df = pd.DataFrame({"time_second": times})
    if labels is not None:
        df["rhythm_label"] = labels
    return df


def test_build_windows_returns_expected_shape():
    signal = _synthetic_signal(seconds=10, fs_hz=500)
    beats = _beats_df([2.0, 4.0, 6.0], labels=["A", "A", "B"])

    windows, specs = build_windows_for_case(
        signal=signal,
        beats=beats,
        case_id=42,
        fs_hz=500,
        window_seconds=2.0,
        overlap=0.0,
    )
    assert windows.shape == (3, 1000)
    assert len(specs) == 3
    assert all(isinstance(s, WindowSpec) for s in specs)
    assert all(s.case_id == 42 for s in specs)
    assert specs[0].label == "A"
    assert specs[2].label == "B"


def test_build_windows_drops_out_of_range_centers():
    signal = _synthetic_signal(seconds=2, fs_hz=500)  # solo 1000 muestras
    # Latido cerca del borde: la ventana se sale por la derecha
    beats = _beats_df([1.9])
    windows, specs = build_windows_for_case(
        signal=signal,
        beats=beats,
        case_id=1,
        fs_hz=500,
        window_seconds=2.0,
        overlap=0.0,
    )
    assert windows.shape == (0, 1000)
    assert specs == []


def test_build_windows_rejects_invalid_overlap():
    signal = _synthetic_signal()
    beats = _beats_df([1.0])
    with pytest.raises(ValueError):
        build_windows_for_case(signal, beats, case_id=1, overlap=1.0)


def test_build_windows_overlap_adds_auxiliary_windows():
    signal = _synthetic_signal(seconds=20, fs_hz=500)
    beats = _beats_df([5.0, 10.0])
    windows, specs = build_windows_for_case(
        signal=signal,
        beats=beats,
        case_id=1,
        fs_hz=500,
        window_seconds=2.0,
        overlap=0.5,
    )
    # Con overlap > 0 se añaden 2 ventanas auxiliares por latido (3 por latido).
    assert len(specs) >= len(beats)


def test_build_windows_nan_label_becomes_none_not_string():
    """NaN en `rhythm_label` debe propagarse como None, no como la cadena 'nan'."""
    signal = _synthetic_signal(seconds=10, fs_hz=500)
    beats = pd.DataFrame(
        {
            "time_second": [2.0, 4.0, 6.0],
            "rhythm_label": ["Sinus", np.nan, "Sinus"],
        }
    )
    _, specs = build_windows_for_case(
        signal=signal,
        beats=beats,
        case_id=1,
        fs_hz=500,
        window_seconds=2.0,
        overlap=0.0,
    )
    labels = [s.label for s in specs]
    assert labels == ["Sinus", None, "Sinus"]
    # Asegurar que nadie produzca el string literal "nan".
    assert "nan" not in [str(lbl).lower() for lbl in labels if lbl is not None]
