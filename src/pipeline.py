"""
ECG preprocessing pipeline
for VitalDB Arrhythmia adaptation.
"""

from __future__ import annotations

import numpy as np

from scipy.signal import resample, butter, filtfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# =========================================================
# CONFIG
# =========================================================

TARGET_FS = 500

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0


# Pyvital is the primary backend; scipy is used as fallback when pyvital
# is not installed in the active environment (e.g. Anaconda + Streamlit).
try:
    import pyvital as _pyvital
    _HAS_PYVITAL = True
except ImportError:
    _HAS_PYVITAL = False


# =========================================================
# PRIVATE FALLBACKS (scipy-only)
# =========================================================

def _interp_nans_scipy(signal: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(signal)
    if not nan_mask.any():
        return signal
    x = np.arange(len(signal))
    signal = signal.copy()
    signal[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], signal[~nan_mask])
    return signal


def _bandpass_scipy(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


# =========================================================
# PUBLIC API
# =========================================================

def interpolate_nans(signal):

    signal = np.asarray(signal).flatten()

    if _HAS_PYVITAL:
        return _pyvital.interp_undefined(signal)

    return _interp_nans_scipy(signal)



def resample_ecg(
    signal,
    original_fs,
    target_fs=TARGET_FS
):

    duration_seconds = (
        len(signal) / original_fs
    )

    target_length = int(
        duration_seconds * target_fs
    )

    return resample(
        signal,
        target_length
    )



def bandpass_filter(
    signal,
    fs=TARGET_FS,
    lowcut=LOWCUT_HZ,
    highcut=HIGHCUT_HZ
):

    if _HAS_PYVITAL:
        return _pyvital.band_pass(
            signal,
            srate=fs,
            fl=lowcut,
            fh=highcut,
        )

    return _bandpass_scipy(signal, fs, lowcut, highcut)



def normalize_ecg(signal):

    mean = np.mean(signal)

    std = np.std(signal)

    if std == 0:
        std = 1e-8

    return (
        signal - mean
    ) / std


def build_ecg_pipeline(
    original_fs
):

    pipeline = Pipeline([

        (
            "nan_interpolation",

            FunctionTransformer(
                interpolate_nans
            )
        ),

        (
            "resampling",

            FunctionTransformer(
                lambda x: resample_ecg(
                    x,
                    original_fs=original_fs
                )
            )
        ),

        (
            "bandpass_filter",

            FunctionTransformer(
                bandpass_filter
            )
        ),

        (
            "normalization",

            FunctionTransformer(
                normalize_ecg
            )
        )

    ])

    return pipeline


def preprocess_ecg(
    signal,
    original_fs
):

    pipeline = build_ecg_pipeline(
        original_fs
    )

    processed_signal = (
        pipeline.fit_transform(signal)
    )

    return processed_signal
