"""
ECG preprocessing pipeline
for VitalDB Arrhythmia adaptation.
"""

from __future__ import annotations

import numpy as np
import pyvital

from scipy.signal import resample

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# =========================================================
# CONFIG
# =========================================================

TARGET_FS = 500

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0



def interpolate_nans(signal):

    signal = np.asarray(signal).flatten()

    return pyvital.interp_undefined(
        signal
    )



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

    return pyvital.band_pass(
        signal,
        srate=fs,
        fl=lowcut,
        fh=highcut
    )



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
        pipeline.transform(signal)
    )

    return processed_signal
