"""Extracción de features sobre ventanas de ECG.

Incluye dos grupos de features:

* **Temporales**: estadísticas por ventana (media, std, varianza, rango,
  percentiles, energía, cruces por cero, etc.).
* **RR**: derivadas de los intervalos entre latidos consecutivos (media,
  desviación estándar, RMSSD, pNN50).

Nota metodológica: la columna ``beat_type`` está prohibida como predictora.
Estos extractores trabajan sobre la señal y los tiempos de latido, no sobre
etiquetas categóricas del latido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Features temporales sobre la ventana
# ---------------------------------------------------------------------------
_TIME_FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "var",
    "min",
    "max",
    "range",
    "median",
    "p25",
    "p75",
    "iqr",
    "skew",
    "kurtosis",
    "energy",
    "zero_crossing_rate",
    "abs_mean",
)


def _safe_skew(x: np.ndarray) -> float:
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / (s ** 3))


def _safe_kurtosis(x: np.ndarray) -> float:
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 4).mean() / (s ** 4) - 3.0)


def compute_time_features(window: np.ndarray) -> dict[str, float]:
    """Calcula features estadísticas sobre una única ventana 1-D.

    Parameters
    ----------
    window : numpy.ndarray
        Ventana de ECG (1-D).

    Returns
    -------
    dict[str, float]
        Diccionario con los nombres de :data:`_TIME_FEATURE_NAMES`.
    """
    x = np.asarray(window, dtype=float).ravel()
    if x.size == 0:
        return {name: np.nan for name in _TIME_FEATURE_NAMES}

    diffs = np.diff(np.signbit(x).astype(int))
    zcr = float(np.count_nonzero(diffs)) / x.size

    p25, p50, p75 = np.percentile(x, [25, 50, 75])

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "var": float(x.var()),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "median": float(p50),
        "p25": float(p25),
        "p75": float(p75),
        "iqr": float(p75 - p25),
        "skew": _safe_skew(x),
        "kurtosis": _safe_kurtosis(x),
        "energy": float(np.sum(x ** 2)),
        "zero_crossing_rate": zcr,
        "abs_mean": float(np.mean(np.abs(x))),
    }


def compute_time_features_batch(windows: np.ndarray) -> pd.DataFrame:
    """Aplica :func:`compute_time_features` a un arreglo 2-D ``(n, n_samples)``."""
    if windows.ndim != 2:
        raise ValueError("`windows` debe tener shape (n_windows, n_samples).")
    rows = [compute_time_features(w) for w in windows]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Features de intervalos RR
# ---------------------------------------------------------------------------
def compute_rr_intervals(beat_times: np.ndarray) -> np.ndarray:
    """Devuelve los intervalos RR (en segundos) a partir de tiempos de latido."""
    bt = np.asarray(beat_times, dtype=float).ravel()
    bt = np.sort(bt[~np.isnan(bt)])
    if bt.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(bt)


def compute_per_beat_rr_features(beat_times: np.ndarray) -> pd.DataFrame:
    """Calcula features RR locales por latido.

    Para cada latido devuelve:
        * ``rr_prev``: intervalo (segundos) al latido anterior. NaN para el primero.
        * ``rr_next``: intervalo al latido siguiente. NaN para el último.
        * ``rr_mean_local``: promedio de ``rr_prev`` y ``rr_next``, ignorando NaN.
        * ``rr_ratio``: ``rr_prev / rr_next``. NaN si alguno es NaN o cero.

    El DataFrame devuelto tiene un índice posicional 0..n-1 y se alinea con
    el orden de ``beat_times``. El llamador es responsable de pasar los
    latidos ordenados temporalmente.
    """
    bt = np.asarray(beat_times, dtype=float).ravel()
    n = bt.size
    rr_prev = np.full(n, np.nan)
    rr_next = np.full(n, np.nan)
    if n >= 2:
        diffs = np.diff(bt)
        rr_prev[1:] = diffs
        rr_next[:-1] = diffs

    stacked = np.vstack([rr_prev, rr_next])
    with np.errstate(invalid="ignore"):
        rr_mean_local = np.nanmean(stacked, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_ratio = rr_prev / np.where(rr_next == 0, np.nan, rr_next)

    return pd.DataFrame(
        {
            "rr_prev": rr_prev,
            "rr_next": rr_next,
            "rr_mean_local": rr_mean_local,
            "rr_ratio": rr_ratio,
        }
    )


def compute_rr_features(beat_times: np.ndarray) -> dict[str, float]:
    """Calcula features clásicas sobre intervalos RR.

    Returns
    -------
    dict[str, float]
        Diccionario con: ``rr_count``, ``rr_mean``, ``rr_std``, ``rr_min``,
        ``rr_max``, ``rr_rmssd``, ``rr_pnn50``.
    """
    rr = compute_rr_intervals(beat_times)
    if rr.size == 0:
        return {
            "rr_count": 0,
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_min": np.nan,
            "rr_max": np.nan,
            "rr_rmssd": np.nan,
            "rr_pnn50": np.nan,
        }

    diffs = np.diff(rr) if rr.size >= 2 else np.empty(0, dtype=float)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else np.nan
    if diffs.size:
        pnn50 = float(np.mean(np.abs(diffs) > 0.05))
    else:
        pnn50 = np.nan

    return {
        "rr_count": int(rr.size),
        "rr_mean": float(rr.mean()),
        "rr_std": float(rr.std()),
        "rr_min": float(rr.min()),
        "rr_max": float(rr.max()),
        "rr_rmssd": rmssd,
        "rr_pnn50": pnn50,
    }
