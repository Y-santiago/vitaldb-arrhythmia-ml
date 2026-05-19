"""Construcción de ventanas temporales alrededor de cada latido.

La unidad de análisis es una ventana centrada (o anclada) en el tiempo de un
latido anotado. Se admite sobrelapamiento entre ventanas consecutivas.

Las ventanas se extraen sobre la señal ECG cruda (ver :mod:`src.download`).
Se delega al `caller` el filtrado previo de latidos según calidad de señal y
etiqueta de ritmo (ver :mod:`src.preprocessing`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    BEAT_TIME_COLUMN,
    DEFAULT_ECG_FS_HZ,
    DEFAULT_WINDOW_OVERLAP,
    DEFAULT_WINDOW_SECONDS,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class WindowSpec:
    """Especificación de una ventana extraída.

    Attributes
    ----------
    case_id : int | str
        Caso al que pertenece la ventana.
    beat_index : int
        Índice del latido (fila de la tabla de anotaciones) que ancla la
        ventana.
    start_sample : int
        Muestra inicial de la ventana en la señal cruda.
    end_sample : int
        Muestra final (exclusiva) de la ventana en la señal cruda.
    label : str | None
        Etiqueta de ritmo asociada. ``None`` si el ventaneador se usa para
        inferencia sin etiquetas.
    """
    case_id: int | str
    beat_index: int
    start_sample: int
    end_sample: int
    label: str | None = None


def _seconds_to_samples(seconds: float, fs_hz: int) -> int:
    if seconds < 0:
        raise ValueError("`seconds` debe ser no negativo.")
    return int(round(seconds * fs_hz))


def build_windows_for_case(signal: np.ndarray,
                           beats: pd.DataFrame,
                           case_id: int | str,
                           fs_hz: int = DEFAULT_ECG_FS_HZ,
                           window_seconds: float = DEFAULT_WINDOW_SECONDS,
                           overlap: float = DEFAULT_WINDOW_OVERLAP,
                           beat_time_column: str = BEAT_TIME_COLUMN,
                           label_column: str = TARGET_COLUMN
                           ) -> tuple[np.ndarray, list[WindowSpec]]:
    """Construye ventanas centradas en cada latido del caso.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal ECG 1-D del caso.
    beats : pandas.DataFrame
        Anotaciones del caso. Debe incluir al menos `beat_time_column`.
    case_id : int | str
        Identificador del caso (solo se propaga a las :class:`WindowSpec`).
    fs_hz : int
        Frecuencia de muestreo de la señal en Hz.
    window_seconds : float
        Duración total de la ventana, en segundos.
    overlap : float
        Proporción de sobrelapamiento ``[0.0, 1.0)``. Las ventanas siempre se
        anclan en latidos; este parámetro controla un desplazamiento adicional
        sumando ventanas auxiliares por latido cuando es mayor que cero.
    beat_time_column : str
        Columna con el tiempo del latido en segundos.
    label_column : str
        Columna con la etiqueta de ritmo. Puede no existir si el DataFrame no
        está etiquetado.

    Returns
    -------
    windows : numpy.ndarray
        Arreglo 2-D ``(n_windows, n_samples)`` con las ventanas extraídas.
        Las ventanas que se salen de la señal se descartan.
    specs : list[WindowSpec]
        Metadatos por ventana, alineados fila-a-fila con `windows`.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("`overlap` debe estar en [0.0, 1.0).")
    if beat_time_column not in beats.columns:
        raise KeyError(
            f"La columna '{beat_time_column}' no está en el DataFrame de latidos."
        )

    window_samples = _seconds_to_samples(window_seconds, fs_hz)
    half = window_samples // 2
    n_total = signal.shape[0]

    step_seconds = window_seconds * (1.0 - overlap) if overlap > 0 else 0.0
    step_samples = _seconds_to_samples(step_seconds, fs_hz)

    rows: list[np.ndarray] = []
    specs: list[WindowSpec] = []

    labels = beats[label_column] if label_column in beats.columns else None

    for idx, beat_time in beats[beat_time_column].items():
        if pd.isna(beat_time):
            continue
        center = int(round(float(beat_time) * fs_hz))

        # Ventana principal: centrada en el latido.
        centers = [center]
        # Ventanas auxiliares por sobrelapamiento (a ambos lados).
        if step_samples > 0:
            centers.extend([center - step_samples, center + step_samples])

        if labels is None:
            label_value = None
        else:
            raw = labels.loc[idx]
            # `pd.isna` cubre NaN (float), None y NaT; cualquiera de esos casos
            # se trata como "sin etiqueta" para evitar la cadena literal "nan".
            label_value = None if pd.isna(raw) else str(raw)

        for c in centers:
            start = c - half
            end = start + window_samples
            if start < 0 or end > n_total:
                continue
            rows.append(signal[start:end])
            specs.append(
                WindowSpec(
                    case_id=case_id,
                    beat_index=int(idx),
                    start_sample=int(start),
                    end_sample=int(end),
                    label=label_value,
                )
            )

    if not rows:
        return np.empty((0, window_samples), dtype=signal.dtype), []
    return np.stack(rows, axis=0), specs
