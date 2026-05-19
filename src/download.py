"""Descarga de señal ECG desde VitalDB usando la librería oficial `vitaldb`.

El paquete de PhysioNet contiene metadata y anotaciones, pero NO la señal
cruda. La señal se obtiene de VitalDB indexando por `case_id`.

Funciones:
    * :func:`load_ecg_from_vitaldb` — carga el ECG de un único caso.
    * :func:`save_ecg_npy`          — persiste la señal a disco en `.npy`.

Nota: las descargas pueden ser lentas. Se recomienda cachear cada caso en
`data/raw/vitaldb_waveforms/` (excluido del repositorio).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_ECG_FS_HZ,
    DEFAULT_ECG_TRACK_NAME,
    VITALDB_WAVEFORMS_DIR,
)


def load_ecg_from_vitaldb(case_id: int,
                          track_name: str = DEFAULT_ECG_TRACK_NAME,
                          sampling_rate_hz: int = DEFAULT_ECG_FS_HZ) -> np.ndarray:
    """Carga una señal ECG desde VitalDB para un `case_id` dado.

    Parameters
    ----------
    case_id : int
        Identificador del caso en VitalDB.
    track_name : str
        Nombre del canal ECG a descargar (ej. ``"SNUADC/ECG_II"``).
    sampling_rate_hz : int
        Frecuencia de muestreo objetivo en Hz.

    Returns
    -------
    numpy.ndarray
        Señal 1-D del ECG para el caso solicitado. Puede contener ``NaN`` en
        tramos sin registro; las decisiones de limpieza se delegan a
        :mod:`src.preprocessing`.

    Raises
    ------
    ImportError
        Si la librería ``vitaldb`` no está instalada.
    """
    try:
        import vitaldb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "La librería `vitaldb` no está instalada. "
            "Instala las dependencias con `pip install -r requirements.txt`."
        ) from exc

    interval_seconds = 1.0 / sampling_rate_hz
    signal = vitaldb.load_case(case_id, [track_name], interval_seconds)

    if signal is None:
        raise RuntimeError(
            f"VitalDB no devolvió señal para case_id={case_id}, "
            f"track='{track_name}'."
        )

    signal = np.asarray(signal)
    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal[:, 0]
    return signal


def save_ecg_npy(signal: np.ndarray,
                 case_id: int,
                 output_dir: str | Path = VITALDB_WAVEFORMS_DIR) -> Path:
    """Guarda la señal ECG en formato ``.npy``.

    El nombre del archivo es ``case_<case_id>.npy``. La carpeta destino se
    crea si no existe.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"case_{case_id}.npy"
    np.save(out_path, signal)
    return out_path
