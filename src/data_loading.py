"""Carga de metadata y anotaciones de la VitalDB Arrhythmia Database (PhysioNet).

Este módulo NO descarga datos: asume que el paquete de PhysioNet fue colocado
manualmente bajo `data/raw/physionet_annotations/` (ver README §6).

Funciones principales:
    * :func:`load_metadata`        — lee `metadata.csv`.
    * :func:`load_annotations_for_case` — lee el archivo de anotaciones de un
      caso individual.
    * :func:`load_all_annotations` — concatena anotaciones de varios casos.
    * :func:`merge_metadata_and_annotations` — une por `case_id`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    CASE_ID_COLUMN,
    PHYSIONET_DIR,
)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def load_metadata(physionet_dir: str | Path = PHYSIONET_DIR,
                  filename: str = "metadata.csv") -> pd.DataFrame:
    """Carga `metadata.csv` desde el paquete de PhysioNet.

    Parameters
    ----------
    physionet_dir : str | Path
        Carpeta que contiene el paquete de PhysioNet.
    filename : str
        Nombre del archivo de metadata. Por defecto `metadata.csv`.

    Returns
    -------
    pandas.DataFrame
        DataFrame con la metadata cruda, sin transformaciones.

    Raises
    ------
    FileNotFoundError
        Si el archivo no se encuentra en `physionet_dir`.
    """
    path = Path(physionet_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de metadata en: {path}. "
            "Revisa la sección §6 del README sobre descarga de datos."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Anotaciones
# ---------------------------------------------------------------------------
def _resolve_annotations_dir(physionet_dir: str | Path) -> Path:
    """Devuelve la carpeta de anotaciones dentro del paquete de PhysioNet.

    El paquete oficial suele incluir una subcarpeta llamada `Annotation_Files`.
    Si no existe se devuelve directamente `physionet_dir` para permitir
    estructuras alternativas.
    """
    base = Path(physionet_dir)
    candidate = base / "Annotation_Files"
    return candidate if candidate.exists() else base


def load_annotations_for_case(case_id: int | str,
                              physionet_dir: str | Path = PHYSIONET_DIR,
                              extension: str = ".csv") -> pd.DataFrame:
    """Carga las anotaciones de un único caso.

    No interpreta ni renombra columnas; entrega el DataFrame tal cual está en
    disco para que el ETL posterior decida las transformaciones.

    Parameters
    ----------
    case_id : int | str
        Identificador del caso.
    physionet_dir : str | Path
        Carpeta raíz del paquete de PhysioNet.
    extension : str
        Extensión del archivo de anotaciones. Por defecto `.csv`.

    Returns
    -------
    pandas.DataFrame
        Anotaciones del caso. Se garantiza que la columna `case_id` esté
        presente en el DataFrame devuelto.
    """
    annotations_dir = _resolve_annotations_dir(physionet_dir)
    matches = sorted(annotations_dir.glob(f"*{case_id}*{extension}"))
    if not matches:
        raise FileNotFoundError(
            f"No se encontró un archivo de anotaciones para case_id={case_id} "
            f"en {annotations_dir}."
        )
    if len(matches) > 1:
        # Caso ambiguo: el usuario decide cómo desambiguar. Se levanta para no
        # cargar silenciosamente un archivo equivocado.
        raise ValueError(
            f"Múltiples archivos coinciden con case_id={case_id}: {matches}. "
            "Ajusta el patrón o la estructura de carpetas."
        )
    df = pd.read_csv(matches[0])
    if CASE_ID_COLUMN not in df.columns:
        df[CASE_ID_COLUMN] = case_id
    return df


def load_all_annotations(case_ids: Iterable[int | str] | None = None,
                         physionet_dir: str | Path = PHYSIONET_DIR,
                         extension: str = ".csv") -> pd.DataFrame:
    """Carga y concatena anotaciones de varios casos.

    Parameters
    ----------
    case_ids : iterable de int | str | None
        Identificadores de casos a cargar. Si es ``None`` se intenta cargar
        todos los archivos del directorio de anotaciones.
    physionet_dir : str | Path
        Carpeta raíz del paquete de PhysioNet.
    extension : str
        Extensión de los archivos de anotaciones.

    Returns
    -------
    pandas.DataFrame
        Anotaciones concatenadas. Si no hay coincidencias devuelve un
        DataFrame vacío.
    """
    annotations_dir = _resolve_annotations_dir(physionet_dir)

    if case_ids is None:
        files = sorted(annotations_dir.glob(f"*{extension}"))
        frames = [pd.read_csv(f) for f in files]
    else:
        frames = [
            load_annotations_for_case(cid, physionet_dir, extension)
            for cid in case_ids
        ]

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Merge metadata + anotaciones
# ---------------------------------------------------------------------------
def merge_metadata_and_annotations(metadata: pd.DataFrame,
                                   annotations: pd.DataFrame,
                                   on: str = CASE_ID_COLUMN,
                                   how: str = "inner") -> pd.DataFrame:
    """Une metadata y anotaciones por `case_id`.

    Parameters
    ----------
    metadata : pandas.DataFrame
        DataFrame de metadata por caso.
    annotations : pandas.DataFrame
        DataFrame de anotaciones por latido.
    on : str
        Columna sobre la que se hace el join. Por defecto `case_id`.
    how : {"inner", "left", "right", "outer"}
        Tipo de join.

    Returns
    -------
    pandas.DataFrame
        DataFrame combinado.
    """
    if on not in metadata.columns:
        raise KeyError(f"La columna '{on}' no está en metadata.")
    if on not in annotations.columns:
        raise KeyError(f"La columna '{on}' no está en annotations.")
    return annotations.merge(metadata, on=on, how=how, suffixes=("", "_meta"))
