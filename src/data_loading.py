"""Carga de metadata y anotaciones de la VitalDB Arrhythmia Database (PhysioNet).

Este módulo NO descarga datos: asume que el paquete de PhysioNet fue colocado
manualmente bajo `data/raw/physionet_annotations/` (ver README §6).

Notas sobre el formato real del paquete:
    * Las anotaciones viven en `Annotation_Files/` (o directamente en la carpeta
      raíz si el usuario las movió).
    * Cada archivo se llama ``Annotation_file_<case_id>.csv`` (singular en el
      paquete oficial; el código admite también la variante plural).
    * Los CSV de anotaciones NO contienen una columna ``case_id``: el
      identificador del caso vive únicamente en el nombre del archivo y se
      inyecta como columna al cargarlos.

Funciones principales:
    * :func:`load_metadata`             — lee `metadata.csv`.
    * :func:`load_annotations_for_case` — lee el archivo de anotaciones de un
      caso individual.
    * :func:`load_all_annotations`      — concatena anotaciones de varios casos.
    * :func:`merge_metadata_and_annotations` — une por `case_id`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    ANNOTATION_FILENAME_REGEX,
    CASE_ID_COLUMN,
    PHYSIONET_DIR,
)


_ANNOTATION_PATTERN = re.compile(ANNOTATION_FILENAME_REGEX, re.IGNORECASE)


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
# Anotaciones — helpers
# ---------------------------------------------------------------------------
def _resolve_annotations_dir(physionet_dir: str | Path) -> Path:
    """Devuelve la carpeta de anotaciones dentro del paquete de PhysioNet.

    El paquete oficial usa `Annotation_Files/`. Si no existe se devuelve
    `physionet_dir` para soportar estructuras alternativas.
    """
    base = Path(physionet_dir)
    candidate = base / "Annotation_Files"
    return candidate if candidate.exists() else base


def _case_id_from_filename(path: str | Path) -> int | None:
    """Extrae el `case_id` (entero) de un nombre tipo `Annotation_file_<id>.csv`.

    Devuelve ``None`` si el nombre no coincide con el patrón.
    """
    name = Path(path).name
    match = _ANNOTATION_PATTERN.match(name)
    return int(match.group(1)) if match else None


def _candidate_filenames(case_id: int | str) -> list[str]:
    """Construye los posibles nombres exactos del archivo de un caso."""
    return [f"Annotation_file_{case_id}.csv", f"Annotations_file_{case_id}.csv"]


def _read_annotation_file(path: Path) -> pd.DataFrame:
    """Lee un único archivo de anotaciones e inyecta `case_id` desde el nombre."""
    df = pd.read_csv(path)
    if CASE_ID_COLUMN not in df.columns:
        case_id = _case_id_from_filename(path)
        if case_id is None:
            raise ValueError(
                f"No se pudo inferir `case_id` desde el nombre del archivo: {path.name}. "
                "Se esperaba el patrón `Annotation_file_<id>.csv`."
            )
        df[CASE_ID_COLUMN] = case_id
    return df


# ---------------------------------------------------------------------------
# Anotaciones — API pública
# ---------------------------------------------------------------------------
def load_annotations_for_case(case_id: int | str,
                              physionet_dir: str | Path = PHYSIONET_DIR,
                              extension: str = ".csv") -> pd.DataFrame:
    """Carga las anotaciones de un único caso.

    Busca el archivo por nombre exacto (`Annotation_file_<case_id>.csv` o la
    variante plural) y, si no existe, cae a un glob por sufijo controlado.
    El identificador del caso se inyecta como columna ``case_id`` porque el
    CSV original no la contiene.

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
        Anotaciones del caso, con la columna `case_id` poblada.
    """
    annotations_dir = _resolve_annotations_dir(physionet_dir)

    # 1) Coincidencia exacta por nombre (preferida).
    for candidate in _candidate_filenames(case_id):
        path = annotations_dir / candidate
        if path.exists():
            return _read_annotation_file(path)

    # 2) Fallback: matchear por regex y filtrar por id exacto extraído del nombre.
    target = int(case_id) if str(case_id).isdigit() else case_id
    for path in sorted(annotations_dir.glob(f"*{extension}")):
        parsed = _case_id_from_filename(path)
        if parsed is not None and parsed == target:
            return _read_annotation_file(path)

    raise FileNotFoundError(
        f"No se encontró un archivo de anotaciones para case_id={case_id} "
        f"en {annotations_dir}. Patrón esperado: `Annotation_file_<id>.csv`."
    )


def load_all_annotations(case_ids: Iterable[int | str] | None = None,
                         physionet_dir: str | Path = PHYSIONET_DIR,
                         extension: str = ".csv") -> pd.DataFrame:
    """Carga y concatena anotaciones de varios casos.

    Para cada archivo cargado se inyecta la columna ``case_id`` derivada del
    nombre, garantizando que el DataFrame de salida sea apto para hacer merge
    contra `metadata.csv`.

    Parameters
    ----------
    case_ids : iterable de int | str | None
        Identificadores a cargar. Si es ``None`` se cargan todos los archivos
        del directorio cuyo nombre coincida con el patrón
        `Annotation_file_<id>.csv`.
    physionet_dir : str | Path
        Carpeta raíz del paquete de PhysioNet.
    extension : str
        Extensión de los archivos de anotaciones.

    Returns
    -------
    pandas.DataFrame
        Anotaciones concatenadas (con columna `case_id`). DataFrame vacío si
        no hay coincidencias.
    """
    annotations_dir = _resolve_annotations_dir(physionet_dir)

    if case_ids is None:
        candidate_paths = sorted(annotations_dir.glob(f"*{extension}"))
        files = [p for p in candidate_paths if _case_id_from_filename(p) is not None]
        frames = [_read_annotation_file(p) for p in files]
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
        DataFrame de anotaciones por latido. Debe contener `case_id`
        (`load_all_annotations` / `load_annotations_for_case` lo inyectan).
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
        raise KeyError(
            f"La columna '{on}' no está en annotations. "
            "Asegúrate de cargar las anotaciones con `load_all_annotations` o "
            "`load_annotations_for_case` para que se inyecte desde el nombre del archivo."
        )
    return annotations.merge(metadata, on=on, how=how, suffixes=("", "_meta"))
