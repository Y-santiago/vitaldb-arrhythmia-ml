"""Preprocesamiento de anotaciones.

Aplica las reglas metodológicas definidas en el README:
    * Excluir la clase ``Noise`` (y otras etiquetas listadas en
      :data:`config.EXCLUDED_RHYTHM_LABELS`).
    * Excluir registros con ``bad_signal_quality``.
    * Validar la presencia de columnas requeridas.
    * Eliminar duplicados exactos.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import (
    EXCLUDED_RHYTHM_LABELS,
    SIGNAL_QUALITY_COLUMN,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
def exclude_rhythm_labels(df: pd.DataFrame,
                          labels: Iterable[str] = EXCLUDED_RHYTHM_LABELS,
                          target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Devuelve el DataFrame sin las filas cuya etiqueta de ritmo esté en `labels`.

    Por defecto se excluye la clase ``Noise``.
    """
    if target_column not in df.columns:
        raise KeyError(
            f"La columna objetivo '{target_column}' no está en el DataFrame."
        )
    mask = ~df[target_column].isin(set(labels))
    return df.loc[mask].copy()


def exclude_bad_signal_quality(df: pd.DataFrame,
                               column: str = SIGNAL_QUALITY_COLUMN) -> pd.DataFrame:
    """Excluye filas con `bad_signal_quality` verdadero.

    Acepta valores booleanos o convertibles a booleano (``0/1``, ``"true"``/
    ``"false"``). Si la columna no existe se devuelve el DataFrame intacto y
    se asume que no hay marcas de mala calidad.
    """
    if column not in df.columns:
        return df.copy()
    flag = df[column]
    if flag.dtype == object:
        normalized = flag.astype(str).str.strip().str.lower()
        is_bad = normalized.isin({"true", "1", "yes", "y"})
    else:
        is_bad = flag.astype(bool)
    return df.loc[~is_bad].copy()


# ---------------------------------------------------------------------------
# Validación y limpieza
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame,
                     required: Iterable[str]) -> None:
    """Verifica que `required` esté contenido en las columnas de `df`.

    Levanta ``KeyError`` listando explícitamente las columnas faltantes.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Faltan columnas requeridas en el DataFrame: {missing}. "
            f"Disponibles: {list(df.columns)}"
        )


def drop_exact_duplicates(df: pd.DataFrame,
                          subset: Iterable[str] | None = None) -> pd.DataFrame:
    """Elimina filas duplicadas exactas (todas las columnas por defecto)."""
    return df.drop_duplicates(subset=list(subset) if subset is not None else None).copy()


def apply_basic_filters(df: pd.DataFrame,
                        target_column: str = TARGET_COLUMN,
                        signal_quality_column: str = SIGNAL_QUALITY_COLUMN,
                        excluded_labels: Iterable[str] = EXCLUDED_RHYTHM_LABELS) -> pd.DataFrame:
    """Aplica en orden los filtros básicos definidos por la metodología.

    Pasos:
        1. ``exclude_bad_signal_quality``
        2. ``exclude_rhythm_labels`` con `excluded_labels`
        3. ``drop_exact_duplicates`` sobre todas las columnas.
    """
    out = exclude_bad_signal_quality(df, column=signal_quality_column)
    out = exclude_rhythm_labels(out, labels=excluded_labels, target_column=target_column)
    out = drop_exact_duplicates(out)
    return out
