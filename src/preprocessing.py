"""Preprocesamiento de anotaciones y preprocesamiento tabular para modelado.

Aplica las reglas metodológicas definidas en el README:
    * Excluir la clase ``Noise`` (y otras etiquetas listadas en
      :data:`config.EXCLUDED_RHYTHM_LABELS`).
    * Excluir registros con ``bad_signal_quality``.
    * Validar la presencia de columnas requeridas.
    * Eliminar duplicados exactos.

También expone :func:`build_tabular_preprocessor`, que construye un
``ColumnTransformer`` con imputación + escalado para numéricas e
imputación + ``OneHotEncoder(handle_unknown="ignore")`` para
categóricas. El preprocesador queda envuelto dentro de cada ``Pipeline``,
por lo que su ``fit`` solo ve el train del split externo (sin fuga al
test).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    EXCLUDED_RHYTHM_LABELS,
    SIGNAL_QUALITY_COLUMN,
    TABULAR_OHE_MIN_FREQUENCY,
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


# ---------------------------------------------------------------------------
# Preprocesador tabular (Imputer + Scaler / Imputer + OneHotEncoder)
# ---------------------------------------------------------------------------
def build_tabular_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    with_scaling: bool = True,
    ohe_min_frequency: int = TABULAR_OHE_MIN_FREQUENCY,
) -> ColumnTransformer:
    """Construye el ``ColumnTransformer`` para features tabulares.

    Parameters
    ----------
    numeric_features : sequence of str
        Nombres de columnas numéricas a procesar.
    categorical_features : sequence of str
        Nombres de columnas categóricas a procesar (OneHotEncoder).
    with_scaling : bool
        Si ``True``, aplica ``StandardScaler`` a las numéricas tras imputar.
        Conviene para modelos lineales y MLP; para modelos basados en
        árboles puede dejarse en ``True`` también porque el efecto es
        neutral sobre la partición binaria.
    ohe_min_frequency : int
        Frecuencia mínima (en filas de train) que debe tener una categoría
        para mantener su propia columna. El resto va a ``infrequent_sklearn``.
        Limita la explosión dimensional con categóricas de alta cardinalidad.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Listo para usarse como primer step de un ``Pipeline``.
    """
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaling:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=ohe_min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []
    if list(numeric_features):
        transformers.append(("num", numeric_pipeline, list(numeric_features)))
    if list(categorical_features):
        transformers.append(("cat", categorical_pipeline, list(categorical_features)))

    if not transformers:
        raise ValueError(
            "No se proporcionaron features ni numéricas ni categóricas. "
            "Verifica la clasificación del dataset antes de modelar."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # cualquier columna no listada se descarta sin warning
        verbose_feature_names_out=True,
    )


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
