"""Pipelines baseline y validación por grupos.

Reglas metodológicas codificadas aquí:
    * Split train/test **por `case_id`** (`GroupKFold` / `GroupShuffleSplit`).
    * `beat_type` y otras columnas listadas en
      :data:`config.FORBIDDEN_FEATURE_COLUMNS` no pueden entrar como features.

Pipeline base:
    ``SimpleImputer(strategy="median") -> StandardScaler -> Classifier``

El imputer es necesario porque algunas features pueden contener NaN (ventanas
degeneradas, divisiones por cero en estadísticas, etc.). El escalado se aplica
en pipelines lineales; para Random Forest se omite por ser invariante a
escala. Todas las etapas viven dentro del mismo ``Pipeline`` para evitar
fugas (el `fit` se hace solo sobre train).
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DEFAULT_N_SPLITS,
    FORBIDDEN_FEATURE_COLUMNS,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Validación de features
# ---------------------------------------------------------------------------
def assert_no_forbidden_features(feature_columns: Iterable[str],
                                 forbidden: Iterable[str] = FORBIDDEN_FEATURE_COLUMNS
                                 ) -> None:
    """Levanta ``ValueError`` si alguna columna prohibida aparece como feature.

    Por defecto bloquea ``beat_type``, la columna objetivo, ``case_id`` y la
    marca de calidad de señal.
    """
    forbidden_set = set(forbidden)
    leaked = [c for c in feature_columns if c in forbidden_set]
    if leaked:
        raise ValueError(
            f"Columnas prohibidas como features: {leaked}. "
            "Revisa la metodología: `beat_type` no puede usarse como predictor."
        )


# ---------------------------------------------------------------------------
# Splits por grupo
# ---------------------------------------------------------------------------
def make_group_split(X: pd.DataFrame | np.ndarray,
                     y: pd.Series | np.ndarray,
                     groups: pd.Series | np.ndarray,
                     test_size: float = 0.2,
                     random_state: int = RANDOM_SEED
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Single split estratificado por grupo (`case_id`).

    Returns
    -------
    train_idx, test_idx : numpy.ndarray
        Índices posicionales para train y test.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def safe_n_splits(n_splits_requested: int,
                  groups: pd.Series | np.ndarray) -> int:
    """Recorta ``n_splits`` al número de grupos únicos disponibles.

    ``GroupKFold`` exige ``n_splits <= n_groups``. Este helper devuelve el
    mínimo entre lo pedido y los grupos disponibles, asegurando además que
    haya al menos 2 grupos para que el split tenga sentido.

    Raises
    ------
    ValueError
        Si hay menos de 2 grupos únicos.
    """
    n_groups = int(np.unique(np.asarray(groups)).shape[0])
    if n_groups < 2:
        raise ValueError(
            f"Se necesitan al menos 2 `case_id` únicos para hacer GroupKFold. "
            f"Recibí n_groups={n_groups}."
        )
    return min(int(n_splits_requested), n_groups)


def make_group_kfold(X: pd.DataFrame | np.ndarray,
                     y: pd.Series | np.ndarray,
                     groups: pd.Series | np.ndarray,
                     n_splits: int = DEFAULT_N_SPLITS
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Iterador de folds ``GroupKFold`` por `case_id`.

    Ajusta automáticamente ``n_splits`` con :func:`safe_n_splits` si el número
    de grupos disponibles es menor que el solicitado.
    """
    n_splits_eff = safe_n_splits(n_splits, groups)
    kf = GroupKFold(n_splits=n_splits_eff)
    yield from kf.split(X, y, groups=groups)


# ---------------------------------------------------------------------------
# Pipelines baseline (Imputer -> Scaler -> Clf)
# ---------------------------------------------------------------------------
def _build_imputer() -> SimpleImputer:
    """SimpleImputer con mediana — robusto a outliers y NaN aislados."""
    return SimpleImputer(strategy="median")


def build_logreg_pipeline(class_weight: str | None = "balanced",
                          random_state: int = RANDOM_SEED,
                          max_iter: int = 2000) -> Pipeline:
    """Pipeline ``Imputer -> Scaler -> LogisticRegression`` multinomial."""
    return Pipeline(
        steps=[
            ("imputer", _build_imputer()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight=class_weight,
                    multi_class="auto",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_rf_pipeline(n_estimators: int = 300,
                      class_weight: str | None = "balanced",
                      random_state: int = RANDOM_SEED) -> Pipeline:
    """Pipeline ``Imputer -> RandomForestClassifier``.

    Random Forest no requiere escalado de features.
    """
    return Pipeline(
        steps=[
            ("imputer", _build_imputer()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgb_pipeline(random_state: int = RANDOM_SEED) -> Pipeline:
    """Pipeline ``Imputer -> XGBClassifier``. Requiere `xgboost` instalado.

    El balanceo en multiclase se maneja vía ``sample_weight`` al llamar
    ``fit`` (no se predefine aquí). Para inferencia se obtiene mediante
    :func:`sklearn.utils.class_weight.compute_sample_weight`.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "XGBoost no está instalado. Ejecuta `pip install xgboost`."
        ) from exc

    return Pipeline(
        steps=[
            ("imputer", _build_imputer()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
