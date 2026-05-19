"""Pipelines baseline y validación por grupos.

Regla metodológica clave: la separación train/test debe hacerse **por
`case_id`**, nunca por ventana ni por latido aleatorio. El módulo expone:

* :func:`make_group_split`           — split simple respetando grupos.
* :func:`make_group_kfold`           — iterador de folds por grupo.
* :func:`build_logreg_pipeline`      — baseline lineal con escalado.
* :func:`build_rf_pipeline`          — baseline Random Forest.
* :func:`build_xgb_pipeline`         — baseline XGBoost (si está instalado).
* :func:`assert_no_forbidden_features` — chequea que ``beat_type`` y otras
  columnas prohibidas no estén entre las features.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def make_group_kfold(X: pd.DataFrame | np.ndarray,
                     y: pd.Series | np.ndarray,
                     groups: pd.Series | np.ndarray,
                     n_splits: int = DEFAULT_N_SPLITS
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Iterador de folds ``GroupKFold`` por `case_id`."""
    kf = GroupKFold(n_splits=n_splits)
    yield from kf.split(X, y, groups=groups)


# ---------------------------------------------------------------------------
# Pipelines baseline
# ---------------------------------------------------------------------------
def build_logreg_pipeline(class_weight: str | None = "balanced",
                          random_state: int = RANDOM_SEED) -> Pipeline:
    """Pipeline ``StandardScaler -> LogisticRegression`` multinomial."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    multi_class="auto",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_rf_pipeline(n_estimators: int = 300,
                      class_weight: str | None = "balanced",
                      random_state: int = RANDOM_SEED) -> Pipeline:
    """Pipeline con un ``RandomForestClassifier`` (sin escalado)."""
    return Pipeline(
        steps=[
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
    """Pipeline con ``XGBClassifier``. Requiere `xgboost` instalado.

    El balanceo de clases no está predefinido aquí; XGBoost expone
    ``scale_pos_weight`` solo para binario. Para multiclase considera ajustar
    `sample_weight` al entrenar.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "XGBoost no está instalado. Ejecuta `pip install xgboost`."
        ) from exc

    return Pipeline(
        steps=[
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
