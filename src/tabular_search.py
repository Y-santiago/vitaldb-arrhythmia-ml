"""Búsqueda de hiperparámetros sobre el dataset tabular filtrado.

Este es el módulo **activo** de modelado en la fase actual. Reemplaza a
``src/search.py``, que queda como flujo legacy basado en ECG crudo.

Componentes principales:
    * :func:`classify_features` — separa columnas en numéricas / categóricas
      tras descartar las listadas en ``config.TABULAR_LEAKAGE_COLUMNS``.
    * :func:`build_pipeline_for_model` — arma ``Pipeline`` con preprocesador
      tabular + clasificador para uno de los modelos del registro.
    * :data:`TABULAR_PARAM_DISTRIBUTIONS` — diccionarios de búsqueda por modelo.
    * :func:`run_tabular_hyperparameter_search` — orquesta toda la fase:
      carga, split por grupo, búsqueda, evaluación final en test, ensamblaje
      de tablas de resultados, persistencia.

Restricciones codificadas:
    * ``case_id``, ``rhythm_label``, ``beat_type``, ``rhythm_classes``,
      ``bad_signal_quality*`` y outcomes posteriores nunca entran al modelo
      (ver ``config.TABULAR_LEAKAGE_COLUMNS``).
    * Split estricto por ``case_id`` (``make_train_test_group_split_with_coverage``).
    * CV interna por grupo (``StratifiedGroupKFold`` con fallback a
      ``GroupKFold``).
    * Test congelado: se evalúa una sola vez al final.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CASE_ID_COLUMN,
    PROCESSED_DIR,
    RANDOM_SEED,
    TABULAR_DATASET_FILENAME,
    TABULAR_LEAKAGE_COLUMNS,
    TABULAR_MAX_CATEGORY_CARDINALITY,
    TABULAR_OHE_MIN_FREQUENCY,
    TARGET_COLUMN,
)
from .evaluation import (
    class_support_per_split,
    classes_missing_in_train,
    confusion_matrix_with_totals,
    per_class_report,
)
from .modeling import (
    assert_no_forbidden_features,
    make_train_test_group_split_with_coverage,
)
from .preprocessing import build_tabular_preprocessor


SCORING_METRICS: dict[str, str] = {
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_weighted": "f1_weighted",
}
PRIMARY_SCORING: str = "f1_macro"


# ---------------------------------------------------------------------------
# Wrapper XGBoost robusto frente a clases ausentes en folds
# ---------------------------------------------------------------------------
class _XGBClassifierSafe:
    """Wrapper que re-encoda etiquetas dentro de ``fit``.

    XGBoost ≥ 2.0 exige que ``y`` use enteros consecutivos en ``[0, num_class)``.
    Con CV por grupo es habitual que un fold quede sin alguna clase, lo que
    produce huecos en el ``LabelEncoder`` externo y rompe ``XGBClassifier.fit``.
    Este wrapper aplica ``LabelEncoder`` dentro de su propio ``fit`` y
    revierte en ``predict``, manteniendo strings hacia afuera del modelo.

    No subclasa ``XGBClassifier`` para no confundir a ``sklearn.base.clone``;
    expone solo ``fit``, ``predict``, ``predict_proba``, ``get_params``,
    ``set_params`` y ``classes_``.
    """

    def __init__(self, **xgb_kwargs):
        from xgboost import XGBClassifier  # import diferido
        self._xgb_cls = XGBClassifier
        self._xgb_kwargs = xgb_kwargs
        self._le = None
        self._xgb = None
        for k, v in xgb_kwargs.items():
            setattr(self, k, v)

    def get_params(self, deep: bool = True) -> dict:
        return dict(self._xgb_kwargs)

    def set_params(self, **params):
        self._xgb_kwargs.update(params)
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, **kwargs):
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._xgb = self._xgb_cls(**self._xgb_kwargs)
        self._xgb.fit(X, y_enc, **kwargs)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        y_pred_enc = self._xgb.predict(X)
        return self._le.inverse_transform(np.asarray(y_pred_enc, dtype=int))

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)


# ---------------------------------------------------------------------------
# Clasificación de features (numéricas / categóricas / leakage)
# ---------------------------------------------------------------------------
def classify_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = CASE_ID_COLUMN,
    leakage: Iterable[str] = TABULAR_LEAKAGE_COLUMNS,
    max_categorical_cardinality: int = TABULAR_MAX_CATEGORY_CARDINALITY,
) -> dict[str, list[str]]:
    """Separa columnas en numéricas / categóricas / excluidas.

    Returns
    -------
    dict con claves:
        ``numeric_features``, ``categorical_features``,
        ``leakage_excluded``, ``high_cardinality_excluded``,
        ``constant_excluded``.
    """
    leakage_set = set(leakage)
    numeric: list[str] = []
    categorical: list[str] = []
    leak: list[str] = []
    high_card: list[str] = []
    constant: list[str] = []

    for col in df.columns:
        if col in leakage_set:
            leak.append(col)
            continue
        n_unique = int(df[col].nunique(dropna=True))
        if n_unique <= 1:
            constant.append(col)
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            if n_unique <= max_categorical_cardinality:
                categorical.append(col)
            else:
                high_card.append(col)
        else:
            high_card.append(col)

    # Verificación dura: ni el target ni el grupo deben filtrarse como features.
    assert_no_forbidden_features(numeric + categorical, forbidden=leakage_set)

    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "leakage_excluded": leak,
        "high_cardinality_excluded": high_card,
        "constant_excluded": constant,
    }


# ---------------------------------------------------------------------------
# Pipeline factories por modelo
# ---------------------------------------------------------------------------
def build_pipeline_for_model(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Construye ``Pipeline(preprocessor → clf)`` para un modelo del registro."""
    pre = build_tabular_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        with_scaling=True,
        ohe_min_frequency=TABULAR_OHE_MIN_FREQUENCY,
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=random_state,
        )
    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        clf = _XGBClassifierSafe(
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif model_name == "linear_svc":
        clf = LinearSVC(
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
            dual="auto",
        )
    elif model_name == "mlp":
        clf = MLPClassifier(
            random_state=random_state,
            max_iter=200,
            early_stopping=False,
        )
    else:
        raise KeyError(f"Modelo desconocido: {model_name!r}")

    return Pipeline(steps=[("preprocessor", pre), ("clf", clf)])


TABULAR_PARAM_DISTRIBUTIONS: dict[str, dict[str, Any]] = {
    "logreg": {
        "clf__C": loguniform(1e-3, 1e2),
    },
    "decision_tree": {
        "clf__max_depth": [None, 5, 10, 15, 20, 30],
        "clf__min_samples_split": [2, 5, 10, 20, 50, 100],
        "clf__min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "clf__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 5, 7, 10],
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
    "linear_svc": {
        "clf__C": loguniform(1e-3, 1e2),
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
        "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [5e-4, 1e-3, 5e-3],
        "clf__activation": ["relu", "tanh"],
    },
}

TABULAR_MODEL_NAMES: tuple[str, ...] = tuple(TABULAR_PARAM_DISTRIBUTIONS.keys())


# ---------------------------------------------------------------------------
# CV builder (StratifiedGroupKFold con fallback a GroupKFold)
# ---------------------------------------------------------------------------
def build_cv_splitter(groups_train: np.ndarray,
                      y_train: np.ndarray,
                      n_splits: int,
                      prefer_stratified: bool = True):
    """Devuelve (cv_obj, cv_name, n_splits_efectivo)."""
    n_groups = int(np.unique(groups_train).shape[0])
    n_splits_eff = max(2, min(int(n_splits), n_groups))

    if prefer_stratified:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            cv = StratifiedGroupKFold(
                n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_SEED
            )
            # Verificar viabilidad: si lanza al construir folds, caemos.
            _ = list(cv.split(np.zeros((len(y_train), 1)), y_train, groups=groups_train))
            return cv, "StratifiedGroupKFold", n_splits_eff
        except Exception:  # noqa: BLE001
            pass

    return GroupKFold(n_splits=n_splits_eff), "GroupKFold", n_splits_eff


# ---------------------------------------------------------------------------
# Estructuras de resultado
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    model: str
    status: str
    best_params: dict | None
    best_cv_score_primary: float
    cv_metrics: dict
    test_metrics: dict
    fit_seconds: float
    n_iter_effective: int
    best_estimator: Any = None
    y_pred_test: np.ndarray | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Búsqueda por modelo
# ---------------------------------------------------------------------------
def run_search_for_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    cv,
    n_iter: int,
    random_state: int = RANDOM_SEED,
    n_jobs: int = -1,
) -> ModelResult:
    pipe = build_pipeline_for_model(
        model_name=model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )
    param_dist = TABULAR_PARAM_DISTRIBUTIONS[model_name]

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=SCORING_METRICS,
        refit=PRIMARY_SCORING,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score=np.nan,
        return_train_score=False,
        verbose=0,
    )

    t0 = time.time()
    search.fit(X_train, y_train, groups=groups_train)
    fit_seconds = time.time() - t0

    best_idx = int(search.best_index_)
    cv_results = search.cv_results_
    cv_metrics: dict[str, float] = {}
    for metric in SCORING_METRICS:
        col = f"mean_test_{metric}"
        cv_metrics[f"cv_{metric}"] = (
            float(cv_results[col][best_idx]) if col in cv_results else float("nan")
        )

    return ModelResult(
        model=model_name,
        status="ok",
        best_params=dict(search.best_params_),
        best_cv_score_primary=float(search.best_score_),
        cv_metrics=cv_metrics,
        test_metrics={},  # se llena en evaluate_on_test
        fit_seconds=fit_seconds,
        n_iter_effective=int(len(cv_results["params"])),
        best_estimator=search.best_estimator_,
    )


def evaluate_on_test(result: ModelResult,
                     X_test: pd.DataFrame,
                     y_test: np.ndarray) -> ModelResult:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if result.best_estimator is None:
        return result

    y_pred = result.best_estimator.predict(X_test)
    result.y_pred_test = y_pred
    result.test_metrics = {
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    return result


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def extract_feature_importance(estimator: Pipeline) -> pd.DataFrame | None:
    """Extrae importancia (o coeficientes) del clasificador final.

    Devuelve un DataFrame con columnas ``feature`` e ``importance`` (o
    una columna por clase si son coeficientes multinomiales).
    Devuelve ``None`` si no se puede extraer (modelo sin atributo o
    nombres de features no disponibles).
    """
    try:
        feature_names = estimator.named_steps["preprocessor"].get_feature_names_out()
    except Exception:  # noqa: BLE001
        return None

    clf = estimator.named_steps["clf"]

    # Caso 1: feature_importances_ (DecisionTree, RandomForest, XGB wrapper)
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "_xgb") and clf._xgb is not None and hasattr(clf._xgb, "feature_importances_"):
        # _XGBClassifierSafe expone el XGB underlying en `_xgb`.
        importances = clf._xgb.feature_importances_

    if importances is not None:
        if len(importances) != len(feature_names):
            return None
        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    # Caso 2: coef_ (LogReg, LinearSVC)
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if coef.ndim == 1:
            df = pd.DataFrame({"feature": feature_names, "coef": coef})
            df["abs_coef"] = df["coef"].abs()
            return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
        # Multinomial: una fila por clase
        rows = []
        classes = getattr(clf, "classes_", [f"class_{i}" for i in range(coef.shape[0])])
        for c_idx, cls in enumerate(classes):
            for f_idx, fname in enumerate(feature_names):
                rows.append({"class": cls, "feature": fname, "coef": float(coef[c_idx, f_idx])})
        df = pd.DataFrame(rows)
        df["abs_coef"] = df["coef"].abs()
        return df.sort_values(["class", "abs_coef"], ascending=[True, False]).reset_index(drop=True)

    return None


# ---------------------------------------------------------------------------
# Carga y preparación del dataset
# ---------------------------------------------------------------------------
def load_tabular_modeling_dataset(parquet_path: str | Path | None = None) -> pd.DataFrame:
    """Carga el parquet del dataset filtrado (no aplica nuevos filtros)."""
    if parquet_path is None:
        parquet_path = PROCESSED_DIR / TABULAR_DATASET_FILENAME
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero "
            "scripts/02_build_filtered_tabular_modeling_dataset.py."
        )
    return pd.read_parquet(path)
