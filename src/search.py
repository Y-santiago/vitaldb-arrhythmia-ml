"""[LEGACY — pipeline exploratorio ECG crudo, pausado en esta fase]

Búsqueda de hiperparámetros multi-modelo y multi-ventana sobre features
derivadas de señal ECG cruda (ventaneo + estadísticas temporales + RR).

ESTADO: línea exploratoria histórica. NO se usa como flujo principal de
modelado en la fase actual. Para el flujo activo, ver
``src/tabular_search.py`` (modelado tabular sobre anotaciones + metadata
filtradas, sin descarga ni ventaneo de ECG).

Razón del pausado: con la cohorte real disponible y el tiempo de descarga
de los 482 ECG desde VitalDB, el flujo basado en señal cruda no era viable
para entregar un baseline reproducible en plazo razonable. El flujo
tabular se construye únicamente sobre lo que ya está en disco bajo
``data/raw/physionet_annotations/`` y no necesita VitalDB en runtime.

------------------------------------------------------------------------

Este módulo orquesta el pipeline completo de modelado:
    1. Carga el parquet de features de un tamaño de ventana dado.
    2. Construye ``X``, ``y``, ``groups`` excluyendo columnas prohibidas
       (``beat_type``, ``rhythm_label``, ``case_id``, etc.).
    3. Hace el split 80/20 por ``case_id`` con cobertura de clases vía
       :func:`src.modeling.make_train_test_group_split_with_coverage`.
    4. Para cada modelo del :data:`MODEL_REGISTRY`, ejecuta
       :class:`sklearn.model_selection.RandomizedSearchCV` con CV por grupo
       (StratifiedGroupKFold si es viable, GroupKFold como fallback).
    5. Evalúa el modelo *una sola vez* sobre el test congelado.
    6. Devuelve estructuras serializables para escribir reportes.

El módulo no escribe a disco por sí mismo. El script
``scripts/03_run_hyperparameter_search.py`` y el notebook
``notebooks/06_full_modeling_hyperparameter_search.ipynb`` consumen estas
funciones y persisten los outputs en ``reports/``.

Restricciones metodológicas:
    * ``beat_type`` está prohibido como feature
      (:data:`config.FORBIDDEN_FEATURE_COLUMNS`).
    * El test no se toca durante la búsqueda; solo al final.
    * Toda CV interna es por ``case_id`` (sin fuga entre folds).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import (
    BEAT_TIME_COLUMN,
    CASE_ID_COLUMN,
    FORBIDDEN_FEATURE_COLUMNS,
    PROCESSED_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from .modeling import (
    assert_no_forbidden_features,
    make_train_test_group_split_with_coverage,
)


# ---------------------------------------------------------------------------
# Configuración: métricas reportadas y columnas que no son features
# ---------------------------------------------------------------------------
SCORING_METRICS: dict[str, str] = {
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_weighted": "f1_weighted",
}

PRIMARY_SCORING: str = "f1_macro"

# Metadatos de cada ventana que no deben entrar al modelo.
NON_FEATURE_METADATA_COLUMNS: tuple[str, ...] = (
    "beat_index",
    "start_sample",
    "end_sample",
    "window_seconds",
    BEAT_TIME_COLUMN,
)


# ---------------------------------------------------------------------------
# Registro de modelos
# ---------------------------------------------------------------------------
def _logreg_factory() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _decision_tree_factory() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _random_forest_factory() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class _XGBClassifierSafe(BaseEstimator, ClassifierMixin):
    """Wrapper de XGBClassifier compatible con sklearn ≥ 1.6 + XGBoost ≥ 2.0.

    Hereda de ``BaseEstimator`` y ``ClassifierMixin`` para satisfacer el
    protocolo ``__sklearn_tags__`` introducido en sklearn 1.6.

    XGBoost ≥ 2.0 requiere etiquetas enteras consecutivas [0, num_class).
    Este wrapper aplica ``LabelEncoder`` en ``fit`` y lo revierte en
    ``predict``/``predict_proba``, manteniendo etiquetas string en la API
    externa y siendo completamente transparente para ``Pipeline`` +
    ``RandomizedSearchCV``.

    ``BaseEstimator.get_params()`` / ``set_params()`` funcionan
    correctamente porque todos los parámetros de ``__init__`` están
    declarados explícitamente y asignados a ``self.<mismo_nombre>``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1,
        tree_method: str = "hist",
        device: str = "cpu",
        random_state: int | None = None,
        n_jobs: int = 1,
        eval_metric: str = "mlogloss",
        verbosity: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.tree_method = tree_method
        self.device = device
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.verbosity = verbosity

    def fit(self, X, y, **kwargs):
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._xgb = XGBClassifier(**self.get_params())
        self._xgb.fit(X, y_enc, **kwargs)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        y_pred_enc = self._xgb.predict(X)
        return self._le.inverse_transform(np.asarray(y_pred_enc, dtype=int))

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)


def _detect_cuda() -> tuple[bool, str]:
    """Detecta GPU NVIDIA via nvidia-smi. Sin dependencias externas.

    Returns (cuda_disponible, nombre_gpu).
    """
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            gpu_name = r.stdout.strip().split("\n")[0]
            if gpu_name:
                return True, gpu_name
    except Exception:
        pass
    return False, ""


def _xgboost_factory() -> Pipeline:
    """XGBoost envuelto con re-encoding seguro por fit (ver :class:`_XGBClassifierSafe`).

    Detecta CUDA automáticamente: si hay GPU disponible usa ``device='cuda'``
    y ``n_jobs=1`` (la GPU maneja el paralelismo internamente).
    """
    _cuda, _ = _detect_cuda()
    _device  = "cuda" if _cuda else "cpu"
    _njobs   = 1 if _cuda else -1
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                _XGBClassifierSafe(
                    tree_method="hist",
                    device=_device,
                    random_state=RANDOM_SEED,
                    n_jobs=_njobs,
                    eval_metric="mlogloss",
                    verbosity=0,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.3,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    min_child_weight=1,
                ),
            ),
        ]
    )


def _linear_svc_factory() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    max_iter=5000,
                    dual="auto",
                ),
            ),
        ]
    )


def _mlp_factory() -> Pipeline:
    """MLP con etiquetas string.

    ``early_stopping=True`` invoca internamente ``np.isnan`` sobre las
    predicciones, lo cual falla cuando las etiquetas son strings (sklearn
    issue conocido). Lo dejamos ``False`` y compensamos con un
    ``max_iter`` razonable.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    random_state=RANDOM_SEED,
                    max_iter=300,
                    early_stopping=False,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pipeline_factory: Any
    param_distributions: dict[str, Any]
    needs_label_encoding: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        name="logreg",
        pipeline_factory=_logreg_factory,
        param_distributions={
            "clf__C": loguniform(1e-3, 1e2),
        },
    ),
    "decision_tree": ModelSpec(
        name="decision_tree",
        pipeline_factory=_decision_tree_factory,
        param_distributions={
            "clf__max_depth": [None, 3, 5, 7, 10, 15, 20, 30],
            "clf__min_samples_split": [2, 5, 10, 20, 50],
            "clf__min_samples_leaf": [1, 2, 5, 10, 20],
            "clf__criterion": ["gini", "entropy"],
        },
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        pipeline_factory=_random_forest_factory,
        param_distributions={
            "clf__n_estimators": [100, 200, 300, 500, 800],
            "clf__max_depth": [None, 5, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__max_features": ["sqrt", "log2", 0.5, 0.8],
        },
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        pipeline_factory=_xgboost_factory,
        param_distributions={
            "clf__n_estimators": [100, 200, 400, 800],
            "clf__max_depth": [3, 5, 7, 10],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5],
        },
        needs_label_encoding=False,  # El re-encoding ahora vive en el wrapper.
    ),
    "linear_svc": ModelSpec(
        name="linear_svc",
        pipeline_factory=_linear_svc_factory,
        param_distributions={
            "clf__C": loguniform(1e-3, 1e2),
        },
    ),
    "mlp": ModelSpec(
        name="mlp",
        pipeline_factory=_mlp_factory,
        param_distributions={
            "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [5e-4, 1e-3, 5e-3],
            "clf__activation": ["relu", "tanh"],
        },
    ),
}


# ---------------------------------------------------------------------------
# Preparación de datasets
# ---------------------------------------------------------------------------
def prepare_dataset_for_modeling(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = CASE_ID_COLUMN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Construye ``X``, ``y``, ``groups`` y la lista de feature names.

    Excluye explícitamente columnas prohibidas (``FORBIDDEN_FEATURE_COLUMNS``)
    y metadatos por ventana (``beat_index``, ``start_sample``, etc.). Aborta
    si alguna columna prohibida llega a entrar como feature.
    """
    if target_col not in df.columns:
        raise KeyError(f"Falta la columna objetivo: {target_col!r}")
    if group_col not in df.columns:
        raise KeyError(f"Falta la columna de grupo: {group_col!r}")

    non_feature = set(FORBIDDEN_FEATURE_COLUMNS) | set(NON_FEATURE_METADATA_COLUMNS)
    feature_cols = [c for c in df.columns if c not in non_feature]
    assert_no_forbidden_features(feature_cols)

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    groups = df[group_col].to_numpy()
    return X, y, groups, feature_cols


def load_feature_dataset(parquet_path: str | Path) -> pd.DataFrame:
    """Carga un parquet de features y limpia etiquetas inválidas."""
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    mask = (
        df[TARGET_COLUMN]
        .astype(str).str.strip().str.lower()
        .isin({"nan", "none", ""})
    )
    df = df.loc[~mask].copy()
    return df


# ---------------------------------------------------------------------------
# CV: StratifiedGroupKFold con fallback a GroupKFold
# ---------------------------------------------------------------------------
def build_cv_splitter(groups_train: np.ndarray,
                      y_train: np.ndarray,
                      n_splits: int,
                      prefer_stratified: bool = True):
    """Construye el iterador de folds para CV interna.

    Intenta usar ``StratifiedGroupKFold`` (requiere sklearn ≥ 1.0). Si no
    está disponible o no es viable, cae a ``GroupKFold``. ``n_splits`` se
    recorta automáticamente al número de grupos en train.
    """
    n_groups = int(np.unique(groups_train).shape[0])
    n_splits_eff = max(2, min(int(n_splits), n_groups))

    if prefer_stratified:
        try:
            from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-not-found]

            # StratifiedGroupKFold puede fallar si una clase tiene menos
            # ejemplos que folds. Probamos y caemos a GroupKFold si lanza.
            cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_SEED)
            _ = list(cv.split(np.zeros((len(y_train), 1)), y_train, groups=groups_train))
            return cv, "StratifiedGroupKFold", n_splits_eff
        except Exception:  # noqa: BLE001 — caemos a GroupKFold
            pass

    return GroupKFold(n_splits=n_splits_eff), "GroupKFold", n_splits_eff


# ---------------------------------------------------------------------------
# Ejecución de la búsqueda
# ---------------------------------------------------------------------------
def run_search_for_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cv,
    n_iter: int,
    scoring: dict[str, str] = SCORING_METRICS,
    refit: str = PRIMARY_SCORING,
    random_state: int = RANDOM_SEED,
    n_jobs: int = -1,
) -> dict:
    """Corre ``RandomizedSearchCV`` para un modelo y devuelve resultados."""
    pipe = spec.pipeline_factory()
    search = RandomizedSearchCV(
        pipe,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score=np.nan,
        return_train_score=False,
        verbose=0,
    )

    t0 = time.time()
    # El re-encoding (cuando aplica) lo hace el wrapper del modelo en su .fit().
    # Aquí pasamos siempre las etiquetas originales para no romper el scoring.
    le = None
    search.fit(X_train, y_train, groups=groups_train)
    fit_seconds = time.time() - t0

    best_idx = int(search.best_index_)
    cv_results = search.cv_results_

    cv_metrics = {}
    for metric_name in scoring:
        col = f"mean_test_{metric_name}"
        if col in cv_results:
            cv_metrics[f"cv_{metric_name}"] = float(cv_results[col][best_idx])
        else:
            cv_metrics[f"cv_{metric_name}"] = float("nan")

    return {
        "model": spec.name,
        "best_params": search.best_params_,
        "best_cv_score_primary": float(search.best_score_),
        "fit_seconds": fit_seconds,
        "best_estimator": search.best_estimator_,
        "label_encoder": le,
        "cv_metrics": cv_metrics,
        "n_iter_effective": int(len(cv_results["params"])),
    }


def evaluate_on_test(result: dict,
                     X_test: np.ndarray,
                     y_test: np.ndarray) -> dict:
    """Predice una sola vez sobre el test congelado y calcula métricas."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    estimator = result["best_estimator"]
    y_pred = estimator.predict(X_test)

    metrics = {
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    return {"y_pred": y_pred, **metrics}


# ---------------------------------------------------------------------------
# Orquestación full: una ventana × N modelos
# ---------------------------------------------------------------------------
@dataclass
class WindowRunConfig:
    window_seconds: float
    parquet_path: Path
    models_to_run: list[str]
    n_iter: int
    n_splits: int
    test_size: float = 0.2
    random_state: int = RANDOM_SEED
    n_jobs: int = -1
    split_max_attempts: int = 200


def run_one_window(cfg: WindowRunConfig) -> dict:
    """Ejecuta el pipeline completo para un único tamaño de ventana.

    Devuelve un dict con:
        * ``window_seconds``
        * ``split_info`` (del split with-coverage)
        * ``cv_info`` (tipo de splitter, n_splits efectivo)
        * ``models``: lista de resultados por modelo (best_params, CV
          metrics, test metrics, y_pred)
        * ``y_test``, ``groups_test``: para reportes posteriores
        * ``feature_cols``
        * ``parquet_path``, ``shape``
    """
    df = load_feature_dataset(cfg.parquet_path)
    X, y, groups, feature_cols = prepare_dataset_for_modeling(df)

    train_idx, test_idx, split_info = make_train_test_group_split_with_coverage(
        X, y, groups,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        max_attempts=cfg.split_max_attempts,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    cv, cv_name, n_splits_eff = build_cv_splitter(
        groups_train=groups_train,
        y_train=y_train,
        n_splits=cfg.n_splits,
        prefer_stratified=True,
    )

    model_results: list[dict] = []
    for model_name in cfg.models_to_run:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Modelo desconocido: {model_name!r}")
        spec = MODEL_REGISTRY[model_name]
        try:
            res = run_search_for_model(
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                cv=cv,
                n_iter=cfg.n_iter,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
            )
            test = evaluate_on_test(res, X_test, y_test)
            res["test_metrics"] = {k: v for k, v in test.items() if k != "y_pred"}
            res["y_pred_test"] = test["y_pred"]
            res["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 — el resto de modelos debe seguir
            res = {
                "model": model_name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "best_params": None,
                "best_cv_score_primary": float("nan"),
                "cv_metrics": {f"cv_{m}": float("nan") for m in SCORING_METRICS},
                "test_metrics": {f"test_{m}": float("nan") for m in SCORING_METRICS},
                "fit_seconds": float("nan"),
                "y_pred_test": None,
                "best_estimator": None,
                "label_encoder": None,
                "n_iter_effective": 0,
            }
        model_results.append(res)

    return {
        "window_seconds": cfg.window_seconds,
        "parquet_path": str(cfg.parquet_path),
        "shape": df.shape,
        "split_info": split_info,
        "cv_info": {"splitter": cv_name, "n_splits_effective": n_splits_eff},
        "feature_cols": feature_cols,
        "models": model_results,
        "y_test": y_test,
        "groups_test": groups_test,
    }


# ---------------------------------------------------------------------------
# Persistencia de resultados a CSV / figuras
# ---------------------------------------------------------------------------
def _flatten_for_csv(window_result: dict, model_result: dict) -> dict:
    """Aplana un resultado por (ventana, modelo) en una fila para CSV."""
    row = {
        "window_seconds": window_result["window_seconds"],
        "model": model_result["model"],
        "status": model_result.get("status", "ok"),
        "n_iter_effective": model_result.get("n_iter_effective", 0),
        "fit_seconds": model_result.get("fit_seconds", float("nan")),
        "best_cv_score_primary": model_result.get("best_cv_score_primary", float("nan")),
    }
    row.update(model_result.get("cv_metrics", {}))
    row.update(model_result.get("test_metrics", {}))
    return row


def assemble_comparison_table(window_results: list[dict]) -> pd.DataFrame:
    """Tabla comparativa global: una fila por (ventana, modelo)."""
    rows = []
    for wr in window_results:
        for mr in wr["models"]:
            rows.append(_flatten_for_csv(wr, mr))
    return pd.DataFrame(rows)


def assemble_best_hyperparameters_table(window_results: list[dict]) -> pd.DataFrame:
    """Tabla de hiperparámetros del mejor candidato por (ventana, modelo)."""
    rows = []
    for wr in window_results:
        for mr in wr["models"]:
            rows.append(
                {
                    "window_seconds": wr["window_seconds"],
                    "model": mr["model"],
                    "status": mr.get("status", "ok"),
                    "best_params_json": json.dumps(mr.get("best_params"), default=str),
                }
            )
    return pd.DataFrame(rows)


def assemble_class_support_table(window_results: list[dict]) -> pd.DataFrame:
    """Soporte por clase en train/test para cada tamaño de ventana."""
    from .evaluation import class_support_per_split

    rows = []
    for wr in window_results:
        # Reconstruir y_train / y_test mirando los grupos del split.
        # No los persistimos en el dict para evitar duplicación; los recuperamos
        # de los y_pred del primer modelo OK y de y_test (que sí guardamos).
        # Lo más limpio: cargar de nuevo el parquet y aplicar el mismo split.
        df = load_feature_dataset(wr["parquet_path"])
        X, y, groups, _ = prepare_dataset_for_modeling(df)
        train_idx, test_idx, _ = make_train_test_group_split_with_coverage(
            X, y, groups,
            test_size=wr["split_info"]["requested_test_size"],
            random_state=wr["split_info"]["chosen_seed"],
            max_attempts=1,  # mismo seed exacto -> mismo split
        )
        sup = class_support_per_split(y[train_idx], y[test_idx])
        # sup tiene index = clases + 'TOTAL' y columnas train/test/total
        for cls, r in sup.iterrows():
            rows.append(
                {
                    "window_seconds": wr["window_seconds"],
                    "class": cls,
                    "train": int(r["train"]),
                    "test": int(r["test"]),
                    "total": int(r["total"]),
                }
            )
    return pd.DataFrame(rows)


def assemble_missing_classes_table(window_results: list[dict]) -> pd.DataFrame:
    """Clases ausentes en train o en test, por ventana."""
    rows = []
    for wr in window_results:
        info = wr["split_info"]
        for cls in info.get("classes_only_in_train", []):
            rows.append(
                {
                    "window_seconds": wr["window_seconds"],
                    "class": cls,
                    "missing_in": "test",
                }
            )
        for cls in info.get("classes_only_in_test", []):
            rows.append(
                {
                    "window_seconds": wr["window_seconds"],
                    "class": cls,
                    "missing_in": "train",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["window_seconds", "class", "missing_in"])
    return pd.DataFrame(rows)


def pick_best_overall(comparison_df: pd.DataFrame,
                      primary: str = "test_f1_macro") -> dict | None:
    """Selecciona el mejor (ventana, modelo) por una métrica de test.

    Filtra filas con status != 'ok'. Devuelve un dict con la fila ganadora
    o ``None`` si no hay candidatos válidos.
    """
    ok = comparison_df.loc[comparison_df["status"] == "ok"].copy()
    if ok.empty or primary not in ok.columns:
        return None
    ok = ok.dropna(subset=[primary])
    if ok.empty:
        return None
    winner = ok.sort_values(primary, ascending=False).iloc[0]
    return winner.to_dict()
