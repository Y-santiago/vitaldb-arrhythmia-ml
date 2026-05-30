"""Búsqueda de hiperparámetros para la tarea binaria
`normal_sinus` vs `arrhythmia_or_abnormal`.

Reutiliza el preprocesador tabular (`build_tabular_preprocessor`) y el
split robusto por grupo (`make_train_test_group_split_with_coverage`), y
añade:
    * un registro de 12 modelos pensados para tareas binarias con
      desbalance moderado, con manejo gracioso de dependencias opcionales
      (`imbalanced-learn`, `lightgbm`, `catboost`);
    * scoring binario (`balanced_accuracy`, `f1`, `roc_auc`,
      `average_precision`, `recall`, `precision`);
    * un wrapper de XGBoost compatible con sklearn ≥ 1.6;
    * helpers para selección de umbral por Youden J / F1 en train.

NO usa señal ECG cruda. NO usa `beat_type`, `case_id` ni el target como
features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    BINARY_LEAKAGE_COLUMNS,
    BINARY_NEGATIVE_CLASS,
    BINARY_POSITIVE_CLASS,
    BINARY_TARGET_COLUMN,
    CASE_ID_COLUMN,
    PROCESSED_DIR,
    RANDOM_SEED,
    TABULAR_MAX_CATEGORY_CARDINALITY,
    TABULAR_OHE_MIN_FREQUENCY,
    TARGET_COLUMN,
)
from .modeling import (
    assert_no_forbidden_features,
    make_train_test_group_split_with_coverage,
)
from .preprocessing import build_tabular_preprocessor

# Metadatos por fila que no deben entrar como features.
BINARY_NON_FEATURE_METADATA_COLUMNS: tuple[str, ...] = (
    "beat_index",
    "start_sample",
    "end_sample",
    "window_seconds",
)


def _build_scoring_dict() -> dict[str, Any]:
    """Scoring dict robusto a etiquetas string para CV interna.

    ``f1`` / ``precision`` / ``recall`` requieren ``pos_label`` porque los
    scorers genéricos asumen ``pos_label=1`` y fallan con etiquetas string.
    ``balanced_accuracy`` y ``accuracy`` no lo requieren. ``roc_auc`` y
    ``average_precision`` se omiten de la CV interna porque dependen de
    `predict_proba`/`decision_function` y no todos los modelos del registro
    (p. ej. ``dummy_most_frequent``, ``LinearSVC`` con sklearn ≥ 1.5) los
    exponen de forma uniforme; estas dos métricas se calculan únicamente
    sobre el test al final.
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, make_scorer,
    )
    pos = BINARY_POSITIVE_CLASS
    return {
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, pos_label=pos, zero_division=0),
        "precision": make_scorer(precision_score, pos_label=pos, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=pos, zero_division=0),
    }


BINARY_SCORING_METRICS: dict[str, Any] = _build_scoring_dict()
PRIMARY_BINARY_SCORING: str = "balanced_accuracy"


# ---------------------------------------------------------------------------
# Wrapper XGBoost binario
# ---------------------------------------------------------------------------
class _XGBBinaryClassifierSafe(BaseEstimator, ClassifierMixin):
    """XGBoost binario con label-encoding interno.

    Mapea ``BINARY_NEGATIVE_CLASS`` -> 0 y ``BINARY_POSITIVE_CLASS`` -> 1
    en ``fit``, e invierte en ``predict``. Acepta ``scale_pos_weight``
    calculado externamente (a partir del train) o ``"auto"`` para que se
    calcule dentro de cada fit como `n_negative / n_positive`.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        scale_pos_weight: float | str = "auto",
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        random_state: int | None = None,
        n_jobs: int = -1,
        eval_metric: str = "logloss",
        verbosity: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.verbosity = verbosity

    def fit(self, X, y, **kwargs):
        from xgboost import XGBClassifier

        # Map strings -> 0/1 (negative=0, positive=1).
        y_arr = np.asarray(y)
        y_enc = np.where(y_arr == BINARY_POSITIVE_CLASS, 1, 0).astype(int)

        if self.scale_pos_weight == "auto":
            n_pos = int(y_enc.sum())
            n_neg = int(len(y_enc) - n_pos)
            spw = float(n_neg / max(n_pos, 1))
        else:
            spw = float(self.scale_pos_weight)

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "scale_pos_weight": spw,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "eval_metric": self.eval_metric,
            "verbosity": self.verbosity,
            "objective": "binary:logistic",
        }
        self._xgb = XGBClassifier(**params)
        self._xgb.fit(X, y_enc, **kwargs)
        self.classes_ = np.array([BINARY_NEGATIVE_CLASS, BINARY_POSITIVE_CLASS])
        return self

    def predict(self, X):
        y_enc = self._xgb.predict(X)
        return np.where(np.asarray(y_enc, dtype=int) == 1,
                        BINARY_POSITIVE_CLASS, BINARY_NEGATIVE_CLASS)

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)

    def decision_function(self, X):
        # Devuelve la prob de la clase positiva: útil para calcular scores
        # cuando algunos modelos solo expongan `predict_proba`.
        return self.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Registro de modelos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinaryModelSpec:
    name: str
    pipeline_factory: Callable[[list[str], list[str]], Pipeline]
    param_distributions: dict[str, Any]
    notes: str = ""


def _make_pipe(numeric: list[str], categorical: list[str], clf) -> Pipeline:
    pre = build_tabular_preprocessor(
        numeric_features=numeric,
        categorical_features=categorical,
        with_scaling=True,
        ohe_min_frequency=TABULAR_OHE_MIN_FREQUENCY,
    )
    return Pipeline(steps=[("preprocessor", pre), ("clf", clf)])


# --- Fábricas individuales ---
def _dummy_factory(numeric, categorical):
    return _make_pipe(numeric, categorical, DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED))


def _logreg_factory(numeric, categorical):
    return _make_pipe(
        numeric, categorical,
        LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ),
    )


def _sgd_log_loss_factory(numeric, categorical):
    return _make_pipe(
        numeric, categorical,
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=200,
            n_jobs=-1,
        ),
    )


def _linear_svc_factory(numeric, categorical):
    return _make_pipe(
        numeric, categorical,
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5000,
            dual="auto",
        ),
    )


def _hist_gb_factory(numeric, categorical):
    return _make_pipe(
        numeric, categorical,
        HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )


def _rf_factory(numeric, categorical):
    return _make_pipe(
        numeric, categorical,
        RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )


def _extra_trees_factory(numeric, categorical):
    return _make_pipe(
        numeric, categorical,
        ExtraTreesClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )


def _balanced_rf_factory(numeric, categorical):
    from imblearn.ensemble import BalancedRandomForestClassifier
    return _make_pipe(
        numeric, categorical,
        BalancedRandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            sampling_strategy="auto",
            replacement=True,
            bootstrap=False,
        ),
    )


def _easy_ensemble_factory(numeric, categorical):
    from imblearn.ensemble import EasyEnsembleClassifier
    return _make_pipe(
        numeric, categorical,
        EasyEnsembleClassifier(random_state=RANDOM_SEED, n_jobs=-1),
    )


def _xgboost_factory(numeric, categorical):
    return _make_pipe(numeric, categorical, _XGBBinaryClassifierSafe(random_state=RANDOM_SEED))


def _lightgbm_factory(numeric, categorical):
    from lightgbm import LGBMClassifier
    return _make_pipe(
        numeric, categorical,
        LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
    )


def _catboost_factory(numeric, categorical):
    from catboost import CatBoostClassifier
    return _make_pipe(
        numeric, categorical,
        CatBoostClassifier(
            auto_class_weights="Balanced",
            random_state=RANDOM_SEED,
            verbose=False,
        ),
    )


def _is_available(import_name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(import_name) is not None


def build_binary_model_registry() -> dict[str, BinaryModelSpec]:
    """Construye el registro saltando modelos cuya dependencia opcional falte."""
    reg: dict[str, BinaryModelSpec] = {}

    reg["dummy_most_frequent"] = BinaryModelSpec(
        name="dummy_most_frequent",
        pipeline_factory=_dummy_factory,
        param_distributions={"clf__strategy": ["most_frequent"]},
        notes="Baseline obligatorio (clase dominante).",
    )

    reg["logreg_balanced"] = BinaryModelSpec(
        name="logreg_balanced",
        pipeline_factory=_logreg_factory,
        param_distributions={"clf__C": loguniform(1e-3, 1e2)},
        notes="LogisticRegression con class_weight='balanced'.",
    )

    reg["sgd_log_loss"] = BinaryModelSpec(
        name="sgd_log_loss",
        pipeline_factory=_sgd_log_loss_factory,
        param_distributions={
            "clf__alpha": loguniform(1e-6, 1e-2),
            "clf__l1_ratio": [0.0, 0.15, 0.5, 0.85, 1.0],
            "clf__penalty": ["l2", "elasticnet"],
        },
        notes="SGD con log_loss; rápido sobre datasets grandes.",
    )

    reg["linear_svc_balanced"] = BinaryModelSpec(
        name="linear_svc_balanced",
        pipeline_factory=_linear_svc_factory,
        param_distributions={"clf__C": loguniform(1e-3, 1e2)},
        notes="LinearSVC con class_weight='balanced'; sin predict_proba nativo.",
    )

    reg["hist_gradient_boosting"] = BinaryModelSpec(
        name="hist_gradient_boosting",
        pipeline_factory=_hist_gb_factory,
        param_distributions={
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__max_iter": [100, 200, 400],
            "clf__max_depth": [None, 4, 6, 8, 10],
            "clf__min_samples_leaf": [10, 20, 50, 100],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        },
        notes="HistGradientBoosting nativo de sklearn con class_weight.",
    )

    reg["random_forest_balanced"] = BinaryModelSpec(
        name="random_forest_balanced",
        pipeline_factory=_rf_factory,
        param_distributions={
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2", 0.5],
        },
        notes="RandomForest con class_weight='balanced_subsample'.",
    )

    reg["extra_trees_balanced"] = BinaryModelSpec(
        name="extra_trees_balanced",
        pipeline_factory=_extra_trees_factory,
        param_distributions={
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2", 0.5],
        },
        notes="ExtraTrees con class_weight='balanced'.",
    )

    if _is_available("imblearn"):
        reg["balanced_random_forest"] = BinaryModelSpec(
            name="balanced_random_forest",
            pipeline_factory=_balanced_rf_factory,
            param_distributions={
                "clf__n_estimators": [200, 400, 800],
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_leaf": [1, 2, 5],
            },
            notes="BalancedRandomForest (imbalanced-learn).",
        )
        reg["easy_ensemble"] = BinaryModelSpec(
            name="easy_ensemble",
            pipeline_factory=_easy_ensemble_factory,
            param_distributions={
                "clf__n_estimators": [10, 20],
            },
            notes="EasyEnsemble (imbalanced-learn). Lento; pocas iter.",
        )

    if _is_available("xgboost"):
        reg["xgboost_binary"] = BinaryModelSpec(
            name="xgboost_binary",
            pipeline_factory=_xgboost_factory,
            param_distributions={
                "clf__n_estimators": [200, 400, 800],
                "clf__max_depth": [4, 6, 8, 10],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
                "clf__min_child_weight": [1, 3, 5],
            },
            notes="XGBoost binario con scale_pos_weight auto sobre train.",
        )

    if _is_available("lightgbm"):
        reg["lightgbm_binary"] = BinaryModelSpec(
            name="lightgbm_binary",
            pipeline_factory=_lightgbm_factory,
            param_distributions={
                "clf__n_estimators": [200, 400, 800],
                "clf__num_leaves": [31, 63, 127],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__min_child_samples": [10, 20, 50],
            },
            notes="LightGBM binario con class_weight='balanced'.",
        )

    if _is_available("catboost"):
        reg["catboost_binary"] = BinaryModelSpec(
            name="catboost_binary",
            pipeline_factory=_catboost_factory,
            param_distributions={
                "clf__iterations": [200, 400],
                "clf__depth": [4, 6, 8],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__l2_leaf_reg": [1, 3, 5],
            },
            notes="CatBoost binario con auto_class_weights='Balanced'.",
        )

    return reg


# ---------------------------------------------------------------------------
# Carga + clasificación de columnas
# ---------------------------------------------------------------------------
def load_binary_modeling_dataset(parquet_path: str | None = None) -> pd.DataFrame:
    from pathlib import Path
    from .config import BINARY_DATASET_FILENAME
    if parquet_path is None:
        path = PROCESSED_DIR / BINARY_DATASET_FILENAME
    else:
        path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta scripts/05_build_binary_rhythm_modeling_dataset.py."
        )
    return pd.read_parquet(path)


def classify_binary_features(
    df: pd.DataFrame,
    leakage: Iterable[str] = BINARY_LEAKAGE_COLUMNS,
    max_categorical_cardinality: int = TABULAR_MAX_CATEGORY_CARDINALITY,
    max_missing_pct: float = 99.0,
) -> dict[str, list[str]]:
    """Separa columnas y bloquea las prohibidas en el set de predictores."""
    leakage_set = set(leakage) | set(BINARY_NON_FEATURE_METADATA_COLUMNS)
    numeric: list[str] = []
    categorical: list[str] = []
    leak: list[str] = []
    high_card: list[str] = []
    constant: list[str] = []
    too_missing: list[str] = []
    for col in df.columns:
        if col in leakage_set:
            leak.append(col)
            continue
        n_unique = int(df[col].nunique(dropna=True))
        if n_unique <= 1:
            constant.append(col)
            continue
        miss_pct = float(df[col].isna().mean() * 100)
        if miss_pct > max_missing_pct:
            too_missing.append(col)
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

    # Bloqueo final: verifica que ninguna columna prohibida se filtró.
    assert_no_forbidden_features(numeric + categorical, forbidden=leakage_set)
    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "leakage_excluded": leak,
        "high_cardinality_excluded": high_card,
        "constant_excluded": constant,
        "too_missing_excluded": too_missing,
    }


# ---------------------------------------------------------------------------
# Split por grupo asegurando ambas clases binarias
# ---------------------------------------------------------------------------
def make_binary_group_train_test_split_with_coverage(
    X, y, groups,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    max_attempts: int = 500,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split 80/20 por `case_id` que garantiza ambas clases en train y test.

    Reutiliza :func:`make_train_test_group_split_with_coverage` y luego
    verifica que las dos clases binarias estén en ambos lados; si la
    primera elección las deja en un solo lado, prueba semillas adicionales.
    """
    train_idx, test_idx, info = make_train_test_group_split_with_coverage(
        X, y, groups,
        test_size=test_size, random_state=random_state, max_attempts=max_attempts,
    )
    train_classes = set(np.asarray(y)[train_idx])
    test_classes = set(np.asarray(y)[test_idx])
    expected = {BINARY_NEGATIVE_CLASS, BINARY_POSITIVE_CLASS}
    info["binary_classes_in_train"] = sorted(train_classes)
    info["binary_classes_in_test"] = sorted(test_classes)
    info["binary_coverage_ok"] = expected.issubset(train_classes) and expected.issubset(test_classes)
    if not info["binary_coverage_ok"]:
        info["binary_coverage_warning"] = (
            "El split no cubre ambas clases binarias en ambos lados. "
            "Sube `max_attempts` o revisa la composición de casos."
        )
    return train_idx, test_idx, info


# ---------------------------------------------------------------------------
# CV builder
# ---------------------------------------------------------------------------
def build_binary_cv_splitter(groups_train: np.ndarray,
                             y_train: np.ndarray,
                             n_splits: int,
                             prefer_stratified: bool = True):
    n_groups = int(np.unique(groups_train).shape[0])
    n_splits_eff = max(2, min(int(n_splits), n_groups))
    if prefer_stratified:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=RANDOM_SEED)
            _ = list(cv.split(np.zeros((len(y_train), 1)), y_train, groups=groups_train))
            return cv, "StratifiedGroupKFold", n_splits_eff
        except Exception:  # noqa: BLE001
            pass
    return GroupKFold(n_splits=n_splits_eff), "GroupKFold", n_splits_eff


# ---------------------------------------------------------------------------
# Search por modelo
# ---------------------------------------------------------------------------
@dataclass
class BinaryModelResult:
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
    y_score_test: np.ndarray | None = None
    chosen_threshold: float | None = None
    threshold_method: str | None = None
    error: str | None = None


def run_binary_search_for_model(
    spec: BinaryModelSpec,
    X_train,
    y_train,
    groups_train,
    numeric_features: list[str],
    categorical_features: list[str],
    cv,
    n_iter: int,
    random_state: int = RANDOM_SEED,
    n_jobs: int = -1,
) -> BinaryModelResult:
    pipe = spec.pipeline_factory(numeric_features, categorical_features)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring=BINARY_SCORING_METRICS,
        refit=PRIMARY_BINARY_SCORING,
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
    cvr = search.cv_results_
    cv_metrics = {}
    for metric in BINARY_SCORING_METRICS:
        col = f"mean_test_{metric}"
        cv_metrics[f"cv_{metric}"] = (
            float(cvr[col][best_idx]) if col in cvr else float("nan")
        )

    return BinaryModelResult(
        model=spec.name,
        status="ok",
        best_params=dict(search.best_params_),
        best_cv_score_primary=float(search.best_score_),
        cv_metrics=cv_metrics,
        test_metrics={},
        fit_seconds=fit_seconds,
        n_iter_effective=int(len(cvr["params"])),
        best_estimator=search.best_estimator_,
    )


# ---------------------------------------------------------------------------
# Scores y umbral
# ---------------------------------------------------------------------------
def get_positive_class_score(estimator, X) -> np.ndarray | None:
    """Devuelve un score continuo para la clase positiva.

    Usa ``predict_proba`` si está disponible, ``decision_function`` como
    fallback. Devuelve ``None`` si no hay forma de obtener scores
    (p. ej. DummyClassifier sin proba).
    """
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            classes = getattr(estimator, "classes_", None)
            if classes is None and hasattr(estimator, "steps"):
                classes = estimator[-1].classes_
            if classes is None:
                # asumir orden alfabético: aabnormal < normal
                # pero más seguro: asumir índice 1 = positive si proba.shape[1] == 2
                return proba[:, 1]
            idx = list(classes).index(BINARY_POSITIVE_CLASS)
            return proba[:, idx]
        except Exception:  # noqa: BLE001
            pass
    if hasattr(estimator, "decision_function"):
        try:
            return np.asarray(estimator.decision_function(X)).ravel()
        except Exception:  # noqa: BLE001
            pass
    return None


def select_threshold_youden_j(y_true_bin: np.ndarray,
                              scores: np.ndarray) -> tuple[float, float]:
    """Selecciona umbral que maximiza Youden J = sensitivity + specificity - 1.

    Trabaja sobre `y_true_bin` en {0, 1} (positivo = 1).
    Devuelve (mejor_umbral, mejor_J).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true_bin, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(j[idx])


def select_threshold_max_f1(y_true_bin: np.ndarray,
                            scores: np.ndarray) -> tuple[float, float]:
    """Selecciona umbral que maximiza F1 de la clase positiva."""
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_true_bin, scores)
    # precision/recall tienen len(thresholds)+1 puntos. Calculamos F1 alineado.
    p = precision[:-1]
    r = recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
        f1 = np.where(np.isfinite(f1), f1, 0.0)
    if f1.size == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])
