"""Train and select the independent Model B binary rhythm pipeline.

This script compares multiple candidate models, runs hyperparameter search on
train only with grouped CV by `case_id`, evaluates the held-out test split once
for the selected winner, and saves app-ready artifacts.

Examples:
    python model_b_pipeline/train_model_b.py --debug
    python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5
    python model_b_pipeline/train_model_b.py --models logreg_balanced sgd_log_loss
"""

from __future__ import annotations

import argparse
import inspect
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_b_pipeline import config_model_b as cfg  # noqa: E402
from model_b_pipeline.utils_model_b import (  # noqa: E402
    compute_binary_metrics,
    ensure_dir,
    get_numeric_and_categorical_features,
    plot_confusion_matrix,
    save_csv,
    save_json,
    select_threshold_max_f1,
    select_threshold_youden_j,
)


SCORING_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "precision_abnormal",
    "recall_abnormal_sensitivity",
    "specificity_normal",
    "f1_abnormal",
    "roc_auc",
    "average_precision",
]


class XGBoostBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible wrapper that keeps Model B string labels.

    `scale_pos_weight` is computed inside `fit()` from the training fold only,
    so CV folds do not leak label distribution from held-out data.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = cfg.RANDOM_STATE,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "xgboost is not installed or could not be imported. "
                "Install xgboost or omit --include-xgboost."
            ) from exc

        y_arr = np.asarray(y)
        y_bin = (y_arr == cfg.POSITIVE_CLASS).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("XGBoostBinaryClassifier needs both binary classes in fit().")
        scale_pos_weight = n_neg / n_pos
        self.classes_ = np.asarray(cfg.CLASS_LABELS, dtype=object)
        self._model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        self._model.fit(X, y_bin)
        return self

    def predict_proba(self, X):
        proba_pos = self._model.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        proba_pos = self.predict_proba(X)[:, 1]
        return np.where(proba_pos >= 0.5, cfg.POSITIVE_CLASS, cfg.NEGATIVE_CLASS)


def make_group_train_test_split_with_coverage(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = cfg.RANDOM_STATE,
    max_attempts: int = 500,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create an external train/test split by `case_id`.

    The seed is not chosen by model performance. Retries only enforce group
    disjointness and presence of both binary classes in both partitions.
    """
    y = df[cfg.TARGET_COLUMN].to_numpy()
    groups = df[cfg.CASE_ID_COLUMN].to_numpy()
    required_classes = set(cfg.CLASS_LABELS)

    for offset in range(max_attempts):
        seed = random_state + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, y, groups=groups))
        y_train = set(pd.Series(y[train_idx]).dropna().unique())
        y_test = set(pd.Series(y[test_idx]).dropna().unique())
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            continue
        if required_classes.issubset(y_train) and required_classes.issubset(y_test):
            return train_idx, test_idx, {
                "chosen_seed": seed,
                "attempt": offset + 1,
                "test_size_requested": test_size,
                "actual_test_fraction_rows": float(len(test_idx) / len(df)),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_train_cases": int(len(train_groups)),
                "n_test_cases": int(len(test_groups)),
                "n_overlap_case_id": 0,
                "no_case_overlap": True,
                "classes_train": sorted(y_train),
                "classes_test": sorted(y_test),
            }

    raise RuntimeError(
        "Could not create a group train/test split with both classes in train "
        f"and test after {max_attempts} attempts."
    )


def _model_b_feature_groups(
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return numeric/categorical Model B features without adding predictors."""
    features = features or cfg.FEATURES_MODEL_B
    if df is not None:
        numeric, categorical = get_numeric_and_categorical_features(df, features)
        if "sex" in features and "sex" not in categorical:
            numeric = [feature for feature in numeric if feature != "sex"]
            categorical = categorical + ["sex"]
        return numeric, categorical
    categorical = [feature for feature in cfg.CATEGORICAL_FEATURES_MODEL_B if feature in features]
    numeric = [feature for feature in features if feature not in categorical]
    return numeric, categorical


def build_preprocessor(
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the preprocessing transformer included in every saved pipeline."""
    features = features or cfg.FEATURES_MODEL_B
    numeric_features, categorical_features = _model_b_feature_groups(df, features)
    encoder_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**encoder_kwargs)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    ), numeric_features, categorical_features


def build_cv_splitter(n_splits: int, random_state: int):
    """Prefer StratifiedGroupKFold; fall back to GroupKFold if unavailable."""
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    except Exception:
        return GroupKFold(n_splits=n_splits)


def _hist_gradient_boosting_supports_class_weight() -> bool:
    return "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters


def _pipeline_for(clf) -> Pipeline:
    preprocessor, _, _ = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])


def build_model_b_registry(
    include_random_forest: bool = False,
    include_xgboost: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return Model B candidate estimators and hyperparameter spaces."""
    hgb_kwargs: dict[str, Any] = {"random_state": cfg.RANDOM_STATE}
    if _hist_gradient_boosting_supports_class_weight():
        hgb_kwargs["class_weight"] = "balanced"

    registry: dict[str, dict[str, Any]] = {
        "dummy_most_frequent": {
            "pipeline": _pipeline_for(DummyClassifier(strategy="most_frequent")),
            "params": {"clf__strategy": ["most_frequent"]},
            "n_iter": 1,
            "uses_sample_weight": False,
            "description": "Baseline DummyClassifier(strategy='most_frequent')",
            "feature_columns": cfg.FEATURES_MODEL_B,
        },
        "logreg_balanced": {
            "pipeline": _pipeline_for(LogisticRegression(class_weight="balanced", max_iter=3000)),
            "params": {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__solver": ["lbfgs"],
            },
            "uses_sample_weight": False,
            "description": "LogisticRegression(class_weight='balanced')",
            "feature_columns": cfg.FEATURES_MODEL_B,
        },
        "sgd_log_loss": {
            "pipeline": _pipeline_for(SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=cfg.RANDOM_STATE,
                max_iter=2000,
                tol=1e-3,
            )),
            "params": {
                "clf__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                "clf__penalty": ["l2", "l1", "elasticnet"],
                "clf__l1_ratio": [0.15, 0.3, 0.5, 0.7],
            },
            "uses_sample_weight": False,
            "description": "SGDClassifier(loss='log_loss', class_weight='balanced')",
            "feature_columns": cfg.FEATURES_MODEL_B,
        },
        "hist_gradient_boosting": {
            "pipeline": _pipeline_for(HistGradientBoostingClassifier(**hgb_kwargs)),
            "params": {
                "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "clf__max_iter": [100, 200, 300],
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [20, 50, 100],
                "clf__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
            "uses_sample_weight": not _hist_gradient_boosting_supports_class_weight(),
            "description": "HistGradientBoostingClassifier",
            "feature_columns": cfg.FEATURES_MODEL_B,
        },
    }

    if include_random_forest:
        registry["random_forest_balanced"] = {
            "pipeline": _pipeline_for(RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
            )),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 5, 10],
                "clf__max_features": ["sqrt", "log2"],
            },
            "uses_sample_weight": False,
            "description": "RandomForestClassifier(class_weight='balanced_subsample')",
            "feature_columns": cfg.FEATURES_MODEL_B,
        }

    if include_xgboost:
        registry["xgboost_binary"] = {
            "pipeline": _pipeline_for(XGBoostBinaryClassifier()),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_lambda": [0.1, 1.0, 3.0],
            },
            "uses_sample_weight": False,
            "description": "Optional XGBoost with train-fold scale_pos_weight",
            "feature_columns": cfg.FEATURES_MODEL_B,
        }
    return registry


def _get_positive_score(estimator, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        if cfg.POSITIVE_CLASS not in classes:
            return None
        return np.asarray(proba[:, classes.index(cfg.POSITIVE_CLASS)])
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(X)
        classes = list(estimator.classes_)
        if len(classes) == 2 and classes[-1] != cfg.POSITIVE_CLASS:
            score = -score
        return np.asarray(score)
    return None


def _predict_from_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(y_score >= threshold, cfg.POSITIVE_CLASS, cfg.NEGATIVE_CLASS)


def _default_threshold_for_scores(y_score: np.ndarray) -> float:
    if np.nanmin(y_score) >= 0.0 and np.nanmax(y_score) <= 1.0:
        return 0.5
    return 0.0


def _score_with_predictions(metric_name: str, estimator, X, y) -> float:
    y_pred = estimator.predict(X)
    return float(compute_binary_metrics(y, y_pred)[metric_name])


def _score_roc_auc(estimator, X, y) -> float:
    score = _get_positive_score(estimator, X)
    if score is None:
        return float("nan")
    y_bin = (np.asarray(y) == cfg.POSITIVE_CLASS).astype(int)
    return float(roc_auc_score(y_bin, score))


def _score_average_precision(estimator, X, y) -> float:
    score = _get_positive_score(estimator, X)
    if score is None:
        return float("nan")
    y_bin = (np.asarray(y) == cfg.POSITIVE_CLASS).astype(int)
    return float(average_precision_score(y_bin, score))


def score_balanced_accuracy(estimator, X, y) -> float:
    return _score_with_predictions("balanced_accuracy", estimator, X, y)


def score_accuracy(estimator, X, y) -> float:
    return _score_with_predictions("accuracy", estimator, X, y)


def score_precision_abnormal(estimator, X, y) -> float:
    return _score_with_predictions("precision_abnormal", estimator, X, y)


def score_recall_abnormal_sensitivity(estimator, X, y) -> float:
    return _score_with_predictions("recall_abnormal_sensitivity", estimator, X, y)


def score_specificity_normal(estimator, X, y) -> float:
    return _score_with_predictions("specificity_normal", estimator, X, y)


def score_f1_abnormal(estimator, X, y) -> float:
    return _score_with_predictions("f1_abnormal", estimator, X, y)


def build_scoring() -> dict[str, Any]:
    """Scoring dictionary used by RandomizedSearchCV."""
    return {
        "balanced_accuracy": score_balanced_accuracy,
        "accuracy": score_accuracy,
        "precision_abnormal": score_precision_abnormal,
        "recall_abnormal_sensitivity": score_recall_abnormal_sensitivity,
        "specificity_normal": score_specificity_normal,
        "f1_abnormal": score_f1_abnormal,
        "roc_auc": _score_roc_auc,
        "average_precision": _score_average_precision,
    }


def _subsample_cases(df: pd.DataFrame, max_cases: int, random_state: int) -> pd.DataFrame:
    all_cases = df[cfg.CASE_ID_COLUMN].drop_duplicates().to_numpy()
    if max_cases >= len(all_cases):
        return df.copy()
    rng = np.random.RandomState(random_state)
    selected = rng.choice(all_cases, size=max_cases, replace=False)
    return df.loc[df[cfg.CASE_ID_COLUMN].isin(selected)].copy()


def _load_model_b_dataset() -> pd.DataFrame:
    if not cfg.MODEL_B_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Missing reduced Model B dataset: "
            f"{cfg.MODEL_B_DATASET_PATH}. Run "
            "python model_b_pipeline/build_model_b_dataset.py first."
        )
    df = pd.read_parquet(cfg.MODEL_B_DATASET_PATH)
    cfg.validate_model_b_feature_set(df)
    return df


def normalize_model_selection(arg_models: list[str] | None) -> list[str] | None:
    """Support both '--models a b' and '--models a,b' forms."""
    if not arg_models:
        return None
    selected: list[str] = []
    for item in arg_models:
        selected.extend(part.strip() for part in item.split(",") if part.strip())
    return selected or None


def select_model_names(arg_models: list[str] | None, registry: dict[str, dict[str, Any]]) -> list[str]:
    selected = normalize_model_selection(arg_models)
    if selected is None:
        return list(registry.keys())
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(
            "Unknown Model B models: "
            f"{unknown}. Valid models for this run: {sorted(registry)}"
        )
    return selected


def _save_split_tables(
    split_info: dict[str, Any],
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
) -> dict[str, Any]:
    overlap = sorted(set(groups_train) & set(groups_test))
    split_info = dict(split_info)
    split_info["n_overlap_case_id"] = int(len(overlap))
    split_info["no_case_overlap"] = len(overlap) == 0

    split_summary = pd.DataFrame([
        {"metric": key, "value": value}
        for key, value in split_info.items()
        if key not in {"classes_train", "classes_test"}
    ] + [
        {"metric": "classes_train", "value": ",".join(split_info["classes_train"])},
        {"metric": "classes_test", "value": ",".join(split_info["classes_test"])},
    ])
    save_csv(split_summary, cfg.TABLES_DIR / "model_b_train_test_split_summary.csv")

    support_rows = []
    for label in cfg.CLASS_LABELS:
        support_rows.append({
            "rhythm_binary": label,
            "train_rows": int((y_train == label).sum()),
            "test_rows": int((y_test == label).sum()),
            "train_cases_with_class": int(len(set(groups_train[y_train == label]))),
            "test_cases_with_class": int(len(set(groups_test[y_test == label]))),
        })
    save_csv(pd.DataFrame(support_rows), cfg.TABLES_DIR / "model_b_class_support_train_test.csv")
    save_csv(pd.DataFrame([{
        "n_overlap_case_id": int(len(overlap)),
        "overlap_case_id_sample": ",".join(map(str, overlap[:20])),
        "no_overlap": len(overlap) == 0,
    }]), cfg.TABLES_DIR / "model_b_case_overlap_check.csv")
    return split_info


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, sort_keys=True)
    except Exception:
        return str(value)


def _flatten_cv_results(model_name: str, search: RandomizedSearchCV) -> pd.DataFrame:
    rows = []
    result = pd.DataFrame(search.cv_results_)
    for _, row in result.iterrows():
        out = {"model": model_name}
        for column, value in row.items():
            if column == "params":
                out[column] = _jsonish(value)
            elif column.startswith("param_"):
                out[column] = _jsonish(value)
            else:
                out[column] = value
        rows.append(out)
    return pd.DataFrame(rows)


def _cv_result_row(
    model_name: str,
    search: RandomizedSearchCV,
    selection_metric: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    best_idx = int(search.best_index_)
    row: dict[str, Any] = {
        "model": model_name,
        "status": "ok",
        "best_params": search.best_params_,
        "selection_metric": selection_metric,
        f"best_cv_{selection_metric}": float(search.best_score_),
        "elapsed_seconds": float(elapsed_seconds),
    }
    for key, value in search.cv_results_.items():
        if key.startswith("mean_test_"):
            row[key.replace("mean_test_", "cv_mean_")] = float(value[best_idx])
        if key.startswith("std_test_"):
            row[key.replace("std_test_", "cv_std_")] = float(value[best_idx])
    return row


def _threshold_rows_for_model(model_name: str, estimator, X_train: pd.DataFrame, y_train: np.ndarray) -> list[dict]:
    y_score_train = _get_positive_score(estimator, X_train)
    if y_score_train is None:
        return [{
            "model": model_name,
            "threshold_name": "default",
            "threshold": "",
            "threshold_selection_data": "train",
            "train_balanced_accuracy": "",
            "train_f1_abnormal": "",
            "notes": "model does not expose predict_proba or decision_function",
        }]
    threshold_specs = [
        {"threshold_name": "default", "threshold": _default_threshold_for_scores(y_score_train)},
        {"threshold_name": "youden_j_train", **select_threshold_youden_j(y_train, y_score_train)},
        {"threshold_name": "max_f1_train", **select_threshold_max_f1(y_train, y_score_train)},
    ]
    rows = []
    for threshold_spec in threshold_specs:
        threshold = float(threshold_spec["threshold"])
        pred_train = _predict_from_threshold(y_score_train, threshold)
        train_metrics = compute_binary_metrics(y_train, pred_train, y_score_train)
        rows.append({
            "model": model_name,
            "threshold_name": threshold_spec["threshold_name"],
            "threshold": threshold,
            "threshold_selection_data": "train",
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_f1_abnormal": train_metrics["f1_abnormal"],
            "train_recall_abnormal_sensitivity": train_metrics["recall_abnormal_sensitivity"],
            "train_specificity_normal": train_metrics["specificity_normal"],
            "notes": "threshold selected on train only",
        })
    return rows


def _choose_operating_threshold(estimator, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, Any]:
    y_score_train = _get_positive_score(estimator, X_train)
    if y_score_train is None:
        return {
            "method": "default",
            "threshold": None,
            "score_kind": "predict",
            "positive_class": cfg.POSITIVE_CLASS,
            "negative_class": cfg.NEGATIVE_CLASS,
        }
    info = select_threshold_youden_j(y_train, y_score_train)
    return {
        "method": "youden_j_train",
        "score_kind": "predict_proba_or_decision_function",
        "positive_class": cfg.POSITIVE_CLASS,
        "negative_class": cfg.NEGATIVE_CLASS,
        **info,
    }


def _predict_with_operating_threshold(estimator, X: pd.DataFrame, threshold_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    y_score = _get_positive_score(estimator, X)
    if y_score is None or threshold_info.get("threshold") is None:
        return estimator.predict(X), y_score
    return _predict_from_threshold(y_score, float(threshold_info["threshold"])), y_score


def _plot_roc_and_pr(y_true: np.ndarray, y_score: np.ndarray, roc_path: Path, pr_path: Path) -> None:
    y_true_bin = (np.asarray(y_true) == cfg.POSITIVE_CLASS).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Model B ROC curve")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=140)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Model B precision-recall curve")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=140)
    plt.close(fig)


def _save_feature_columns(numeric_features: list[str], categorical_features: list[str]) -> Path:
    return save_json({
        "feature_columns": cfg.FEATURES_MODEL_B,
        "features": cfg.FEATURES_MODEL_B,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": cfg.TARGET_COLUMN,
        "positive_class": cfg.POSITIVE_CLASS,
        "negative_class": cfg.NEGATIVE_CLASS,
    }, cfg.MODELS_MODEL_B_DIR / "model_b_feature_columns.json")


def _save_app_artifacts(
    estimator,
    metadata: dict[str, Any],
    threshold_info: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, str]:
    ensure_dir(cfg.MODELS_MODEL_B_DIR)
    pipeline_path = cfg.MODELS_MODEL_B_DIR / "model_b_best_pipeline.joblib"
    metadata_path = cfg.MODELS_MODEL_B_DIR / "model_b_metadata.json"
    threshold_path = cfg.MODELS_MODEL_B_DIR / "model_b_threshold.json"
    feature_path = cfg.MODELS_MODEL_B_DIR / "model_b_feature_columns.json"
    joblib.dump(estimator, pipeline_path)
    _save_feature_columns(numeric_features, categorical_features)
    save_json(threshold_info, threshold_path)
    save_json(metadata, metadata_path)
    return {
        "pipeline": str(pipeline_path.relative_to(cfg.PROJECT_ROOT)),
        "feature_columns": str(feature_path.relative_to(cfg.PROJECT_ROOT)),
        "metadata": str(metadata_path.relative_to(cfg.PROJECT_ROOT)),
        "threshold": str(threshold_path.relative_to(cfg.PROJECT_ROOT)),
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(cfg.TABLES_DIR)
    ensure_dir(cfg.FIGURES_DIR)
    ensure_dir(cfg.MODELS_MODEL_B_DIR)

    if args.selection_metric not in SCORING_METRICS:
        raise ValueError(
            f"Unsupported --selection-metric {args.selection_metric!r}. "
            f"Valid values: {SCORING_METRICS}"
        )

    df = _load_model_b_dataset()
    if args.debug:
        args.n_iter = 3
        args.n_splits = 2
        if args.max_cases is None:
            args.max_cases = 80
    if args.max_cases is not None:
        df = _subsample_cases(df, args.max_cases, args.random_state)
        cfg.validate_model_b_feature_set(df)

    X = df[cfg.FEATURES_MODEL_B].copy()
    y = df[cfg.TARGET_COLUMN].to_numpy()
    groups = df[cfg.CASE_ID_COLUMN].to_numpy()

    train_idx, test_idx, split_info = make_group_train_test_split_with_coverage(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    split_info = _save_split_tables(split_info, y_train, y_test, groups_train, groups_test)

    _, numeric_features, categorical_features = build_preprocessor(df, cfg.FEATURES_MODEL_B)
    registry = build_model_b_registry(
        include_random_forest=args.include_random_forest,
        include_xgboost=args.include_xgboost,
    )
    selected_models = select_model_names(args.models, registry)
    cv = build_cv_splitter(args.n_splits, args.random_state)
    scoring = build_scoring()

    cv_rows: list[dict[str, Any]] = []
    cv_results_all: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    best_estimators: dict[str, Pipeline] = {}
    best_params_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    for model_name in selected_models:
        spec = registry[model_name]
        n_iter = max(1, min(int(args.n_iter), int(spec.get("n_iter", args.n_iter))))
        search = RandomizedSearchCV(
            estimator=spec["pipeline"],
            param_distributions=spec["params"],
            n_iter=n_iter,
            scoring=scoring,
            refit=args.selection_metric,
            cv=cv,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            return_train_score=False,
            error_score=np.nan,
        )
        fit_params = {}
        if spec.get("uses_sample_weight"):
            fit_params["clf__sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        started = time.perf_counter()
        try:
            search.fit(X_train, y_train, groups=groups_train, **fit_params)
            elapsed = time.perf_counter() - started
            cv_rows.append(_cv_result_row(model_name, search, args.selection_metric, elapsed))
            cv_results_all.append(_flatten_cv_results(model_name, search))
            best_estimators[model_name] = search.best_estimator_
            for param, value in search.best_params_.items():
                best_params_rows.append({
                    "model": model_name,
                    "param": param,
                    "value": value,
                    "is_winner": False,
                })
            threshold_rows.extend(_threshold_rows_for_model(model_name, search.best_estimator_, X_train, y_train))
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            failure_rows.append({
                "model": model_name,
                "status": "failed",
                "elapsed_seconds": float(elapsed),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            })
            print(f"[Model B] Model failed and will be skipped: {model_name}: {exc}")

    failures_df = pd.DataFrame(failure_rows, columns=[
        "model",
        "status",
        "elapsed_seconds",
        "error_type",
        "error_message",
        "traceback",
    ])
    save_csv(failures_df, cfg.TABLES_DIR / "model_b_model_failures.csv")
    if cv_results_all:
        save_csv(pd.concat(cv_results_all, ignore_index=True), cfg.TABLES_DIR / "model_b_cv_results_all.csv")
    else:
        save_csv(pd.DataFrame(), cfg.TABLES_DIR / "model_b_cv_results_all.csv")

    if not cv_rows:
        raise RuntimeError("All requested Model B candidate models failed. See model_b_model_failures.csv.")

    selection_col = f"best_cv_{args.selection_metric}"
    cv_df = pd.DataFrame(cv_rows).sort_values(selection_col, ascending=False).reset_index(drop=True)
    winner = str(cv_df.iloc[0]["model"])
    for row in best_params_rows:
        row["is_winner"] = row["model"] == winner

    save_csv(cv_df, cfg.TABLES_DIR / "model_b_model_comparison_cv.csv")
    save_csv(pd.DataFrame(best_params_rows), cfg.TABLES_DIR / "model_b_best_hyperparameters.csv")
    save_csv(pd.DataFrame(threshold_rows), cfg.TABLES_DIR / "model_b_threshold_analysis.csv")

    best_estimator = best_estimators[winner]
    threshold_info = _choose_operating_threshold(best_estimator, X_train, y_train)
    y_pred_best, y_score_best_test = _predict_with_operating_threshold(best_estimator, X_test, threshold_info)
    best_metrics = compute_binary_metrics(y_test, y_pred_best, y_score_best_test)

    test_df = pd.DataFrame([{
        "model": winner,
        "prediction_rule": threshold_info["method"],
        **best_metrics,
    }])
    save_csv(test_df, cfg.TABLES_DIR / "model_b_model_comparison_test.csv")

    report_dict = classification_report(
        y_test,
        y_pred_best,
        labels=cfg.CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    save_csv(
        pd.DataFrame(report_dict).transpose().reset_index(names="label"),
        cfg.TABLES_DIR / "model_b_best_model_classification_report.csv",
    )

    cm_abs = confusion_matrix(y_test, y_pred_best, labels=cfg.CLASS_LABELS)
    cm_norm = confusion_matrix(y_test, y_pred_best, labels=cfg.CLASS_LABELS, normalize="true")
    save_csv(
        pd.DataFrame(cm_abs, index=cfg.CLASS_LABELS, columns=cfg.CLASS_LABELS).reset_index(names="true_label"),
        cfg.TABLES_DIR / "model_b_confusion_matrix_absolute.csv",
    )
    save_csv(
        pd.DataFrame(cm_norm, index=cfg.CLASS_LABELS, columns=cfg.CLASS_LABELS).reset_index(names="true_label"),
        cfg.TABLES_DIR / "model_b_confusion_matrix_normalized.csv",
    )
    plot_confusion_matrix(
        y_test,
        y_pred_best,
        cfg.FIGURES_DIR / "model_b_confusion_matrix_absolute.png",
        normalize=False,
        title="Model B confusion matrix",
    )
    plot_confusion_matrix(
        y_test,
        y_pred_best,
        cfg.FIGURES_DIR / "model_b_confusion_matrix_normalized.png",
        normalize=True,
        title="Model B normalized confusion matrix",
    )
    if y_score_best_test is not None:
        _plot_roc_and_pr(
            y_test,
            y_score_best_test,
            cfg.FIGURES_DIR / "model_b_roc_curve_best_model.png",
            cfg.FIGURES_DIR / "model_b_precision_recall_curve_best_model.png",
        )

    metadata: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_b_pipeline_version": cfg.MODEL_B_PIPELINE_VERSION,
        "script": "model_b_pipeline/train_model_b.py",
        "debug": bool(args.debug),
        "features": cfg.FEATURES_MODEL_B,
        "target_column": cfg.TARGET_COLUMN,
        "positive_class": cfg.POSITIVE_CLASS,
        "negative_class": cfg.NEGATIVE_CLASS,
        "models_requested": selected_models,
        "models_succeeded": list(cv_df["model"]),
        "models_failed": failure_rows,
        "winning_model": winner,
        "selection_metric": args.selection_metric,
        "selection_rule": (
            f"highest mean CV {args.selection_metric} on train folds grouped by case_id"
        ),
        "selection_metric_cv_value": float(cv_df.iloc[0][selection_col]),
        "best_params": cv_df.iloc[0]["best_params"],
        "test_metrics": best_metrics,
        "threshold_used": threshold_info,
        "threshold_method": threshold_info["method"],
        "random_state": int(args.random_state),
        "split_seed": int(split_info["chosen_seed"]),
        "test_size": float(args.test_size),
        "n_train_cases": int(split_info["n_train_cases"]),
        "n_test_cases": int(split_info["n_test_cases"]),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_total_cases_used": int(df[cfg.CASE_ID_COLUMN].nunique()),
        "n_total_rows_used": int(len(df)),
        "n_overlap_case_id": int(split_info["n_overlap_case_id"]),
        "no_case_overlap": bool(split_info["no_case_overlap"]),
        "model_b_dataset": str(cfg.MODEL_B_DATASET_PATH.relative_to(cfg.PROJECT_ROOT)),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "save_model": bool(args.save_model),
    }

    if args.save_model:
        artifacts = _save_app_artifacts(
            best_estimator,
            metadata,
            threshold_info,
            numeric_features,
            categorical_features,
        )
        metadata["artifacts"] = artifacts
        metadata["pipeline_path"] = artifacts["pipeline"]
        save_json(metadata, cfg.MODELS_MODEL_B_DIR / "model_b_metadata.json")
    else:
        save_json(metadata, cfg.TABLES_DIR / "model_b_training_metadata_no_model.json")

    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Use max_cases=80, n_iter=3, n_splits=2.")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--models", nargs="*", default=None, help="Model names separated by spaces or commas.")
    parser.add_argument("--include-random-forest", action="store_true", help="Include optional random_forest_balanced.")
    parser.add_argument("--include-xgboost", action="store_true", help="Include optional xgboost_binary.")
    parser.add_argument("--selection-metric", default="balanced_accuracy", choices=SCORING_METRICS)
    parser.set_defaults(save_model=True)
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-model", dest="save_model", action="store_true", help="Save app artifacts (default).")
    save_group.add_argument("--no-save-model", dest="save_model", action="store_false", help="Do not save model artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = run_training(args)
    print("Model B training completed.")
    print(f"Winning model by CV {metadata['selection_metric']}: {metadata['winning_model']}")
    if metadata.get("save_model"):
        print(f"Best pipeline: {cfg.MODELS_MODEL_B_DIR / 'model_b_best_pipeline.joblib'}")
        print(f"Threshold: {cfg.MODELS_MODEL_B_DIR / 'model_b_threshold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
