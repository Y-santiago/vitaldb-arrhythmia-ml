"""Train the independent Model B binary rhythm classifiers.

Usage:
    python model_b_pipeline/train_model_b.py --debug
    python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
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


def make_group_train_test_split_with_coverage(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = cfg.RANDOM_STATE,
    max_attempts: int = 500,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create an 80/20 train/test split by case_id with both classes present."""
    y = df[cfg.TARGET_COLUMN].to_numpy()
    groups = df[cfg.CASE_ID_COLUMN].to_numpy()
    required_classes = set(cfg.CLASS_LABELS)

    for offset in range(max_attempts):
        seed = random_state + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(df, y, groups=groups))
        y_train = set(pd.Series(y[train_idx]).dropna().unique())
        y_test = set(pd.Series(y[test_idx]).dropna().unique())
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        if train_groups & test_groups:
            continue
        if required_classes.issubset(y_train) and required_classes.issubset(y_test):
            info = {
                "chosen_seed": seed,
                "attempt": offset + 1,
                "test_size_requested": test_size,
                "actual_test_fraction_rows": float(len(test_idx) / len(df)),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_train_cases": int(len(train_groups)),
                "n_test_cases": int(len(test_groups)),
                "classes_train": sorted(y_train),
                "classes_test": sorted(y_test),
            }
            return train_idx, test_idx, info

    raise RuntimeError(
        "Could not create a group train/test split with both classes in "
        f"train and test after {max_attempts} attempts."
    )


def build_preprocessor(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the Model B preprocessing ColumnTransformer."""
    features = features or cfg.FEATURES_MODEL_B
    numeric_features, categorical_features = get_numeric_and_categorical_features(df, features)
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor, numeric_features, categorical_features


def build_cv_splitter(n_splits: int, random_state: int):
    """Prefer StratifiedGroupKFold, with GroupKFold fallback."""
    try:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    except Exception:
        return GroupKFold(n_splits=n_splits)


def _hist_gradient_boosting_supports_class_weight() -> bool:
    return "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters


def build_model_registry(
    df: pd.DataFrame,
    include_random_forest: bool = False,
) -> dict[str, dict]:
    """Build Model B estimators and hyperparameter spaces."""
    preprocessor, _, _ = build_preprocessor(df, cfg.FEATURES_MODEL_B)
    hgb_kwargs = {"random_state": cfg.RANDOM_STATE}
    if _hist_gradient_boosting_supports_class_weight():
        hgb_kwargs["class_weight"] = "balanced"

    registry = {
        "dummy_most_frequent": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]),
            "params": {"clf__strategy": ["most_frequent"]},
            "n_iter": 1,
            "uses_sample_weight": False,
        },
        "logreg_balanced": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=3000)),
            ]),
            "params": {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__solver": ["lbfgs"],
            },
            "uses_sample_weight": False,
        },
        "sgd_log_loss": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    random_state=cfg.RANDOM_STATE,
                    max_iter=2000,
                    tol=1e-3,
                )),
            ]),
            "params": {
                "clf__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                "clf__penalty": ["l2", "elasticnet"],
                "clf__l1_ratio": [0.0, 0.15, 0.5],
            },
            "uses_sample_weight": False,
        },
        "hist_gradient_boosting": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", HistGradientBoostingClassifier(**hgb_kwargs)),
            ]),
            "params": {
                "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "clf__max_iter": [100, 160, 220],
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [20, 50, 100],
                "clf__l2_regularization": [0.0, 0.01, 0.1],
            },
            "uses_sample_weight": not _hist_gradient_boosting_supports_class_weight(),
        },
    }
    if include_random_forest:
        registry["random_forest_balanced"] = {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=cfg.RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            "params": {
                "clf__n_estimators": [150, 250, 400],
                "clf__max_depth": [None, 8, 16, 24],
                "clf__min_samples_leaf": [1, 3, 8],
                "clf__max_features": ["sqrt", 0.7, 1.0],
            },
            "uses_sample_weight": False,
        }
    return registry


def _scoring() -> dict:
    return {
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "precision_abnormal": make_scorer(
            precision_recall_curve_safe_precision,
        ),
        "recall_abnormal_sensitivity": make_scorer(
            precision_recall_curve_safe_recall,
        ),
        "specificity_normal": make_scorer(
            precision_recall_curve_safe_specificity,
        ),
        "f1_abnormal": make_scorer(
            precision_recall_curve_safe_f1,
        ),
    }


def precision_recall_curve_safe_precision(y_true, y_pred) -> float:
    return compute_binary_metrics(y_true, y_pred)["precision_abnormal"]


def precision_recall_curve_safe_recall(y_true, y_pred) -> float:
    return compute_binary_metrics(y_true, y_pred)["recall_abnormal_sensitivity"]


def precision_recall_curve_safe_specificity(y_true, y_pred) -> float:
    return compute_binary_metrics(y_true, y_pred)["specificity_normal"]


def precision_recall_curve_safe_f1(y_true, y_pred) -> float:
    return compute_binary_metrics(y_true, y_pred)["f1_abnormal"]


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


def _select_models(arg_models: str | None, registry: dict[str, dict]) -> list[str]:
    if not arg_models:
        return list(registry.keys())
    selected = [name.strip() for name in arg_models.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(f"Unknown Model B models: {unknown}. Available: {sorted(registry)}")
    return selected


def _get_positive_score(estimator, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        if cfg.POSITIVE_CLASS not in classes:
            return None
        return proba[:, classes.index(cfg.POSITIVE_CLASS)]
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


def _save_split_tables(
    split_info: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
) -> None:
    overlap = sorted(set(groups_train) & set(groups_test))
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
    save_csv(
        pd.DataFrame(support_rows),
        cfg.TABLES_DIR / "model_b_class_support_train_test.csv",
    )
    save_csv(
        pd.DataFrame([{
            "n_overlap_case_id": int(len(overlap)),
            "overlap_case_id_sample": ",".join(map(str, overlap[:20])),
            "no_overlap": len(overlap) == 0,
        }]),
        cfg.TABLES_DIR / "model_b_case_overlap_check.csv",
    )


def _cv_result_row(model_name: str, search: RandomizedSearchCV, elapsed_seconds: float) -> dict:
    best_idx = int(search.best_index_)
    row = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_balanced_accuracy": float(search.best_score_),
        "elapsed_seconds": float(elapsed_seconds),
    }
    for key, value in search.cv_results_.items():
        if key.startswith("mean_test_"):
            row[key.replace("mean_test_", "cv_mean_")] = float(value[best_idx])
        if key.startswith("std_test_"):
            row[key.replace("std_test_", "cv_std_")] = float(value[best_idx])
    return row


def _plot_roc_and_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    roc_path: Path,
    pr_path: Path,
) -> None:
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


def run_training(args: argparse.Namespace) -> dict:
    ensure_dir(cfg.TABLES_DIR)
    ensure_dir(cfg.FIGURES_DIR)
    ensure_dir(cfg.MODELS_MODEL_B_DIR)

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
    _save_split_tables(split_info, y_train, y_test, groups_train, groups_test)

    base_preprocessor, numeric_features, categorical_features = build_preprocessor(
        df,
        cfg.FEATURES_MODEL_B,
    )
    del base_preprocessor

    registry = build_model_registry(df, include_random_forest=args.include_random_forest)
    selected_models = _select_models(args.models, registry)
    cv = build_cv_splitter(args.n_splits, args.random_state)
    scoring = _scoring()

    cv_rows: list[dict] = []
    test_rows: list[dict] = []
    threshold_rows: list[dict] = []
    best_estimators: dict[str, Pipeline] = {}
    best_params_rows: list[dict] = []

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    for model_name in selected_models:
        spec = registry[model_name]
        n_iter = min(int(args.n_iter), int(spec.get("n_iter", args.n_iter)))
        if n_iter < 1:
            n_iter = 1
        search = RandomizedSearchCV(
            estimator=spec["pipeline"],
            param_distributions=spec["params"],
            n_iter=n_iter,
            scoring=scoring,
            refit="balanced_accuracy",
            cv=cv,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        fit_params = {}
        if spec.get("uses_sample_weight"):
            fit_params["clf__sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        started = time.perf_counter()
        search.fit(X_train, y_train, groups=groups_train, **fit_params)
        elapsed = time.perf_counter() - started

        cv_rows.append(_cv_result_row(model_name, search, elapsed))
        best_estimators[model_name] = search.best_estimator_
        for param, value in search.best_params_.items():
            best_params_rows.append({
                "model": model_name,
                "param": param,
                "value": value,
            })

        estimator = search.best_estimator_
        y_pred_default = estimator.predict(X_test)
        y_score_test = _get_positive_score(estimator, X_test)
        metrics = compute_binary_metrics(y_test, y_pred_default, y_score_test)
        test_rows.append({
            "model": model_name,
            "prediction_rule": "default_predict",
            **metrics,
        })

        y_score_train = _get_positive_score(estimator, X_train)
        if y_score_train is not None and y_score_test is not None:
            default_threshold = _default_threshold_for_scores(y_score_train)
            threshold_specs = [
                {"threshold_name": "default_score_threshold", "threshold": default_threshold},
                {"threshold_name": "youden_j_train", **select_threshold_youden_j(y_train, y_score_train)},
                {"threshold_name": "max_f1_train", **select_threshold_max_f1(y_train, y_score_train)},
            ]
            for threshold_spec in threshold_specs:
                threshold = float(threshold_spec["threshold"])
                pred_train = _predict_from_threshold(y_score_train, threshold)
                pred_test = _predict_from_threshold(y_score_test, threshold)
                train_metrics = compute_binary_metrics(y_train, pred_train, y_score_train)
                test_metrics = compute_binary_metrics(y_test, pred_test, y_score_test)
                threshold_rows.append({
                    "model": model_name,
                    "threshold_name": threshold_spec["threshold_name"],
                    "threshold": threshold,
                    "threshold_selection_data": "train",
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_f1_abnormal": train_metrics["f1_abnormal"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_f1_abnormal": test_metrics["f1_abnormal"],
                    "test_recall_abnormal_sensitivity": test_metrics["recall_abnormal_sensitivity"],
                    "test_specificity_normal": test_metrics["specificity_normal"],
                })

    cv_df = pd.DataFrame(cv_rows).sort_values("best_cv_balanced_accuracy", ascending=False)
    test_df = pd.DataFrame(test_rows).sort_values("balanced_accuracy", ascending=False)
    threshold_df = pd.DataFrame(threshold_rows)
    save_csv(cv_df, cfg.TABLES_DIR / "model_b_model_comparison_cv.csv")
    save_csv(test_df, cfg.TABLES_DIR / "model_b_model_comparison_test.csv")
    save_csv(pd.DataFrame(best_params_rows), cfg.TABLES_DIR / "model_b_best_hyperparameters.csv")
    save_csv(threshold_df, cfg.TABLES_DIR / "model_b_threshold_analysis.csv")

    winner = str(cv_df.iloc[0]["model"])
    best_estimator = best_estimators[winner]
    y_score_best_test = _get_positive_score(best_estimator, X_test)
    y_score_best_train = _get_positive_score(best_estimator, X_train)
    if y_score_best_train is not None and y_score_best_test is not None:
        final_threshold_info = select_threshold_youden_j(y_train, y_score_best_train)
        final_threshold_name = "youden_j_train"
        y_pred_best = _predict_from_threshold(
            y_score_best_test,
            float(final_threshold_info["threshold"]),
        )
    else:
        final_threshold_info = {"threshold": None}
        final_threshold_name = "default_predict"
        y_pred_best = best_estimator.predict(X_test)

    best_metrics = compute_binary_metrics(y_test, y_pred_best, y_score_best_test)
    report_dict = classification_report(
        y_test,
        y_pred_best,
        labels=cfg.CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index(names="label")
    save_csv(report_df, cfg.TABLES_DIR / "model_b_best_model_classification_report.csv")

    cm_abs = confusion_matrix(y_test, y_pred_best, labels=cfg.CLASS_LABELS)
    cm_norm = confusion_matrix(y_test, y_pred_best, labels=cfg.CLASS_LABELS, normalize="true")
    cm_abs_df = pd.DataFrame(cm_abs, index=cfg.CLASS_LABELS, columns=cfg.CLASS_LABELS)
    cm_norm_df = pd.DataFrame(cm_norm, index=cfg.CLASS_LABELS, columns=cfg.CLASS_LABELS)
    save_csv(
        cm_abs_df.reset_index(names="true_label"),
        cfg.TABLES_DIR / "model_b_confusion_matrix_absolute.csv",
    )
    save_csv(
        cm_norm_df.reset_index(names="true_label"),
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

    pipeline_path = cfg.MODELS_MODEL_B_DIR / "model_b_best_pipeline.joblib"
    joblib.dump(best_estimator, pipeline_path)
    save_json(
        {
            "features": cfg.FEATURES_MODEL_B,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        cfg.MODELS_MODEL_B_DIR / "model_b_feature_columns.json",
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": "model_b_pipeline/train_model_b.py",
        "debug": bool(args.debug),
        "features": cfg.FEATURES_MODEL_B,
        "winning_model": winner,
        "selection_rule": "highest mean CV balanced_accuracy on train only",
        "best_params": cv_df.iloc[0]["best_params"],
        "test_metrics": best_metrics,
        "threshold_used": {
            "name": final_threshold_name,
            **final_threshold_info,
        },
        "random_state": int(args.random_state),
        "split_seed": int(split_info["chosen_seed"]),
        "test_size": float(args.test_size),
        "n_train_cases": int(split_info["n_train_cases"]),
        "n_test_cases": int(split_info["n_test_cases"]),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_total_cases_used": int(df[cfg.CASE_ID_COLUMN].nunique()),
        "n_total_rows_used": int(len(df)),
        "model_b_dataset": str(cfg.MODEL_B_DATASET_PATH.relative_to(cfg.PROJECT_ROOT)),
        "pipeline_path": str(pipeline_path.relative_to(cfg.PROJECT_ROOT)),
    }
    save_json(metadata, cfg.MODELS_MODEL_B_DIR / "model_b_metadata.json")
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Use fewer cases, n_iter=3, n_splits=2.")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names.")
    parser.add_argument(
        "--include-random-forest",
        action="store_true",
        help="Include optional random_forest_balanced search.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = run_training(args)
    print("Model B training completed.")
    print(f"Winning model by CV balanced accuracy: {metadata['winning_model']}")
    print(f"Best pipeline: {cfg.MODELS_MODEL_B_DIR / 'model_b_best_pipeline.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
