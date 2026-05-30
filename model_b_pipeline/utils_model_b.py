"""Utility functions for the independent Model B pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from model_b_pipeline import config_model_b as cfg


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_base_binary_dataset() -> pd.DataFrame:
    """Load the preferred base binary dataset for Model B.

    This function intentionally does not rebuild upstream datasets. If the
    preferred parquet is absent, the caller gets a clear failure telling them
    which previous flow must be run first.
    """
    if not cfg.BASE_BINARY_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Missing base dataset for Model B: "
            f"{cfg.BASE_BINARY_DATASET_PATH}. Run the previous broad binary "
            "rhythm flow first so it creates "
            "data/processed/binary_rhythm_modeling_dataset.parquet."
        )
    return pd.read_parquet(cfg.BASE_BINARY_DATASET_PATH)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as CSV after creating the parent directory."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def save_json(obj: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON object with stable formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    return path


def get_numeric_and_categorical_features(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], list[str]]:
    """Split Model B features into numeric and categorical columns."""
    numeric: list[str] = []
    categorical: list[str] = []
    for feature in features:
        if pd.api.types.is_bool_dtype(df[feature]):
            categorical.append(feature)
        elif pd.api.types.is_numeric_dtype(df[feature]):
            numeric.append(feature)
        else:
            categorical.append(feature)
    return numeric, categorical


def _positive_scores_are_valid(y_true: pd.Series | np.ndarray) -> bool:
    return len(pd.unique(pd.Series(y_true).dropna())) == 2


def compute_binary_metrics(
    y_true,
    y_pred,
    y_score=None,
    positive_label: str = cfg.POSITIVE_CLASS,
) -> dict[str, float | int]:
    """Compute binary metrics with explicit sensitivity and specificity."""
    negative_label = cfg.NEGATIVE_CLASS
    labels = [negative_label, positive_label]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.size != 4:
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_abnormal": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "recall_abnormal_sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "specificity_normal": specificity,
        "f1_abnormal": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "n_tn": int(tn),
        "n_fp": int(fp),
        "n_fn": int(fn),
        "n_tp": int(tp),
    }
    # Keep the direct calculation visible for auditability.
    metrics["recall_abnormal_sensitivity"] = sensitivity

    if y_score is not None and _positive_scores_are_valid(y_true):
        y_true_bin = (np.asarray(y_true) == positive_label).astype(int)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["average_precision"] = float(
                average_precision_score(y_true_bin, y_score)
            )
        except ValueError:
            metrics["average_precision"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    normalize: bool = False,
    title: str | None = None,
) -> Path:
    """Plot a binary confusion matrix with the fixed Model B label order."""
    labels = [cfg.NEGATIVE_CLASS, cfg.POSITIVE_CLASS]
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )
    fmt = ".2f" if normalize else "d"
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Normalized confusion matrix" if normalize else "Confusion matrix"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def select_threshold_youden_j(
    y_true,
    y_score,
    positive_label: str = cfg.POSITIVE_CLASS,
) -> dict[str, float]:
    """Select the threshold maximizing Youden's J statistic on given scores."""
    y_true_bin = (np.asarray(y_true) == positive_label).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return {"threshold": 0.5, "youden_j": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    scores = tpr - fpr
    scores = np.where(finite_mask, scores, -np.inf)
    idx = int(np.argmax(scores))
    return {
        "threshold": float(thresholds[idx]),
        "youden_j": float(scores[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(1.0 - fpr[idx]),
    }


def select_threshold_max_f1(
    y_true,
    y_score,
    positive_label: str = cfg.POSITIVE_CLASS,
) -> dict[str, float]:
    """Select the threshold maximizing positive-class F1 on given scores."""
    y_true_bin = (np.asarray(y_true) == positive_label).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    idx = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }
