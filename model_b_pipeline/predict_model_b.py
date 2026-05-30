"""Inference helpers for app usage of the saved Model B pipeline.

Typical app usage:

    from model_b_pipeline.predict_model_b import predict_model_b_dataframe

    predictions = predict_model_b_dataframe(input_df)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_b_pipeline import config_model_b as cfg  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Model B artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_b_artifacts(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the saved app artifacts for Model B."""
    models_dir = Path(models_dir) if models_dir is not None else cfg.MODELS_MODEL_B_DIR
    pipeline_path = models_dir / "model_b_best_pipeline.joblib"
    feature_path = models_dir / "model_b_feature_columns.json"
    metadata_path = models_dir / "model_b_metadata.json"
    threshold_path = models_dir / "model_b_threshold.json"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            "Missing Model B pipeline artifact. Expected "
            f"{pipeline_path}. Run python model_b_pipeline/train_model_b.py --debug "
            "or a full training run first."
        )
    return {
        "pipeline": joblib.load(pipeline_path),
        "feature_columns": _read_json(feature_path),
        "metadata": _read_json(metadata_path),
        "threshold": _read_json(threshold_path),
        "paths": {
            "pipeline": pipeline_path,
            "feature_columns": feature_path,
            "metadata": metadata_path,
            "threshold": threshold_path,
        },
    }


def _required_features(feature_columns_artifact: dict[str, Any]) -> list[str]:
    return list(
        feature_columns_artifact.get("feature_columns")
        or feature_columns_artifact.get("features")
        or cfg.FEATURES_MODEL_B
    )


def validate_model_b_input_columns(df: pd.DataFrame, required_features: list[str]) -> None:
    """Raise a clear error if app input lacks required Model B columns."""
    missing = [column for column in required_features if column not in df.columns]
    if missing:
        raise ValueError(
            "Input dataframe is missing Model B required columns: "
            f"{missing}. Expected exactly these predictor columns: {required_features}"
        )


def _get_positive_score(pipeline, X: pd.DataFrame, positive_class: str) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if positive_class not in classes:
            return None
        return np.asarray(proba[:, classes.index(positive_class)])
    if hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X)
        classes = list(pipeline.classes_)
        if len(classes) == 2 and classes[-1] != positive_class:
            score = -score
        return np.asarray(score)
    return None


def _default_threshold_for_scores(score: np.ndarray) -> float:
    if np.nanmin(score) >= 0.0 and np.nanmax(score) <= 1.0:
        return 0.5
    return 0.0


def predict_model_b_dataframe(
    df: pd.DataFrame,
    models_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return Model B app predictions for a dataframe.

    The input dataframe may contain extra columns, but it must contain all 25
    Model B predictors. Columns are ordered according to the saved artifact
    before inference.
    """
    artifacts = load_model_b_artifacts(models_dir=models_dir)
    pipeline = artifacts["pipeline"]
    feature_columns = _required_features(artifacts["feature_columns"])
    metadata = artifacts["metadata"]
    threshold_info = artifacts["threshold"]
    validate_model_b_input_columns(df, feature_columns)

    X = df.loc[:, feature_columns].copy()
    positive_class = metadata.get("positive_class", cfg.POSITIVE_CLASS)
    negative_class = metadata.get("negative_class", cfg.NEGATIVE_CLASS)
    score = _get_positive_score(pipeline, X, positive_class)

    out = pd.DataFrame(index=df.index)
    if score is None:
        prediction = pipeline.predict(X)
        out["model_b_score_arrhythmia_or_abnormal"] = np.nan
        out["model_b_prediction"] = prediction
        out["model_b_prediction_default_threshold"] = prediction
        out["model_b_threshold_used"] = np.nan
        out["model_b_threshold_method"] = "default_predict"
        return out

    stored_threshold = threshold_info.get("threshold")
    if stored_threshold is None:
        prediction = pipeline.predict(X)
        threshold_used = np.nan
    else:
        threshold_used = float(stored_threshold)
        prediction = np.where(score >= threshold_used, positive_class, negative_class)

    default_threshold = _default_threshold_for_scores(score)
    prediction_default = np.where(score >= default_threshold, positive_class, negative_class)
    out["model_b_score_arrhythmia_or_abnormal"] = score
    out["model_b_prediction"] = prediction
    out["model_b_prediction_default_threshold"] = prediction_default
    out["model_b_threshold_used"] = threshold_used
    out["model_b_threshold_method"] = threshold_info.get("method", "default")
    return out


def _load_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .parquet or .csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = _load_input_table(args.input)
    predictions = predict_model_b_dataframe(df, models_dir=args.models_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote Model B predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
