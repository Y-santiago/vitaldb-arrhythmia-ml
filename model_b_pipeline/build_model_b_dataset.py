"""Build the reduced independent Model B dataset.

Usage:
    python model_b_pipeline/build_model_b_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_b_pipeline import config_model_b as cfg  # noqa: E402
from model_b_pipeline.utils_model_b import (  # noqa: E402
    ensure_dir,
    get_numeric_and_categorical_features,
    load_base_binary_dataset,
    save_csv,
)


def _feature_list(df: pd.DataFrame) -> pd.DataFrame:
    numeric, categorical = get_numeric_and_categorical_features(df, cfg.FEATURES_MODEL_B)
    kinds = {feature: "numeric" for feature in numeric}
    kinds.update({feature: "categorical" for feature in categorical})
    return pd.DataFrame([
        {
            "order": i,
            "feature": feature,
            "kind": kinds[feature],
            "dtype": str(df[feature].dtype),
            "is_forbidden": feature in cfg.FORBIDDEN_COLUMNS_MODEL_B,
            "is_additional_excluded": feature in cfg.ADDITIONAL_EXCLUDED_COLUMNS_MODEL_B,
        }
        for i, feature in enumerate(cfg.FEATURES_MODEL_B, start=1)
    ])


def _schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if column == cfg.CASE_ID_COLUMN:
            role = "case_id"
        elif column == cfg.TARGET_COLUMN:
            role = "target"
        else:
            role = "feature"
        rows.append({
            "column": column,
            "role": role,
            "dtype": str(df[column].dtype),
            "n_missing": int(df[column].isna().sum()),
            "missing_pct": float(df[column].isna().mean() * 100),
            "n_unique": int(df[column].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def build_model_b_dataset() -> pd.DataFrame:
    ensure_dir(cfg.PROCESSED_DIR)
    ensure_dir(cfg.TABLES_DIR)
    df = load_base_binary_dataset()
    cfg.validate_model_b_feature_set(df)

    allowed_columns = [cfg.CASE_ID_COLUMN, cfg.TARGET_COLUMN] + cfg.FEATURES_MODEL_B
    reduced = df.loc[:, allowed_columns].copy()
    reduced.to_parquet(cfg.MODEL_B_DATASET_PATH, index=False)

    save_csv(_schema(reduced), cfg.TABLES_DIR / "model_b_dataset_schema.csv")
    save_csv(_feature_list(df), cfg.TABLES_DIR / "model_b_feature_list.csv")
    return reduced


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    reduced = build_model_b_dataset()
    print(
        "Model B dataset written to "
        f"{cfg.MODEL_B_DATASET_PATH} with shape={reduced.shape}."
    )
    print("Columns:", ", ".join(reduced.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
