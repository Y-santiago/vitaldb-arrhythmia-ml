"""Premodel audit for the independent Model B dataset.

Usage:
    python model_b_pipeline/audit_model_b_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

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


def _class_distribution_rows(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[cfg.TARGET_COLUMN].value_counts(dropna=False)
    out = counts.rename_axis("rhythm_binary").reset_index(name="n_rows")
    out["pct_rows"] = out["n_rows"] / len(df) * 100
    return out


def _class_distribution_cases(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_cases = df[cfg.CASE_ID_COLUMN].nunique()
    for label in cfg.CLASS_LABELS:
        n_cases = df.loc[df[cfg.TARGET_COLUMN] == label, cfg.CASE_ID_COLUMN].nunique()
        rows.append({
            "rhythm_binary": label,
            "n_cases_with_at_least_one_row": int(n_cases),
            "pct_cases": float(n_cases / total_cases * 100) if total_cases else 0.0,
        })
    return pd.DataFrame(rows)


def _case_composition(df: pd.DataFrame) -> pd.DataFrame:
    by_case = (
        df.groupby(cfg.CASE_ID_COLUMN)[cfg.TARGET_COLUMN]
        .agg(lambda s: sorted(set(s.dropna())))
        .reset_index(name="classes_present")
    )

    def classify(classes: list[str]) -> str:
        has_neg = cfg.NEGATIVE_CLASS in classes
        has_pos = cfg.POSITIVE_CLASS in classes
        if has_neg and has_pos:
            return "mixed_normal_and_abnormal"
        if has_neg:
            return "only_normal_sinus"
        if has_pos:
            return "only_arrhythmia_or_abnormal"
        return "other_or_missing_target"

    by_case["composition"] = by_case["classes_present"].apply(classify)
    summary = by_case["composition"].value_counts().rename_axis("composition").reset_index(name="n_cases")
    summary["pct_cases"] = summary["n_cases"] / len(by_case) * 100 if len(by_case) else 0.0
    return summary


def _feature_list(df: pd.DataFrame) -> pd.DataFrame:
    numeric, categorical = get_numeric_and_categorical_features(df, cfg.FEATURES_MODEL_B)
    kinds = {feature: "numeric" for feature in numeric}
    kinds.update({feature: "categorical" for feature in categorical})
    rows = []
    for i, feature in enumerate(cfg.FEATURES_MODEL_B, start=1):
        rows.append({
            "order": i,
            "feature": feature,
            "kind": kinds[feature],
            "dtype": str(df[feature].dtype),
            "n_missing": int(df[feature].isna().sum()),
            "missing_pct": float(df[feature].isna().mean() * 100),
            "n_unique": int(df[feature].nunique(dropna=True)),
            "is_forbidden": feature in cfg.FORBIDDEN_COLUMNS_MODEL_B,
            "is_additional_excluded": feature in cfg.ADDITIONAL_EXCLUDED_COLUMNS_MODEL_B,
        })
    return pd.DataFrame(rows)


def _missing_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in cfg.FEATURES_MODEL_B:
        rows.append({
            "feature": feature,
            "dtype": str(df[feature].dtype),
            "n_missing": int(df[feature].isna().sum()),
            "missing_pct": float(df[feature].isna().mean() * 100),
            "n_unique": int(df[feature].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)


def _numeric_descriptives_by_class(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in numeric_features:
        for label in cfg.CLASS_LABELS:
            s = df.loc[df[cfg.TARGET_COLUMN] == label, feature]
            has_values = s.notna().any()
            rows.append({
                "feature": feature,
                "rhythm_binary": label,
                "n_non_missing": int(s.notna().sum()),
                "mean": float(s.mean()) if has_values else float("nan"),
                "std": float(s.std()) if has_values else float("nan"),
                "median": float(s.median()) if has_values else float("nan"),
                "p25": float(s.quantile(0.25)) if has_values else float("nan"),
                "p75": float(s.quantile(0.75)) if has_values else float("nan"),
                "min": float(s.min()) if has_values else float("nan"),
                "max": float(s.max()) if has_values else float("nan"),
            })
    return pd.DataFrame(rows)


def _categorical_descriptives(df: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in categorical_features:
        for label in cfg.CLASS_LABELS:
            s = df.loc[df[cfg.TARGET_COLUMN] == label, feature]
            counts = s.value_counts(dropna=False).head(10)
            rows.append({
                "feature": feature,
                "rhythm_binary": label,
                "n_non_missing": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
                "n_unique_non_missing": int(s.nunique(dropna=True)),
                "top_values": "; ".join(f"{value}:{count}" for value, count in counts.items()),
            })
    return pd.DataFrame(rows)


def _write_figures(
    rows_dist: pd.DataFrame,
    missing_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    ensure_dir(figures_dir)
    sns.set_theme(context="notebook", style="whitegrid")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.barplot(
        data=rows_dist,
        x="rhythm_binary",
        y="n_rows",
        hue="rhythm_binary",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Model B class distribution by rows")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    for i, value in enumerate(rows_dist["n_rows"]):
        ax.text(i, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_b_class_distribution_rows.png", dpi=140)
    plt.close(fig)

    top_missing = missing_df.sort_values("missing_pct", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.35 * len(top_missing) + 1.2)))
    sns.barplot(
        data=top_missing,
        x="missing_pct",
        y="feature",
        hue="feature",
        palette="rocket_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("Model B missing values by feature")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_b_missing_values_top.png", dpi=140)
    plt.close(fig)


def run_audit() -> dict[str, Path]:
    ensure_dir(cfg.TABLES_DIR)
    ensure_dir(cfg.FIGURES_DIR)
    df = load_base_binary_dataset()
    cfg.validate_model_b_feature_set(df)

    feature_df = _feature_list(df)
    missing_df = _missing_values(df)
    rows_dist = _class_distribution_rows(df)
    cases_dist = _class_distribution_cases(df)
    case_composition = _case_composition(df)
    numeric, categorical = get_numeric_and_categorical_features(df, cfg.FEATURES_MODEL_B)

    audit_df = pd.DataFrame([
        {"metric": "source_dataset", "value": str(cfg.BASE_BINARY_DATASET_PATH.relative_to(cfg.PROJECT_ROOT))},
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_case_id", "value": int(df[cfg.CASE_ID_COLUMN].nunique())},
        {"metric": "n_features_model_b", "value": int(len(cfg.FEATURES_MODEL_B))},
        {"metric": "n_numeric_features", "value": int(len(numeric))},
        {"metric": "n_categorical_features", "value": int(len(categorical))},
        {"metric": "target_column", "value": cfg.TARGET_COLUMN},
        {"metric": "positive_class", "value": cfg.POSITIVE_CLASS},
        {"metric": "negative_class", "value": cfg.NEGATIVE_CLASS},
        {"metric": "all_features_present", "value": True},
        {"metric": "forbidden_columns_in_features", "value": ",".join(sorted(set(cfg.FEATURES_MODEL_B) & cfg.FORBIDDEN_COLUMNS_MODEL_B)) or "none"},
        {"metric": "additional_excluded_columns_in_features", "value": ",".join(sorted(set(cfg.FEATURES_MODEL_B) & cfg.ADDITIONAL_EXCLUDED_COLUMNS_MODEL_B)) or "none"},
    ])

    paths = {
        "audit": save_csv(audit_df, cfg.TABLES_DIR / "model_b_dataset_audit.csv"),
        "feature_list": save_csv(feature_df, cfg.TABLES_DIR / "model_b_feature_list.csv"),
        "missing": save_csv(missing_df, cfg.TABLES_DIR / "model_b_missing_values.csv"),
        "class_rows": save_csv(rows_dist, cfg.TABLES_DIR / "model_b_class_distribution_rows.csv"),
        "class_cases": save_csv(cases_dist, cfg.TABLES_DIR / "model_b_class_distribution_cases.csv"),
        "case_composition": save_csv(case_composition, cfg.TABLES_DIR / "model_b_case_composition.csv"),
        "numeric_descriptives": save_csv(
            _numeric_descriptives_by_class(df, numeric),
            cfg.TABLES_DIR / "model_b_numeric_descriptives_by_class.csv",
        ),
        "categorical_descriptives": save_csv(
            _categorical_descriptives(df, categorical),
            cfg.TABLES_DIR / "model_b_categorical_descriptives.csv",
        ),
    }
    _write_figures(rows_dist, missing_df, cfg.FIGURES_DIR)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    paths = run_audit()
    print("Model B audit completed.")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
