"""Auditoría del dataset tabular filtrado.

Carga todas las anotaciones disponibles, las une con `metadata.csv` por
`case_id`, aplica los filtros del proyecto (excluye `Noise`,
`bad_signal_quality` y etiquetas nulas) y genera 4 CSVs descriptivos en
`reports/tables/`.

No produce datos modelables: ese trabajo está en
`scripts/02_build_filtered_tabular_modeling_dataset.py`.

Outputs (no se versionan; ver `.gitignore`):
    * reports/tables/tabular_dataset_audit.csv         (resumen global)
    * reports/tables/tabular_class_distribution.csv    (conteo por rhythm_label)
    * reports/tables/tabular_cases_per_class.csv       (cases_per_class)
    * reports/tables/tabular_missing_values.csv        (faltantes por columna)
    * reports/tables/tabular_columns_classification.csv (numérica / categórica / leakage)

Uso:
    python scripts/01_audit_filtered_tabular_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loading import load_all_annotations, load_metadata, merge_metadata_and_annotations  # noqa: E402
from src.preprocessing import apply_basic_filters  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402


def _classify_columns(df: pd.DataFrame,
                      leakage: tuple[str, ...],
                      max_card: int) -> pd.DataFrame:
    """Clasifica cada columna como numeric / categorical / leakage / high_cardinality."""
    rows = []
    leakage_set = set(leakage)
    for col in df.columns:
        if col in leakage_set:
            kind = "leakage"
            cardinality = int(df[col].nunique(dropna=True))
        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            kind = "numeric"
            cardinality = int(df[col].nunique(dropna=True))
        elif pd.api.types.is_bool_dtype(df[col]):
            kind = "categorical"
            cardinality = int(df[col].nunique(dropna=True))
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            cardinality = int(df[col].nunique(dropna=True))
            kind = "categorical" if cardinality <= max_card else "high_cardinality"
        else:
            kind = "other"
            cardinality = int(df[col].nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "kind": kind,
                "n_unique": cardinality,
                "n_missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--max-categorical-cardinality",
        type=int,
        default=config.TABULAR_MAX_CATEGORY_CARDINALITY,
        help=f"Cardinalidad máxima para categóricas (default: {config.TABULAR_MAX_CATEGORY_CARDINALITY}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=config.TABLES_DIR,
        help="Carpeta destino (default: reports/tables/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger = get_logger("audit_tabular")
    ensure_dir(args.output_dir)

    # ----------------------------------------------------------------------
    # 1. Carga
    # ----------------------------------------------------------------------
    annotations = load_all_annotations()
    metadata = load_metadata()
    logger.info("Anotaciones: shape=%s", annotations.shape)
    logger.info("Metadata:    shape=%s", metadata.shape)

    n_rows_raw = len(annotations)
    n_cases_raw = int(annotations[config.CASE_ID_COLUMN].nunique())

    merged = merge_metadata_and_annotations(metadata, annotations, on=config.CASE_ID_COLUMN, how="inner")
    logger.info("Merge: shape=%s", merged.shape)

    # ----------------------------------------------------------------------
    # 2. Filtros del proyecto
    # ----------------------------------------------------------------------
    filtered = apply_basic_filters(merged)
    # apply_basic_filters NO elimina NaN ni cadenas "nan" del target.
    filtered = filtered.dropna(subset=[config.TARGET_COLUMN])
    mask_string_nan = (
        filtered[config.TARGET_COLUMN]
        .astype(str).str.strip().str.lower()
        .isin({"nan", "none", ""})
    )
    filtered = filtered.loc[~mask_string_nan].copy()

    n_rows_filtered = len(filtered)
    n_cases_filtered = int(filtered[config.CASE_ID_COLUMN].nunique())

    logger.info(
        "Filtros: %d→%d filas, %d→%d cases",
        n_rows_raw, n_rows_filtered, n_cases_raw, n_cases_filtered,
    )

    # ----------------------------------------------------------------------
    # 3. Tabla resumen global
    # ----------------------------------------------------------------------
    cols_class = _classify_columns(filtered, config.TABULAR_LEAKAGE_COLUMNS, args.max_categorical_cardinality)
    n_numeric = int((cols_class["kind"] == "numeric").sum())
    n_categorical = int((cols_class["kind"] == "categorical").sum())
    n_leakage = int((cols_class["kind"] == "leakage").sum())
    n_high_card = int((cols_class["kind"] == "high_cardinality").sum())

    n_duplicates = int(filtered.duplicated().sum())

    summary = pd.DataFrame(
        [
            {"metric": "rows_before_filters", "value": n_rows_raw},
            {"metric": "rows_after_filters",  "value": n_rows_filtered},
            {"metric": "cases_before_filters", "value": n_cases_raw},
            {"metric": "cases_after_filters",  "value": n_cases_filtered},
            {"metric": "n_classes_rhythm_label", "value": int(filtered[config.TARGET_COLUMN].nunique())},
            {"metric": "n_columns_total", "value": int(filtered.shape[1])},
            {"metric": "n_columns_numeric_candidate", "value": n_numeric},
            {"metric": "n_columns_categorical_candidate", "value": n_categorical},
            {"metric": "n_columns_leakage_excluded", "value": n_leakage},
            {"metric": "n_columns_high_cardinality_excluded", "value": n_high_card},
            {"metric": "n_duplicate_rows", "value": n_duplicates},
        ]
    )
    summary_path = args.output_dir / "tabular_dataset_audit.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Guardado %s", summary_path)

    # ----------------------------------------------------------------------
    # 4. Distribución de clases
    # ----------------------------------------------------------------------
    dist = (
        filtered[config.TARGET_COLUMN]
        .value_counts(dropna=False)
        .rename_axis(config.TARGET_COLUMN)
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    dist["pct_rows"] = (dist["n_rows"] / dist["n_rows"].sum() * 100).round(3)
    dist_path = args.output_dir / "tabular_class_distribution.csv"
    dist.to_csv(dist_path, index=False)
    logger.info("Guardado %s", dist_path)

    # ----------------------------------------------------------------------
    # 5. Casos únicos por clase
    # ----------------------------------------------------------------------
    cases_per_class = (
        filtered.groupby(config.TARGET_COLUMN)[config.CASE_ID_COLUMN]
        .nunique()
        .rename("n_cases_with_class")
        .reset_index()
        .sort_values("n_cases_with_class", ascending=False)
    )
    cases_per_class["pct_cases"] = (
        cases_per_class["n_cases_with_class"] / n_cases_filtered * 100
    ).round(3)
    cpc_path = args.output_dir / "tabular_cases_per_class.csv"
    cases_per_class.to_csv(cpc_path, index=False)
    logger.info("Guardado %s", cpc_path)

    # ----------------------------------------------------------------------
    # 6. Faltantes por columna
    # ----------------------------------------------------------------------
    missing = cols_class[["column", "dtype", "kind", "n_missing", "missing_pct"]].copy()
    missing = missing.sort_values(["n_missing", "column"], ascending=[False, True])
    miss_path = args.output_dir / "tabular_missing_values.csv"
    missing.to_csv(miss_path, index=False)
    logger.info("Guardado %s", miss_path)

    # ----------------------------------------------------------------------
    # 7. Clasificación de columnas (extra, útil para el dataset builder)
    # ----------------------------------------------------------------------
    cls_path = args.output_dir / "tabular_columns_classification.csv"
    cols_class.to_csv(cls_path, index=False)
    logger.info("Guardado %s", cls_path)

    # ----------------------------------------------------------------------
    # 8. Log final
    # ----------------------------------------------------------------------
    logger.info("Top 5 rhythm_label por filas:")
    for _, r in dist.head(5).iterrows():
        logger.info("  %-30s %d (%.2f%%)", r[config.TARGET_COLUMN], int(r["n_rows"]), r["pct_rows"])
    logger.info("Top 5 rhythm_label por # de cases:")
    for _, r in cases_per_class.head(5).iterrows():
        logger.info("  %-30s %d (%.2f%%)", r[config.TARGET_COLUMN], int(r["n_cases_with_class"]), r["pct_cases"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
