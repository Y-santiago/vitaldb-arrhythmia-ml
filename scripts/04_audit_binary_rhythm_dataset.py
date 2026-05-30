"""Auditoría del dataset binario `normal_sinus` vs `arrhythmia_or_abnormal`.

Carga `data/processed/filtered_tabular_modeling_dataset.parquet` (o lo
regenera invocando `scripts/02_build_filtered_tabular_modeling_dataset.py`
si está ausente), aplica el mapeo binario declarado en
`src/config.BINARY_LABEL_MAPPING` + `BINARY_EXCLUDED_LABELS`, y genera 9
CSVs + 3 figuras descriptivas en `reports/tables/` y `reports/figures/`.

Genera, entre otros:
    * `binary_dataset_audit.csv` — resumen global.
    * `binary_label_mapping.csv` — mapeo `rhythm_label -> rhythm_binary`.
    * `binary_class_distribution.csv` — filas por clase binaria.
    * `binary_cases_per_class.csv` — casos por clase binaria.
    * `binary_case_composition.csv` — casos solo normales / solo anormales /
      mixtos.
    * `binary_missing_values.csv`.
    * `binary_feature_characterization_numeric.csv` — descriptivos por clase.
    * `binary_feature_characterization_categorical.csv` — cardinalidad por
      clase.
    * `binary_excluded_rows_summary.csv`.
    * `binary_class_distribution_rows.png`.
    * `binary_class_distribution_cases.png`.
    * `binary_missing_values_top.png`.

NO usa ECG crudo. NO descarga señales.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_source_parquet(logger) -> Path:
    """Si no existe el parquet tabular, lanza el script 02 para generarlo."""
    src = config.PROCESSED_DIR / config.TABULAR_DATASET_FILENAME
    if src.exists():
        return src
    logger.warning("No existe %s. Ejecutando scripts/02_build_filtered_tabular_modeling_dataset.py...", src)
    builder = PROJECT_ROOT / "scripts" / "02_build_filtered_tabular_modeling_dataset.py"
    if not builder.exists():
        raise FileNotFoundError(f"No encuentro {builder}.")
    subprocess.check_call([sys.executable, str(builder)])
    if not src.exists():
        raise RuntimeError(f"El builder no produjo {src}.")
    return src


def _apply_binary_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica el mapeo binario y devuelve (df_mapeado, conteos_excluidos)."""
    mapper = config.map_rhythm_label_to_binary
    df = df.copy()
    df[config.BINARY_TARGET_COLUMN] = df[config.TARGET_COLUMN].map(mapper)

    # Razones por las que las filas quedan excluidas (rhythm_binary NaN).
    original = df[config.TARGET_COLUMN].astype(str).str.strip()
    excluded_mask = df[config.BINARY_TARGET_COLUMN].isna()

    reason_counts = {}
    if excluded_mask.any():
        ex_labels = original.loc[excluded_mask]
        for lbl, n in ex_labels.value_counts(dropna=False).items():
            reason = config.BINARY_EXCLUDED_LABELS.get(
                lbl,
                "no contemplada en BINARY_LABEL_MAPPING; revisar manualmente"
                if lbl and lbl.lower() not in {"nan", "none", "null", ""}
                else "etiqueta nula/inválida",
            )
            reason_counts.setdefault(reason, {"label": lbl, "n_rows": 0})
            reason_counts[reason]["label"] = lbl
            reason_counts[reason]["n_rows"] += int(n)

    return df, reason_counts


def _classify_columns(df: pd.DataFrame,
                      leakage: tuple[str, ...],
                      max_card: int,
                      max_missing_pct: float = 99.0) -> dict[str, list[str]]:
    """Separa columnas por tipo y motivo de exclusión."""
    leakage_set = set(leakage)
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
        missing_pct = float(df[col].isna().mean() * 100)
        if missing_pct > max_missing_pct:
            too_missing.append(col)
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            if n_unique <= max_card:
                categorical.append(col)
            else:
                high_card.append(col)
        else:
            high_card.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "leakage": leak,
        "high_cardinality": high_card,
        "constant": constant,
        "too_missing": too_missing,
    }


def _numeric_characterization(df_bin: pd.DataFrame,
                              numeric_cols: list[str]) -> pd.DataFrame:
    """Estadísticos descriptivos por clase binaria."""
    rows = []
    for col in numeric_cols:
        for cls in (config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS):
            sub = df_bin.loc[df_bin[config.BINARY_TARGET_COLUMN] == cls, col]
            rows.append({
                "feature": col,
                "class": cls,
                "n_non_null": int(sub.notna().sum()),
                "mean": float(sub.mean()) if sub.notna().any() else float("nan"),
                "std": float(sub.std()) if sub.notna().any() else float("nan"),
                "median": float(sub.median()) if sub.notna().any() else float("nan"),
                "p25": float(sub.quantile(0.25)) if sub.notna().any() else float("nan"),
                "p75": float(sub.quantile(0.75)) if sub.notna().any() else float("nan"),
                "min": float(sub.min()) if sub.notna().any() else float("nan"),
                "max": float(sub.max()) if sub.notna().any() else float("nan"),
            })
    return pd.DataFrame(rows)


def _categorical_characterization(df_bin: pd.DataFrame,
                                  categorical_cols: list[str]) -> pd.DataFrame:
    """Cardinalidad y top categorías por clase binaria."""
    rows = []
    for col in categorical_cols:
        for cls in (config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS):
            sub = df_bin.loc[df_bin[config.BINARY_TARGET_COLUMN] == cls, col]
            counts = sub.value_counts(dropna=True)
            top = counts.head(3)
            rows.append({
                "feature": col,
                "class": cls,
                "n_non_null": int(sub.notna().sum()),
                "n_unique": int(counts.shape[0]),
                "top_values": "; ".join(f"{k}:{v}" for k, v in top.items()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _plot_class_distribution_rows(df_bin: pd.DataFrame, out_path: Path) -> None:
    counts = df_bin[config.BINARY_TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                hue=counts.index, palette="Set2", legend=False)
    ax.set_title("Distribución binaria por filas")
    ax.set_xlabel("rhythm_binary")
    ax.set_ylabel("# filas (latidos)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_class_distribution_cases(df_bin: pd.DataFrame, out_path: Path) -> None:
    cases_pos = df_bin.loc[df_bin[config.BINARY_TARGET_COLUMN] == config.BINARY_POSITIVE_CLASS,
                            config.CASE_ID_COLUMN].nunique()
    cases_neg = df_bin.loc[df_bin[config.BINARY_TARGET_COLUMN] == config.BINARY_NEGATIVE_CLASS,
                            config.CASE_ID_COLUMN].nunique()
    s = pd.Series([cases_neg, cases_pos],
                  index=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=s.index, y=s.values, ax=ax,
                hue=s.index, palette="Set2", legend=False)
    ax.set_title("Distribución binaria por casos (al menos 1 latido)")
    ax.set_xlabel("rhythm_binary")
    ax.set_ylabel("# casos")
    for i, v in enumerate(s.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_missing_top(missing_df: pd.DataFrame, out_path: Path, top: int = 20) -> None:
    top_df = missing_df.sort_values("missing_pct", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_df) + 1))
    sns.barplot(x=top_df["missing_pct"], y=top_df["column"], ax=ax,
                hue=top_df["column"], palette="rocket_r", legend=False)
    ax.set_title(f"Top {top} columnas con más faltantes (%)")
    ax.set_xlabel("missing %")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--max-categorical-cardinality",
        type=int,
        default=config.TABULAR_MAX_CATEGORY_CARDINALITY,
    )
    p.add_argument(
        "--max-missing-pct",
        type=float,
        default=99.0,
        help="Columnas con >X%% de faltantes se marcan como excluidas (default: 99%%).",
    )
    p.add_argument("--output-dir", type=Path, default=config.TABLES_DIR)
    p.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger = get_logger("audit_binary")
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)
    sns.set_theme(context="notebook", style="whitegrid")

    # 1. Carga
    src_path = _ensure_source_parquet(logger)
    df = pd.read_parquet(src_path)
    logger.info("Fuente: %s | shape=%s", src_path.name, df.shape)

    n_rows_before = int(len(df))
    n_cases_before = int(df[config.CASE_ID_COLUMN].nunique())

    # 2. Mapeo binario
    df_mapped, reason_counts = _apply_binary_mapping(df)
    excluded_mask = df_mapped[config.BINARY_TARGET_COLUMN].isna()
    df_bin = df_mapped.loc[~excluded_mask].copy()

    n_rows_after = int(len(df_bin))
    n_cases_after = int(df_bin[config.CASE_ID_COLUMN].nunique())
    logger.info("Tras mapeo binario + filtros: %d -> %d filas, %d -> %d cases",
                n_rows_before, n_rows_after, n_cases_before, n_cases_after)

    # 3. Tabla de mapeo
    mapping_rows = [
        {"rhythm_label": k, "rhythm_binary": v, "decision": "include"}
        for k, v in config.BINARY_LABEL_MAPPING.items()
    ] + [
        {"rhythm_label": k, "rhythm_binary": None,
         "decision": f"exclude: {reason}"}
        for k, reason in config.BINARY_EXCLUDED_LABELS.items()
    ]
    pd.DataFrame(mapping_rows).to_csv(args.output_dir / "binary_label_mapping.csv", index=False)

    # 4. Distribución binaria global por filas
    dist = (
        df_bin[config.BINARY_TARGET_COLUMN].value_counts()
        .rename_axis(config.BINARY_TARGET_COLUMN)
        .reset_index(name="n_rows")
    )
    dist["pct_rows"] = (dist["n_rows"] / dist["n_rows"].sum() * 100).round(3)
    dist.to_csv(args.output_dir / "binary_class_distribution.csv", index=False)

    # 5. Casos por clase binaria
    cases_per_class = (
        df_bin.groupby(config.BINARY_TARGET_COLUMN)[config.CASE_ID_COLUMN]
        .nunique().rename("n_cases").reset_index()
    )
    cases_per_class["pct_cases"] = (cases_per_class["n_cases"] / n_cases_after * 100).round(3)
    cases_per_class.to_csv(args.output_dir / "binary_cases_per_class.csv", index=False)

    # 6. Composición por caso
    per_case = df_bin.groupby(config.CASE_ID_COLUMN)[config.BINARY_TARGET_COLUMN].agg(set)
    only_neg = sum(1 for s in per_case if s == {config.BINARY_NEGATIVE_CLASS})
    only_pos = sum(1 for s in per_case if s == {config.BINARY_POSITIVE_CLASS})
    mixed = sum(1 for s in per_case if len(s) > 1)
    case_comp = pd.DataFrame([
        {"category": "only_normal_sinus", "n_cases": only_neg,
         "pct_cases": round(only_neg / n_cases_after * 100, 3)},
        {"category": "only_arrhythmia_or_abnormal", "n_cases": only_pos,
         "pct_cases": round(only_pos / n_cases_after * 100, 3)},
        {"category": "mixed", "n_cases": mixed,
         "pct_cases": round(mixed / n_cases_after * 100, 3)},
    ])
    case_comp.to_csv(args.output_dir / "binary_case_composition.csv", index=False)

    # 7. Filas excluidas y motivos
    excluded_rows = []
    for reason, info in reason_counts.items():
        excluded_rows.append({
            "rhythm_label_original": info["label"],
            "n_rows_excluded": int(info["n_rows"]),
            "reason": reason,
        })
    pd.DataFrame(excluded_rows).to_csv(
        args.output_dir / "binary_excluded_rows_summary.csv", index=False
    )

    # 8. Clasificación de columnas (en df_bin, no en df_mapped, para evaluar
    # constantes/faltantes sobre los datos que van al modelo).
    cls = _classify_columns(
        df_bin,
        leakage=config.BINARY_LEAKAGE_COLUMNS,
        max_card=args.max_categorical_cardinality,
        max_missing_pct=args.max_missing_pct,
    )

    # 9. Caracterización numérica y categórica por clase
    num_char = _numeric_characterization(df_bin, cls["numeric"])
    cat_char = _categorical_characterization(df_bin, cls["categorical"])
    num_char.to_csv(args.output_dir / "binary_feature_characterization_numeric.csv", index=False)
    cat_char.to_csv(args.output_dir / "binary_feature_characterization_categorical.csv", index=False)

    # 10. Faltantes por columna
    missing_rows = []
    for col in df_bin.columns:
        missing_rows.append({
            "column": col,
            "dtype": str(df_bin[col].dtype),
            "n_missing": int(df_bin[col].isna().sum()),
            "missing_pct": round(float(df_bin[col].isna().mean() * 100), 3),
        })
    missing_df = pd.DataFrame(missing_rows).sort_values(
        ["missing_pct", "column"], ascending=[False, True]
    )
    missing_df.to_csv(args.output_dir / "binary_missing_values.csv", index=False)

    # 11. Resumen global
    summary = pd.DataFrame([
        {"metric": "rows_before_binary_filter", "value": n_rows_before},
        {"metric": "rows_after_binary_filter",  "value": n_rows_after},
        {"metric": "cases_before",              "value": n_cases_before},
        {"metric": "cases_after",               "value": n_cases_after},
        {"metric": "n_rhythm_label_original",   "value": int(df[config.TARGET_COLUMN].nunique())},
        {"metric": "n_columns_numeric",         "value": len(cls["numeric"])},
        {"metric": "n_columns_categorical",     "value": len(cls["categorical"])},
        {"metric": "n_columns_leakage",         "value": len(cls["leakage"])},
        {"metric": "n_columns_high_cardinality", "value": len(cls["high_cardinality"])},
        {"metric": "n_columns_constant",        "value": len(cls["constant"])},
        {"metric": "n_columns_too_missing",     "value": len(cls["too_missing"])},
        {"metric": "n_rows_only_normal_cases",      "value": only_neg},
        {"metric": "n_rows_only_abnormal_cases",    "value": only_pos},
        {"metric": "n_rows_mixed_cases",            "value": mixed},
    ])
    summary.to_csv(args.output_dir / "binary_dataset_audit.csv", index=False)

    # 12. Figuras
    _plot_class_distribution_rows(df_bin, args.figures_dir / "binary_class_distribution_rows.png")
    _plot_class_distribution_cases(df_bin, args.figures_dir / "binary_class_distribution_cases.png")
    _plot_missing_top(missing_df, args.figures_dir / "binary_missing_values_top.png")

    # 13. Logs
    logger.info("Distribución por filas:\n%s", dist.to_string(index=False))
    logger.info("Casos por clase binaria:\n%s", cases_per_class.to_string(index=False))
    logger.info("Composición por caso:\n%s", case_comp.to_string(index=False))
    logger.info("Outputs en %s y %s", args.output_dir, args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
