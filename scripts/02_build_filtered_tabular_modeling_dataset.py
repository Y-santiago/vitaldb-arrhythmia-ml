"""Construye el dataset tabular filtrado para modelado.

Une anotaciones (PhysioNet) con `metadata.csv` por `case_id`, aplica los
filtros del proyecto y deriva features temporales por latido dentro de
cada caso. NO usa ECG crudo, NO descarga señales desde VitalDB, NO
ventana señal.

Lo que entra al parquet:
    * Identificadores conservados para trazabilidad: `case_id`,
      `rhythm_label`, `time_second`.
    * Metadatos del paciente / cirugía / anestesia presentes en
      `metadata.csv`, excepto las columnas listadas en
      `config.TABULAR_LEAKAGE_COLUMNS`.
    * Features temporales derivadas dentro del caso:
        - `rr_prev`, `rr_next` (segundos hasta el latido previo/siguiente)
        - `hr_inst_from_rr_prev` (frecuencia instantánea en bpm)
        - `position_in_case` (posición relativa de la fila dentro del caso,
          entre 0 y 1)
    * `beat_type` se conserva en el parquet **únicamente para análisis
      descriptivo**. Está en la lista de leakage de modelado y nunca debe
      entrar como predictor.

Salida (ignorada por `.gitignore`):
    data/processed/filtered_tabular_modeling_dataset.parquet

Uso:
    python scripts/02_build_filtered_tabular_modeling_dataset.py
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
from src.data_loading import (  # noqa: E402
    load_all_annotations,
    load_metadata,
    merge_metadata_and_annotations,
)
from src.preprocessing import apply_basic_filters  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402


def _drop_label_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con `rhythm_label` nulo o cadena 'nan' / 'none' / vacía."""
    out = df.dropna(subset=[config.TARGET_COLUMN]).copy()
    mask = (
        out[config.TARGET_COLUMN]
        .astype(str).str.strip().str.lower()
        .isin({"nan", "none", ""})
    )
    return out.loc[~mask].copy()


def _add_within_case_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features temporales por latido dentro de cada caso.

    Importante: se ordena por `case_id, time_second` antes de calcular
    diferencias. Cada caso se trata por separado para que un latido del
    caso A no use información del caso B.
    """
    out = df.sort_values([config.CASE_ID_COLUMN, config.BEAT_TIME_COLUMN]).reset_index(drop=True)
    g = out.groupby(config.CASE_ID_COLUMN, sort=False)

    # RR locales (en segundos).
    out["rr_prev"] = g[config.BEAT_TIME_COLUMN].diff()
    out["rr_next"] = g[config.BEAT_TIME_COLUMN].diff(periods=-1).abs()

    # Frecuencia instantánea (bpm) calculada a partir del RR previo.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["hr_inst_from_rr_prev"] = 60.0 / out["rr_prev"]
    out.loc[out["rr_prev"] <= 0, "hr_inst_from_rr_prev"] = np.nan

    # Posición relativa del latido dentro de la duración del caso (0..1).
    case_min = g[config.BEAT_TIME_COLUMN].transform("min")
    case_max = g[config.BEAT_TIME_COLUMN].transform("max")
    span = case_max - case_min
    pos = (out[config.BEAT_TIME_COLUMN] - case_min) / span.where(span > 0, np.nan)
    out["position_in_case"] = pos.fillna(0.0)

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limitar a los primeros N casos (útil para debugging).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_DIR / config.TABULAR_DATASET_FILENAME,
        help="Parquet de salida (default: data/processed/filtered_tabular_modeling_dataset.parquet).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger = get_logger("build_tabular")
    ensure_dir(config.PROCESSED_DIR)

    # ----------------------------------------------------------------------
    # 1. Carga + merge
    # ----------------------------------------------------------------------
    annotations = load_all_annotations()
    metadata = load_metadata()

    if args.max_cases is not None:
        keep_ids = (
            annotations[config.CASE_ID_COLUMN]
            .drop_duplicates()
            .head(args.max_cases)
            .tolist()
        )
        annotations = annotations.loc[annotations[config.CASE_ID_COLUMN].isin(keep_ids)].copy()
        logger.info("--max-cases=%d → %d filas de anotación", args.max_cases, len(annotations))

    merged = merge_metadata_and_annotations(
        metadata, annotations, on=config.CASE_ID_COLUMN, how="inner"
    )
    logger.info("Merge: shape=%s", merged.shape)

    # ----------------------------------------------------------------------
    # 2. Filtros (Noise, bad_signal_quality, rhythm_label inválido)
    # ----------------------------------------------------------------------
    filtered = apply_basic_filters(merged)
    filtered = _drop_label_nans(filtered)
    logger.info("Post-filtros: shape=%s", filtered.shape)

    # ----------------------------------------------------------------------
    # 3. Coerciones de tipos en metadatos
    # ----------------------------------------------------------------------
    # `age` viene como string porque algunas filas usan ">89" para anonimizar
    # a pacientes ancianos. Lo coercemos a numérico tratando ">89" como 89.0.
    if "age" in filtered.columns and not pd.api.types.is_numeric_dtype(filtered["age"]):
        n_gt89 = int((filtered["age"].astype(str).str.strip() == ">89").sum())
        filtered["age"] = (
            filtered["age"]
            .astype(str).str.strip().str.replace(">89", "89", regex=False)
        )
        filtered["age"] = pd.to_numeric(filtered["age"], errors="coerce")
        logger.info("Coerced `age` a numérico (>89 → 89.0 en %d filas).", n_gt89)

    # ----------------------------------------------------------------------
    # 4. Features temporales por caso
    # ----------------------------------------------------------------------
    enriched = _add_within_case_time_features(filtered)

    # ----------------------------------------------------------------------
    # 4. Descartar columnas que nunca aportan (todo NaN, todo el mismo
    #    valor). Estas habrían quedado fuera del modelo de todos modos,
    #    pero las quitamos del parquet para reducir tamaño.
    # ----------------------------------------------------------------------
    cols_to_drop_constant = []
    for col in enriched.columns:
        if col in (config.CASE_ID_COLUMN, config.TARGET_COLUMN, config.BEAT_TIME_COLUMN):
            continue
        n_unique = enriched[col].nunique(dropna=True)
        if n_unique <= 1:
            cols_to_drop_constant.append(col)
    if cols_to_drop_constant:
        logger.info(
            "Columnas constantes / vacías descartadas (%d): %s",
            len(cols_to_drop_constant), cols_to_drop_constant,
        )
        enriched = enriched.drop(columns=cols_to_drop_constant)

    # ----------------------------------------------------------------------
    # 5. Persistencia
    # ----------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    logger.info("Guardado %s | shape=%s", args.output, enriched.shape)
    logger.info(
        "Cases en el parquet: %d | clases únicas: %d",
        int(enriched[config.CASE_ID_COLUMN].nunique()),
        int(enriched[config.TARGET_COLUMN].nunique()),
    )
    logger.info(
        "Distribución de rhythm_label (top 10):\n%s",
        enriched[config.TARGET_COLUMN].value_counts(dropna=False).head(10).to_string(),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
