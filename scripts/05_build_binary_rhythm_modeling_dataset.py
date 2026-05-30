"""Construye el dataset binario para modelado.

Parte de `data/processed/filtered_tabular_modeling_dataset.parquet`, aplica
el mapeo binario y añade nuevas features RR rolling por caso (sin usar
señal ECG cruda).

Salida (ignorada por `.gitignore`):
    data/processed/binary_rhythm_modeling_dataset.parquet

Columnas conservadas:
    * `case_id`, `time_second` — para trazabilidad y split.
    * `rhythm_label`, `rhythm_binary` — etiquetas (no entran como features).
    * metadata + features temporales heredadas del dataset tabular.
    * nuevas features RR rolling por caso para ventanas {5, 10, 20} latidos:
        - `rr_prev_rolling_mean_W`, `rr_prev_rolling_std_W`
        - `rr_rmssd_W`, `rr_pnn50_W`
        - `local_hr_mean_W`

Restricciones:
    * `beat_type` y otras columnas en `BINARY_LEAKAGE_COLUMNS` NO se usan
      al construir features rolling (solo se usan `time_second` y `rr_prev`).
    * Todas las features rolling se calculan dentro del `case_id` (ordenado
      por `time_second`).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402


ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20)
RMSSD_PNN50_THRESHOLD_SEC: float = 0.050  # umbral clásico pNN50: 50 ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_source_parquet(logger) -> Path:
    src = config.PROCESSED_DIR / config.TABULAR_DATASET_FILENAME
    if src.exists():
        return src
    logger.warning("No existe %s. Ejecutando scripts/02_build_filtered_tabular_modeling_dataset.py...", src)
    subprocess.check_call(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "02_build_filtered_tabular_modeling_dataset.py")]
    )
    if not src.exists():
        raise RuntimeError(f"El builder no produjo {src}.")
    return src


def _add_rolling_rr_features(df: pd.DataFrame,
                             windows: tuple[int, ...] = ROLLING_WINDOWS,
                             threshold_sec: float = RMSSD_PNN50_THRESHOLD_SEC) -> pd.DataFrame:
    """Añade features RR rolling por `case_id`.

    Para cada `case_id` (ordenado por `time_second`) calcula, sobre los
    últimos W latidos:
        * media y desviación estándar de `rr_prev`
        * RMSSD (raíz cuadrática media de las diferencias sucesivas de RR)
        * pNN50 (% de diferencias > 50 ms)
        * frecuencia cardíaca media local (bpm) = 60 / mean(rr_prev_W)

    El cálculo se hace por grupo. NO se usa información de otros casos.
    NO se usa `beat_type` ni `rhythm_label`.
    """
    df = df.sort_values([config.CASE_ID_COLUMN, config.BEAT_TIME_COLUMN]).reset_index(drop=True)
    rr_prev = df["rr_prev"]
    g = df.groupby(config.CASE_ID_COLUMN, sort=False)["rr_prev"]

    for w in windows:
        # Mean/std de rr_prev en ventana móvil.
        df[f"rr_prev_rolling_mean_{w}"] = g.rolling(window=w, min_periods=2).mean().reset_index(level=0, drop=True)
        df[f"rr_prev_rolling_std_{w}"] = g.rolling(window=w, min_periods=2).std().reset_index(level=0, drop=True)

        # Local HR (bpm) a partir de la media rolling.
        with np.errstate(divide="ignore", invalid="ignore"):
            local_hr = 60.0 / df[f"rr_prev_rolling_mean_{w}"]
        df[f"local_hr_mean_{w}"] = local_hr.replace([np.inf, -np.inf], np.nan)

    # RMSSD y pNN50 sobre las diferencias sucesivas de `rr_prev`.
    # Se computan por grupo con apply (más claro que reescribir rolling).
    def _rmssd_pnn50_for_case(group: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
        rr = group["rr_prev"].to_numpy()
        diffs = np.diff(rr)
        n = len(rr)
        rmssd = np.full(n, np.nan)
        pnn50 = np.full(n, np.nan)
        # diffs tiene longitud n-1: diffs[i] = rr[i+1] - rr[i].
        # La ventana de los últimos `window` valores de rr_prev usa los
        # últimos `window - 1` diffs.
        for i in range(n):
            start_d = max(0, i - (window - 1))
            end_d = i  # exclusivo en diffs[start_d:end_d]
            if end_d - start_d >= 1:
                d = diffs[start_d:end_d]
                d = d[~np.isnan(d)]
                if d.size >= 1:
                    rmssd[i] = float(np.sqrt(np.mean(d ** 2)))
                    pnn50[i] = float(np.mean(np.abs(d) > RMSSD_PNN50_THRESHOLD_SEC))
        return rmssd, pnn50

    for w in windows:
        rmssd_col = np.empty(len(df))
        pnn50_col = np.empty(len(df))
        for cid, group in df.groupby(config.CASE_ID_COLUMN, sort=False):
            rmssd, pnn50 = _rmssd_pnn50_for_case(group, w)
            rmssd_col[group.index] = rmssd
            pnn50_col[group.index] = pnn50
        df[f"rr_rmssd_{w}"] = rmssd_col
        df[f"rr_pnn50_{w}"] = pnn50_col

    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_DIR / config.BINARY_DATASET_FILENAME,
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limitar a los primeros N casos (debug).",
    )
    p.add_argument(
        "--rolling-windows",
        type=str,
        default=None,
        help="CSV de ventanas rolling, ej. '5,10,20'. Default: 5,10,20.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger = get_logger("build_binary")
    ensure_dir(config.PROCESSED_DIR)

    src = _ensure_source_parquet(logger)
    df = pd.read_parquet(src)
    logger.info("Fuente: %s | shape=%s", src.name, df.shape)

    if args.max_cases is not None:
        keep_ids = (
            df[config.CASE_ID_COLUMN].drop_duplicates().head(args.max_cases).tolist()
        )
        df = df.loc[df[config.CASE_ID_COLUMN].isin(keep_ids)].copy()
        logger.info("Tras --max-cases=%d: shape=%s", args.max_cases, df.shape)

    # Mapeo binario y exclusión de filas no contempladas
    df[config.BINARY_TARGET_COLUMN] = df[config.TARGET_COLUMN].map(config.map_rhythm_label_to_binary)
    n_before = len(df)
    df = df.dropna(subset=[config.BINARY_TARGET_COLUMN]).reset_index(drop=True)
    n_after = len(df)
    logger.info("Filas tras mapeo binario: %d -> %d", n_before, n_after)

    # Features rolling RR por caso
    windows = (
        tuple(int(x) for x in args.rolling_windows.split(",") if x.strip())
        if args.rolling_windows else ROLLING_WINDOWS
    )
    logger.info("Calculando rolling RR (windows=%s)...", windows)
    df = _add_rolling_rr_features(df, windows=windows)

    # Descartar columnas constantes que pueden haber aparecido al filtrar
    constant_cols = []
    for col in df.columns:
        if col in {config.CASE_ID_COLUMN, config.TARGET_COLUMN,
                   config.BEAT_TIME_COLUMN, config.BINARY_TARGET_COLUMN}:
            continue
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info("Columnas constantes descartadas: %s", constant_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Guardado %s | shape=%s", args.output, df.shape)
    logger.info("Distribución binaria:\n%s",
                df[config.BINARY_TARGET_COLUMN].value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
