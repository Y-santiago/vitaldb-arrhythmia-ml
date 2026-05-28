"""[LEGACY — pipeline exploratorio ECG crudo, pausado en esta fase]

Para el flujo activo de modelado, ver
``scripts/02_build_filtered_tabular_modeling_dataset.py`` (tabular sin
ventaneo de señal). Este script se conserva como referencia histórica.

------------------------------------------------------------------------

Genera parquets de features por tamaño de ventana (1.2 / 2.0 / 5.0 s).

Para cada `case_<id>.npy` cacheado en `data/raw/vitaldb_waveforms/`:
    1. Carga sus anotaciones desde `data/raw/physionet_annotations/`.
    2. Aplica filtros base (`Noise`, `bad_signal_quality`, etiquetas nulas).
    3. Construye ventanas alrededor de cada latido para cada tamaño.
    4. Calcula features temporales sobre la ventana + RR locales por latido.

Salidas (no se versionan, `.gitignore` las bloquea):
    * data/processed/features_w1p2s.parquet
    * data/processed/features_w2p0s.parquet
    * data/processed/features_w5p0s.parquet

Cada parquet contiene una fila por ventana con las columnas:
    * Metadatos: `case_id`, `time_second`, `rhythm_label`, `beat_type`,
      `window_seconds`, `beat_index`, `start_sample`, `end_sample`.
    * Features temporales: `mean`, `std`, `var`, `min`, `max`, `range`,
      `median`, `p25`, `p75`, `iqr`, `skew`, `kurtosis`, `energy`,
      `zero_crossing_rate`, `abs_mean`.
    * Features RR locales: `rr_prev`, `rr_next`, `rr_mean_local`, `rr_ratio`.

`beat_type` se conserva únicamente para análisis descriptivo. NO debe usarse
como predictor (bloqueado por `src.modeling.assert_no_forbidden_features`).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loading import load_annotations_for_case  # noqa: E402
from src.features import (  # noqa: E402
    compute_per_beat_rr_features,
    compute_time_features_batch,
)
from src.preprocessing import apply_basic_filters  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402
from src.windowing import build_windows_for_case  # noqa: E402


# Tamaños de ventana objetivo del proyecto.
WINDOW_SIZES_SECONDS: tuple[float, ...] = (1.2, 2.0, 5.0)


def _window_filename(window_seconds: float) -> str:
    """Convierte 1.2 -> 'features_w1p2s.parquet', 2.0 -> 'features_w2p0s.parquet'."""
    txt = f"{window_seconds:.1f}".replace(".", "p")
    return f"features_w{txt}s.parquet"


def _discover_available_case_ids() -> list[int]:
    """Lista los `case_id` cuyo `.npy` está cacheado en disco."""
    ids: list[int] = []
    for path in sorted(config.VITALDB_WAVEFORMS_DIR.glob("case_*.npy")):
        m = re.match(r"case_(\d+)\.npy$", path.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def _build_for_case(case_id: int,
                    signal: np.ndarray,
                    beats_filtered: pd.DataFrame,
                    window_seconds: float,
                    fs_hz: int) -> pd.DataFrame:
    """Construye ventanas y features para un caso y un tamaño de ventana."""
    windows, specs = build_windows_for_case(
        signal=signal,
        beats=beats_filtered,
        case_id=case_id,
        fs_hz=fs_hz,
        window_seconds=window_seconds,
        overlap=0.0,
    )
    if windows.shape[0] == 0:
        return pd.DataFrame()

    # Features temporales sobre la señal de cada ventana.
    time_feats = compute_time_features_batch(windows).reset_index(drop=True)

    # RR locales por latido — alineados al índice de `beats_filtered`.
    rr_df = compute_per_beat_rr_features(
        beats_filtered[config.BEAT_TIME_COLUMN].to_numpy()
    )

    # Metadatos por ventana derivados del WindowSpec + lookup contra beats_filtered.
    meta_rows = []
    for s in specs:
        # beat_index del WindowSpec = índice posicional dentro de beats_filtered.
        beat_row = beats_filtered.iloc[s.beat_index]
        meta_rows.append(
            {
                config.CASE_ID_COLUMN: s.case_id,
                "beat_index": s.beat_index,
                "start_sample": s.start_sample,
                "end_sample": s.end_sample,
                config.BEAT_TIME_COLUMN: float(beat_row[config.BEAT_TIME_COLUMN]),
                config.TARGET_COLUMN: s.label,
                config.BEAT_TYPE_COLUMN: (
                    None
                    if pd.isna(beat_row.get(config.BEAT_TYPE_COLUMN))
                    else str(beat_row[config.BEAT_TYPE_COLUMN])
                ),
                "window_seconds": float(window_seconds),
            }
        )
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    # RR features alineadas vía beat_index.
    rr_aligned = rr_df.iloc[meta_df["beat_index"].to_numpy()].reset_index(drop=True)

    out = pd.concat(
        [meta_df, rr_aligned, time_feats],
        axis=1,
    )
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limitar a los primeros N casos disponibles (útil para --debug).",
    )
    p.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="Lista separada por comas. Si se da, ignora --max-cases.",
    )
    p.add_argument(
        "--window-seconds",
        type=str,
        default=None,
        help=(
            "Lista separada por comas. Por defecto procesa los 3 tamaños del "
            "proyecto (1.2, 2.0, 5.0)."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Modo rápido: --max-cases 3 si no se especifica otra cosa.",
    )
    return p.parse_args()


def _select_window_sizes(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return WINDOW_SIZES_SECONDS
    parsed = tuple(float(x.strip()) for x in arg.split(",") if x.strip())
    return parsed


def _select_case_ids(args: argparse.Namespace, available: list[int]) -> list[int]:
    if args.case_ids:
        wanted = {int(x.strip()) for x in args.case_ids.split(",") if x.strip()}
        return [cid for cid in available if cid in wanted]
    if args.max_cases is not None:
        return available[: args.max_cases]
    if args.debug:
        return available[:3]
    return available


def main() -> int:
    args = _parse_args()
    logger = get_logger("build_features")

    ensure_dir(config.PROCESSED_DIR)

    available = _discover_available_case_ids()
    if not available:
        logger.error(
            "No hay archivos `case_<id>.npy` en %s. Ejecuta primero "
            "scripts/01_download_all_available_ecg.py.",
            config.VITALDB_WAVEFORMS_DIR,
        )
        return 1

    case_ids = _select_case_ids(args, available)
    window_sizes = _select_window_sizes(args.window_seconds)

    logger.info("Casos disponibles en disco: %d", len(available))
    logger.info("Casos a procesar:           %d", len(case_ids))
    logger.info("Tamaños de ventana:         %s", window_sizes)

    # Pre-cargar anotaciones filtradas por caso (no depende del tamaño de ventana).
    beats_by_case: dict[int, pd.DataFrame] = {}
    for cid in case_ids:
        try:
            beats = load_annotations_for_case(cid)
        except FileNotFoundError as exc:
            logger.warning("case_id=%d sin anotaciones (%s). Se omite.", cid, exc)
            continue

        beats_f = apply_basic_filters(beats)
        # apply_basic_filters NO elimina NaN en la etiqueta — hacerlo explícito.
        beats_f = beats_f.dropna(subset=[config.TARGET_COLUMN])
        # Bloqueo defensivo de cadena literal "nan".
        mask = (
            beats_f[config.TARGET_COLUMN]
            .astype(str).str.strip().str.lower()
            .isin({"nan", "none", ""})
        )
        beats_f = beats_f.loc[~mask].copy()
        beats_f = beats_f.sort_values(config.BEAT_TIME_COLUMN).reset_index(drop=True)
        if beats_f.empty:
            logger.warning("case_id=%d sin latidos válidos tras filtros.", cid)
            continue
        beats_by_case[cid] = beats_f

    logger.info("Casos con latidos válidos: %d", len(beats_by_case))

    # Generar un parquet por tamaño de ventana.
    for w in window_sizes:
        out_path = config.PROCESSED_DIR / _window_filename(w)
        logger.info("===== Procesando ventana = %.1fs -> %s =====", w, out_path.name)
        per_case_frames: list[pd.DataFrame] = []

        for cid, beats_f in beats_by_case.items():
            npy_path = config.VITALDB_WAVEFORMS_DIR / f"case_{cid}.npy"
            try:
                signal = np.load(npy_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("case_id=%d: error cargando %s: %s", cid, npy_path, exc)
                continue

            t0 = time.time()
            try:
                df_case = _build_for_case(
                    case_id=cid,
                    signal=signal,
                    beats_filtered=beats_f,
                    window_seconds=w,
                    fs_hz=config.DEFAULT_ECG_FS_HZ,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "case_id=%d: error generando features (w=%.1fs): %s",
                    cid, w, exc,
                )
                logger.debug(traceback.format_exc())
                continue

            elapsed = time.time() - t0
            logger.info(
                "  case_id=%d | ventanas=%d | %.1fs",
                cid, len(df_case), elapsed,
            )
            if not df_case.empty:
                per_case_frames.append(df_case)

        if not per_case_frames:
            logger.error("No se generaron features para w=%.1fs. Skip.", w)
            continue

        full = pd.concat(per_case_frames, axis=0, ignore_index=True)
        full.to_parquet(out_path, index=False)
        logger.info(
            "Guardado %s | shape=%s | clases=%s",
            out_path.name, full.shape, sorted(full[config.TARGET_COLUMN].unique().tolist()),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
