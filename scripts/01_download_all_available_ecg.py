"""[LEGACY — descarga de ECG crudo desde VitalDB, pausada en esta fase]

Este script no es parte del flujo de modelado actual, que es **tabular** y
solo usa anotaciones y metadata ya presentes en
``data/raw/physionet_annotations/``. El script se conserva por si más
adelante se reactiva la línea de ECG crudo.

------------------------------------------------------------------------

Descarga todos los ECG faltantes desde VitalDB.

Recorre `metadata.csv`, identifica qué `case_id` ya están cacheados como
`data/raw/vitaldb_waveforms/case_<id>.npy` y descarga únicamente los
faltantes. Tolerante a errores: si un caso falla (timeout, track ausente,
formato inesperado), registra el error y continúa con el siguiente.

Genera además `reports/tables/download_status.csv` con una fila por
`case_id` y columnas:
    * case_id        — identificador del caso.
    * status         — ``cached`` (ya estaba en disco), ``downloaded``
                       (se bajó en esta corrida) o ``error``.
    * n_samples      — longitud de la señal en muestras (NaN si error).
    * duration_sec   — duración en segundos según `DEFAULT_ECG_FS_HZ`.
    * path           — ruta relativa del `.npy` (vacío si error).
    * error          — mensaje del error (vacío si OK).

Uso típico:

    python scripts/01_download_all_available_ecg.py
    python scripts/01_download_all_available_ecg.py --limit 10
    python scripts/01_download_all_available_ecg.py --case-ids 1001,1002

Los datos crudos quedan bajo `data/raw/vitaldb_waveforms/`, que está
excluida del repositorio por `.gitignore`.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Hacer importable `src` cuando se ejecuta desde la raíz del repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loading import load_metadata  # noqa: E402
from src.download import load_ecg_from_vitaldb, save_ecg_npy  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesar como máximo N casos (en orden de aparición en metadata.csv).",
    )
    p.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="Lista separada por comas. Si se da, ignora --limit y procesa solo estos `case_id`.",
    )
    p.add_argument(
        "--track-name",
        type=str,
        default=config.DEFAULT_ECG_TRACK_NAME,
        help=f"Canal ECG a solicitar a VitalDB (default: {config.DEFAULT_ECG_TRACK_NAME}).",
    )
    p.add_argument(
        "--sampling-rate-hz",
        type=int,
        default=config.DEFAULT_ECG_FS_HZ,
        help=f"Frecuencia de muestreo objetivo en Hz (default: {config.DEFAULT_ECG_FS_HZ}).",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Pausa en segundos entre descargas (para no saturar la API).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-descargar incluso si ya existe el .npy en disco.",
    )
    p.add_argument(
        "--output-status",
        type=Path,
        default=config.TABLES_DIR / "download_status.csv",
        help="Ruta del CSV de estado a generar.",
    )
    return p.parse_args()


def _select_case_ids(args: argparse.Namespace, metadata: pd.DataFrame) -> list[int]:
    if args.case_ids:
        ids = [int(x.strip()) for x in args.case_ids.split(",") if x.strip()]
        # Mantener solo los que aparecen en metadata para evitar requests imposibles.
        valid = set(metadata[config.CASE_ID_COLUMN].astype(int).tolist())
        return [cid for cid in ids if cid in valid]

    all_ids = metadata[config.CASE_ID_COLUMN].astype(int).tolist()
    if args.limit is not None:
        return all_ids[: args.limit]
    return all_ids


def main() -> int:
    args = _parse_args()
    logger = get_logger("download_all_ecg")

    ensure_dir(config.VITALDB_WAVEFORMS_DIR)
    ensure_dir(config.TABLES_DIR)

    metadata = load_metadata()
    case_ids = _select_case_ids(args, metadata)
    logger.info("Casos a evaluar: %d", len(case_ids))

    rows: list[dict] = []

    for i, cid in enumerate(case_ids, start=1):
        out_path = config.VITALDB_WAVEFORMS_DIR / f"case_{cid}.npy"
        rel_path = out_path.relative_to(config.PROJECT_ROOT).as_posix()

        # Cache hit.
        if out_path.exists() and not args.force:
            try:
                arr = np.load(out_path, mmap_mode="r")
                n = int(arr.shape[0])
            except Exception:  # noqa: BLE001 — diagnóstico defensivo
                n = -1
            duration = (n / args.sampling_rate_hz) if n > 0 else float("nan")
            rows.append(
                {
                    "case_id": cid,
                    "status": "cached",
                    "n_samples": n,
                    "duration_sec": duration,
                    "path": rel_path,
                    "error": "",
                }
            )
            logger.info("[%d/%d] case_id=%d cached (n=%d)", i, len(case_ids), cid, n)
            continue

        # Descarga.
        try:
            signal = load_ecg_from_vitaldb(
                cid,
                track_name=args.track_name,
                sampling_rate_hz=args.sampling_rate_hz,
            )
            save_ecg_npy(signal, cid, output_dir=config.VITALDB_WAVEFORMS_DIR)
            n = int(signal.shape[0])
            duration = n / args.sampling_rate_hz
            rows.append(
                {
                    "case_id": cid,
                    "status": "downloaded",
                    "n_samples": n,
                    "duration_sec": duration,
                    "path": rel_path,
                    "error": "",
                }
            )
            logger.info(
                "[%d/%d] case_id=%d downloaded (n=%d, %.1fs)",
                i, len(case_ids), cid, n, duration,
            )
        except KeyboardInterrupt:
            logger.warning("Interrumpido por el usuario tras %d casos.", i - 1)
            break
        except Exception as exc:  # noqa: BLE001 — la idea es continuar pese a fallos
            err = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "case_id": cid,
                    "status": "error",
                    "n_samples": float("nan"),
                    "duration_sec": float("nan"),
                    "path": "",
                    "error": err,
                }
            )
            logger.error("[%d/%d] case_id=%d ERROR: %s", i, len(case_ids), cid, err)
            logger.debug(traceback.format_exc())

        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    # Persistir status CSV.
    status_df = pd.DataFrame(rows, columns=["case_id", "status", "n_samples", "duration_sec", "path", "error"])
    args.output_status.parent.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(args.output_status, index=False)
    logger.info("Status guardado en %s", args.output_status)

    # Resumen final.
    counts = status_df["status"].value_counts().to_dict()
    logger.info("Resumen: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
