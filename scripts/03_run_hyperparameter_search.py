"""[LEGACY — búsqueda sobre features ECG, pausada en esta fase]

Para el flujo activo, ver
``scripts/03_run_tabular_hyperparameter_search.py``.

------------------------------------------------------------------------

CLI: corre la búsqueda de hiperparámetros multi-modelo y multi-ventana.

Lee los parquets de features (``features_w1p2s.parquet``, ``features_w2p0s.parquet``,
``features_w5p0s.parquet``), hace split 80/20 por ``case_id`` con cobertura
de clases, ejecuta ``RandomizedSearchCV`` por modelo con CV por grupo
(``StratifiedGroupKFold`` cuando es viable; ``GroupKFold`` como fallback) y
guarda los outputs en ``reports/``.

Uso típico:

    # Full run (puede tardar horas con el dataset completo).
    python scripts/03_run_hyperparameter_search.py

    # Debug: pocas iteraciones, todos los modelos, mismo flujo.
    python scripts/03_run_hyperparameter_search.py --debug

    # Filtrar a un subconjunto.
    python scripts/03_run_hyperparameter_search.py --models logreg,random_forest --windows 2.0

Restricciones:
    * El test no se toca durante la búsqueda; solo al final.
    * Toda la CV interna es por ``case_id``.
    * ``beat_type`` está bloqueado como predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin display para entornos headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    confusion_matrix_with_totals,
    per_class_report,
)
from src.search import (  # noqa: E402
    MODEL_REGISTRY,
    PRIMARY_SCORING,
    WindowRunConfig,
    assemble_best_hyperparameters_table,
    assemble_class_support_table,
    assemble_comparison_table,
    assemble_missing_classes_table,
    pick_best_overall,
    run_one_window,
)
from src.utils import ensure_dir, get_logger  # noqa: E402


WINDOW_FILENAMES = {
    1.2: "features_w1p2s.parquet",
    2.0: "features_w2p0s.parquet",
    5.0: "features_w5p0s.parquet",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--windows",
        type=str,
        default=None,
        help="Lista de tamaños de ventana en segundos, ej '1.2,2.0,5.0'. Default: todos.",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            f"Lista de modelos separados por coma. Disponibles: "
            f"{','.join(MODEL_REGISTRY.keys())}. Default: todos."
        ),
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Combinaciones por modelo en RandomizedSearchCV (default: 30).",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Folds máximos para CV interna (se recortan si hay menos grupos).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción objetivo del test set (default: 0.20).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=config.RANDOM_SEED,
        help="Semilla base.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Procesos paralelos para RandomizedSearchCV (default: -1).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Modo rápido: n_iter=3, n_splits=2, sin paralelismo.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help="Carpeta raíz de outputs (default: reports/).",
    )
    return p.parse_args()


def _select_windows(arg: str | None) -> list[float]:
    if not arg:
        return list(WINDOW_FILENAMES.keys())
    return [float(x.strip()) for x in arg.split(",") if x.strip()]


def _select_models(arg: str | None) -> list[str]:
    if not arg:
        return list(MODEL_REGISTRY.keys())
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [m for m in requested if m not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {list(MODEL_REGISTRY.keys())}")
    return requested


def _confusion_matrix_figure(y_true, y_pred, title: str, output_path: Path,
                             normalize: bool = False) -> bool:
    """Genera un heatmap de matriz de confusión.

    Returns
    -------
    bool
        ``True`` si la figura es válida y se guardó; ``False`` si la
        normalización no aplica (todas las clases reales con soporte 0, etc.).
    """
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique()), key=str)

    if normalize:
        # Solo es válida si todas las clases reales tienen soporte > 0 en y_true.
        y_series = pd.Series(y_true)
        if any(y_series.value_counts().reindex(labels, fill_value=0) == 0):
            return False
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        fmt = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(1.0 + 1.0 * len(labels), 1.0 + 0.8 * len(labels)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return True


def _pivot_by_window(comparison_df: pd.DataFrame,
                     metric: str = "test_f1_macro") -> pd.DataFrame:
    """Pivota a wide: una fila por modelo, una columna por ventana, valores = metric."""
    if metric not in comparison_df.columns:
        return pd.DataFrame()
    return (
        comparison_df
        .pivot_table(index="model", columns="window_seconds", values=metric, aggfunc="first")
        .sort_index()
    )


def main() -> int:
    args = _parse_args()
    logger = get_logger("hp_search")

    if args.debug:
        args.n_iter = 3
        args.n_splits = 2
        args.n_jobs = 1
        logger.warning("DEBUG mode ON: n_iter=3, n_splits=2, n_jobs=1.")

    windows = _select_windows(args.windows)
    models_to_run = _select_models(args.models)

    figures_dir = ensure_dir(args.output_dir / "figures")
    tables_dir = ensure_dir(args.output_dir / "tables")

    logger.info("Ventanas: %s", windows)
    logger.info("Modelos: %s", models_to_run)
    logger.info("n_iter=%d | n_splits=%d | test_size=%.2f", args.n_iter, args.n_splits, args.test_size)

    window_results: list[dict] = []
    for w in windows:
        if w not in WINDOW_FILENAMES:
            logger.error("Ventana %.1fs no es uno de los tamaños soportados (1.2, 2.0, 5.0).", w)
            continue
        parquet_path = config.PROCESSED_DIR / WINDOW_FILENAMES[w]
        if not parquet_path.exists():
            logger.error(
                "Falta %s. Ejecuta primero scripts/02_build_features_all_windows.py.",
                parquet_path,
            )
            continue

        logger.info("===== Ventana %.1fs =====", w)
        cfg = WindowRunConfig(
            window_seconds=w,
            parquet_path=parquet_path,
            models_to_run=models_to_run,
            n_iter=args.n_iter,
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        t0 = time.time()
        wr = run_one_window(cfg)
        logger.info(
            "Ventana %.1fs lista en %.1fs | split=%s | cv=%s",
            w,
            time.time() - t0,
            {k: wr["split_info"][k] for k in ("chosen_seed", "n_classes_covered", "n_total_classes", "actual_test_fraction")},
            wr["cv_info"],
        )
        for mr in wr["models"]:
            status = mr.get("status", "ok")
            if status == "ok":
                logger.info(
                    "  %-14s status=ok  cv_%s=%.3f  test_%s=%.3f  fit=%.1fs",
                    mr["model"], PRIMARY_SCORING,
                    mr["cv_metrics"].get(f"cv_{PRIMARY_SCORING}", float("nan")),
                    PRIMARY_SCORING,
                    mr["test_metrics"].get(f"test_{PRIMARY_SCORING}", float("nan")),
                    mr.get("fit_seconds", float("nan")),
                )
            else:
                logger.warning("  %-14s status=error  %s", mr["model"], mr.get("error", ""))
        window_results.append(wr)

    if not window_results:
        logger.error("Ninguna ventana procesada. Aborto.")
        return 1

    # ------------------------------------------------------------------
    # CSV: comparación global (long format)
    # ------------------------------------------------------------------
    comparison_df = assemble_comparison_table(window_results)
    comparison_df.to_csv(tables_dir / "full_model_comparison.csv", index=False)
    logger.info("Guardado %s", tables_dir / "full_model_comparison.csv")

    # ------------------------------------------------------------------
    # CSV: pivote por ventana (wide; rows=model, cols=window)
    # ------------------------------------------------------------------
    pivot = _pivot_by_window(comparison_df, metric=f"test_{PRIMARY_SCORING}")
    pivot.to_csv(tables_dir / "full_model_comparison_by_window.csv")
    logger.info("Guardado %s", tables_dir / "full_model_comparison_by_window.csv")

    # ------------------------------------------------------------------
    # CSV: mejores hiperparámetros
    # ------------------------------------------------------------------
    bhp = assemble_best_hyperparameters_table(window_results)
    bhp.to_csv(tables_dir / "best_hyperparameters.csv", index=False)
    logger.info("Guardado %s", tables_dir / "best_hyperparameters.csv")

    # ------------------------------------------------------------------
    # CSV: soporte por clase en train/test por ventana
    # ------------------------------------------------------------------
    cs = assemble_class_support_table(window_results)
    cs.to_csv(tables_dir / "class_support_train_test_by_window.csv", index=False)
    logger.info("Guardado %s", tables_dir / "class_support_train_test_by_window.csv")

    # ------------------------------------------------------------------
    # CSV: clases ausentes por split
    # ------------------------------------------------------------------
    cm_missing = assemble_missing_classes_table(window_results)
    cm_missing.to_csv(tables_dir / "classes_missing_by_split.csv", index=False)
    logger.info("Guardado %s", tables_dir / "classes_missing_by_split.csv")

    # ------------------------------------------------------------------
    # Mejor modelo global y artefactos asociados
    # ------------------------------------------------------------------
    winner = pick_best_overall(comparison_df, primary=f"test_{PRIMARY_SCORING}")
    if winner is None:
        logger.warning("No hay un mejor modelo válido (todos los runs fallaron o son NaN).")
        # Persistir un placeholder para que ChatGPT vea que se intentó.
        pd.DataFrame([{"note": "Sin modelos válidos. Revisar logs."}]).to_csv(
            tables_dir / "test_classification_report_best_model.csv", index=False
        )
    else:
        w_win = float(winner["window_seconds"])
        m_win = str(winner["model"])
        logger.info(
            "Mejor modelo global: %s @ ventana=%.1fs (test_%s=%.3f)",
            m_win, w_win, PRIMARY_SCORING, winner[f"test_{PRIMARY_SCORING}"],
        )

        # Localizar el WindowRun y el ModelResult del ganador.
        winning_wr = next(wr for wr in window_results if abs(wr["window_seconds"] - w_win) < 1e-9)
        winning_mr = next(mr for mr in winning_wr["models"] if mr["model"] == m_win and mr.get("status") == "ok")

        y_test = winning_wr["y_test"]
        y_pred = winning_mr["y_pred_test"]

        # CSV: classification_report por clase (incluye support)
        rep_df = per_class_report(y_test, y_pred)
        rep_df.reset_index(names="class_or_avg").to_csv(
            tables_dir / "test_classification_report_best_model.csv", index=False
        )
        logger.info("Guardado %s", tables_dir / "test_classification_report_best_model.csv")

        # PNG: matriz de confusión absoluta
        _confusion_matrix_figure(
            y_true=y_test,
            y_pred=y_pred,
            title=f"Matriz confusión (absoluta) — {m_win} @ {w_win:.1f}s",
            output_path=figures_dir / "confusion_matrix_best_model_absolute.png",
            normalize=False,
        )
        logger.info("Guardado %s", figures_dir / "confusion_matrix_best_model_absolute.png")

        # PNG: matriz normalizada (solo si todas las clases tienen soporte > 0 en test)
        ok = _confusion_matrix_figure(
            y_true=y_test,
            y_pred=y_pred,
            title=f"Matriz confusión (normalizada por fila) — {m_win} @ {w_win:.1f}s",
            output_path=figures_dir / "confusion_matrix_best_model_normalized.png",
            normalize=True,
        )
        if ok:
            logger.info("Guardado %s", figures_dir / "confusion_matrix_best_model_normalized.png")
        else:
            logger.warning("Matriz normalizada omitida: hay clases sin muestras en y_true del test.")

        # CSV adicional informativo: matriz absoluta con totales (útil para auditar)
        cm_full = confusion_matrix_with_totals(y_test, y_pred)
        cm_full.reset_index(names="true_class").to_csv(
            tables_dir / "test_confusion_matrix_best_model_with_totals.csv", index=False
        )
        logger.info("Guardado %s", tables_dir / "test_confusion_matrix_best_model_with_totals.csv")

    # ------------------------------------------------------------------
    # Dump JSON de meta-info: semillas elegidas, CV usada, ganador
    # ------------------------------------------------------------------
    meta = {
        "windows": [
            {
                "window_seconds": wr["window_seconds"],
                "parquet_path": wr["parquet_path"],
                "shape": list(wr["shape"]),
                "split_info": wr["split_info"],
                "cv_info": wr["cv_info"],
                "models": [
                    {
                        "model": mr["model"],
                        "status": mr.get("status", "ok"),
                        "n_iter_effective": mr.get("n_iter_effective", 0),
                        "fit_seconds": mr.get("fit_seconds", float("nan")),
                        "best_cv_score_primary": mr.get("best_cv_score_primary", float("nan")),
                        "best_params": mr.get("best_params"),
                        "cv_metrics": mr.get("cv_metrics", {}),
                        "test_metrics": mr.get("test_metrics", {}),
                        "error": mr.get("error"),
                    }
                    for mr in wr["models"]
                ],
            }
            for wr in window_results
        ],
        "winner": winner,
        "args": {
            "n_iter": args.n_iter,
            "n_splits": args.n_splits,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
            "debug": args.debug,
            "models": models_to_run,
            "windows": windows,
        },
    }
    meta_path = tables_dir / "hyperparameter_search_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Guardado %s", meta_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
