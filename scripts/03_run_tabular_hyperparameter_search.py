"""CLI: búsqueda de hiperparámetros sobre el dataset tabular filtrado.

Lee el parquet generado por `scripts/02_build_filtered_tabular_modeling_dataset.py`,
hace split 80/20 por ``case_id``, corre ``RandomizedSearchCV`` por modelo con
CV por grupo (``StratifiedGroupKFold`` con fallback a ``GroupKFold``) y
persiste todos los CSVs/figuras requeridas en ``reports/``.

Restricciones (codificadas en `src/tabular_search.py`):
    * ``case_id`` solo se usa como grupo, nunca como feature.
    * ``beat_type`` y otras columnas en ``config.TABULAR_LEAKAGE_COLUMNS``
      están bloqueadas (asserts en runtime).
    * El test no se toca durante la búsqueda; solo al final.

Uso típico:
    # Full run (puede tardar horas según hardware)
    python scripts/03_run_tabular_hyperparameter_search.py

    # Debug rápido
    python scripts/03_run_tabular_hyperparameter_search.py --debug

    # Subset de casos / modelos
    python scripts/03_run_tabular_hyperparameter_search.py --max-cases 100 --models logreg,random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    class_support_per_split,
    classes_missing_in_train,
    confusion_matrix_with_totals,
    per_class_report,
)
from src.modeling import make_train_test_group_split_with_coverage  # noqa: E402
from src.tabular_search import (  # noqa: E402
    PRIMARY_SCORING,
    SCORING_METRICS,
    TABULAR_MODEL_NAMES,
    build_cv_splitter,
    classify_features,
    evaluate_on_test,
    extract_feature_importance,
    load_tabular_modeling_dataset,
    run_search_for_model,
)
from src.utils import ensure_dir, get_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Lista separada por coma. Disponibles: {','.join(TABULAR_MODEL_NAMES)}. Default: todos.",
    )
    p.add_argument("--n-iter", type=int, default=30, help="Combinaciones por modelo (default: 30).")
    p.add_argument("--n-splits", type=int, default=5, help="Folds CV (default: 5).")
    p.add_argument("--test-size", type=float, default=0.2, help="Fracción objetivo de test (default: 0.20).")
    p.add_argument("--random-state", type=int, default=config.RANDOM_SEED)
    p.add_argument("--n-jobs", type=int, default=-1, help="Procesos paralelos.")
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Subsamplear N case_id antes del split (para debug). Default: todos.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Modo rápido: --max-cases 100, --n-iter 3, --n-splits 2, --n-jobs 1.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help="Carpeta destino (default: reports/).",
    )
    return p.parse_args()


def _select_models(arg: str | None) -> list[str]:
    if not arg:
        return list(TABULAR_MODEL_NAMES)
    names = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [m for m in names if m not in TABULAR_MODEL_NAMES]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {list(TABULAR_MODEL_NAMES)}")
    return names


def _subsample_cases(df: pd.DataFrame, max_cases: int, random_state: int) -> pd.DataFrame:
    """Subsamplea por `case_id` manteniendo todas las filas de los casos elegidos."""
    rng = np.random.RandomState(random_state)
    all_ids = df[config.CASE_ID_COLUMN].drop_duplicates().to_numpy()
    if max_cases >= len(all_ids):
        return df
    chosen = rng.choice(all_ids, size=max_cases, replace=False)
    return df.loc[df[config.CASE_ID_COLUMN].isin(chosen)].copy()


def _plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path,
                           normalize: bool = False) -> bool:
    from sklearn.metrics import confusion_matrix
    labels = sorted(set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique()), key=str)
    if normalize:
        # Solo válida si todas las clases reales del test tienen soporte > 0.
        y_series = pd.Series(y_true)
        if any(y_series.value_counts().reindex(labels, fill_value=0) == 0):
            return False
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        fmt = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fmt = "d"
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(labels), 1.0 + 0.7 * len(labels)))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    args = _parse_args()
    logger = get_logger("tabular_search")

    if args.debug:
        args.n_iter = 3
        args.n_splits = 2
        # Mantener n_jobs=-1 también en debug: con n_jobs=1 incluso 100 casos
        # tardan demasiado en RandomForest/XGBoost/MLP.
        if args.max_cases is None:
            args.max_cases = 60
        logger.warning("DEBUG mode ON: n_iter=3 n_splits=2 n_jobs=%d max_cases=%s",
                       args.n_jobs, args.max_cases)

    models = _select_models(args.models)
    figures_dir = ensure_dir(args.output_dir / "figures")
    tables_dir = ensure_dir(args.output_dir / "tables")

    # ------------------------------------------------------------------
    # 1. Carga + clasificación de columnas
    # ------------------------------------------------------------------
    df = load_tabular_modeling_dataset()
    logger.info("Dataset shape: %s | cases=%d | classes=%d",
                df.shape, df[config.CASE_ID_COLUMN].nunique(),
                df[config.TARGET_COLUMN].nunique())

    if args.max_cases is not None:
        df = _subsample_cases(df, args.max_cases, args.random_state)
        logger.info("Tras subsample: shape=%s | cases=%d", df.shape, df[config.CASE_ID_COLUMN].nunique())

    cls = classify_features(df)
    numeric = cls["numeric_features"]
    categorical = cls["categorical_features"]
    logger.info("Features: %d numéricas + %d categóricas", len(numeric), len(categorical))
    logger.info("Leakage excluido: %s", cls["leakage_excluded"])
    logger.info("Alta cardinalidad excluido: %s", cls["high_cardinality_excluded"])
    if cls["constant_excluded"]:
        logger.info("Constantes excluidas: %s", cls["constant_excluded"])

    # Persistir lista de features usadas y columnas excluidas por leakage
    pd.DataFrame({
        "feature": numeric + categorical,
        "kind": ["numeric"] * len(numeric) + ["categorical"] * len(categorical),
    }).to_csv(tables_dir / "tabular_feature_list_used.csv", index=False)
    pd.DataFrame({
        "column": cls["leakage_excluded"] + cls["high_cardinality_excluded"] + cls["constant_excluded"],
        "reason": (
            ["leakage"] * len(cls["leakage_excluded"]) +
            ["high_cardinality"] * len(cls["high_cardinality_excluded"]) +
            ["constant"] * len(cls["constant_excluded"])
        ),
    }).to_csv(tables_dir / "tabular_excluded_columns_leakage.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Split por case_id
    # ------------------------------------------------------------------
    X_df = df[numeric + categorical]
    y = df[config.TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()

    train_idx, test_idx, split_info = make_train_test_group_split_with_coverage(
        X_df, y, groups,
        test_size=args.test_size,
        random_state=args.random_state,
        max_attempts=200,
    )
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    assert set(groups_train).isdisjoint(set(groups_test)), "Fuga de grupo."

    # Persistir resumen del split y soporte por clase
    split_summary_rows = [
        {"metric": "chosen_seed", "value": split_info["chosen_seed"]},
        {"metric": "n_classes_covered", "value": split_info["n_classes_covered"]},
        {"metric": "n_total_classes", "value": split_info["n_total_classes"]},
        {"metric": "n_train_groups", "value": len(split_info["train_groups"])},
        {"metric": "n_test_groups", "value": len(split_info["test_groups"])},
        {"metric": "n_train_rows", "value": int(len(train_idx))},
        {"metric": "n_test_rows", "value": int(len(test_idx))},
        {"metric": "actual_test_fraction", "value": round(split_info["actual_test_fraction"], 4)},
        {"metric": "requested_test_size", "value": split_info["requested_test_size"]},
        {"metric": "classes_only_in_train", "value": ",".join(map(str, split_info["classes_only_in_train"]))},
        {"metric": "classes_only_in_test", "value": ",".join(map(str, split_info["classes_only_in_test"]))},
    ]
    pd.DataFrame(split_summary_rows).to_csv(tables_dir / "tabular_train_test_split_summary.csv", index=False)

    sup_df = class_support_per_split(y_train, y_test).reset_index(names="class")
    sup_df.to_csv(tables_dir / "tabular_class_support_train_test.csv", index=False)
    logger.info("Soporte por clase:\n%s", sup_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. CV builder
    # ------------------------------------------------------------------
    cv, cv_name, n_splits_eff = build_cv_splitter(
        groups_train=groups_train,
        y_train=y_train,
        n_splits=args.n_splits,
        prefer_stratified=True,
    )
    logger.info("CV: %s | n_splits_efectivo=%d", cv_name, n_splits_eff)

    # ------------------------------------------------------------------
    # 4. Búsqueda por modelo
    # ------------------------------------------------------------------
    model_results: list[dict] = []
    fitted_models: dict[str, object] = {}
    for model_name in models:
        logger.info("===== %s =====", model_name)
        t0 = time.time()
        try:
            res = run_search_for_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                numeric_features=numeric,
                categorical_features=categorical,
                cv=cv,
                n_iter=args.n_iter,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            res = evaluate_on_test(res, X_test, y_test)
            logger.info(
                "  ok  | cv_%s=%.3f  test_%s=%.3f  fit=%.1fs",
                PRIMARY_SCORING, res.cv_metrics.get(f"cv_{PRIMARY_SCORING}", float("nan")),
                PRIMARY_SCORING, res.test_metrics.get(f"test_{PRIMARY_SCORING}", float("nan")),
                res.fit_seconds,
            )
            fitted_models[model_name] = res.best_estimator
            row = {
                "model": res.model,
                "status": res.status,
                "fit_seconds": res.fit_seconds,
                "n_iter_effective": res.n_iter_effective,
                "best_cv_score_primary": res.best_cv_score_primary,
                **res.cv_metrics,
                **res.test_metrics,
                "best_params_json": json.dumps(res.best_params, default=str),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("  ERROR en %s: %s", model_name, exc)
            row = {
                "model": model_name,
                "status": "error",
                "fit_seconds": time.time() - t0,
                "n_iter_effective": 0,
                "best_cv_score_primary": float("nan"),
                **{f"cv_{m}": float("nan") for m in SCORING_METRICS},
                **{f"test_{m}": float("nan") for m in SCORING_METRICS},
                "best_params_json": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        model_results.append(row)

    # ------------------------------------------------------------------
    # 5. Tablas comparativas
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(model_results)
    cv_cols = ["model", "status", "n_iter_effective", "fit_seconds", "best_cv_score_primary"] + [f"cv_{m}" for m in SCORING_METRICS]
    test_cols = ["model", "status"] + [f"test_{m}" for m in SCORING_METRICS]
    cv_view = results_df.reindex(columns=[c for c in cv_cols if c in results_df.columns])
    test_view = results_df.reindex(columns=[c for c in test_cols if c in results_df.columns])
    cv_view.to_csv(tables_dir / "tabular_model_comparison_cv.csv", index=False)
    test_view.to_csv(tables_dir / "tabular_model_comparison_test.csv", index=False)

    bhp_cols = ["model", "status", "best_params_json"]
    results_df.reindex(columns=bhp_cols).to_csv(tables_dir / "tabular_best_hyperparameters.csv", index=False)

    logger.info("Comparativa CV:\n%s", cv_view.round(3).to_string(index=False))
    logger.info("Comparativa test:\n%s", test_view.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Mejor modelo + artefactos asociados
    # ------------------------------------------------------------------
    ok = results_df.loc[results_df["status"] == "ok"].copy()
    winner = None
    if not ok.empty and f"test_{PRIMARY_SCORING}" in ok.columns:
        ok = ok.dropna(subset=[f"test_{PRIMARY_SCORING}"])
        if not ok.empty:
            winner = ok.sort_values(f"test_{PRIMARY_SCORING}", ascending=False).iloc[0].to_dict()

    if winner is None:
        logger.warning("Sin ganador válido. Revisar errores en logs.")
        pd.DataFrame([{"note": "Sin modelos válidos. Revisar logs."}]).to_csv(
            tables_dir / "tabular_best_model_classification_report.csv", index=False
        )
    else:
        m_name = winner["model"]
        logger.info("Mejor modelo: %s (test_%s=%.3f)",
                    m_name, PRIMARY_SCORING, winner[f"test_{PRIMARY_SCORING}"])

        best_est = fitted_models[m_name]
        y_pred = best_est.predict(X_test)

        # Reporte por clase
        rep_df = per_class_report(y_test, y_pred)
        rep_df.reset_index(names="class_or_avg").to_csv(
            tables_dir / "tabular_best_model_classification_report.csv", index=False
        )

        # Matrices de confusión
        cm_full = confusion_matrix_with_totals(y_test, y_pred)
        cm_full.reset_index(names="true_class").to_csv(
            tables_dir / "tabular_confusion_matrix_absolute.csv", index=False
        )

        _plot_confusion_matrix(
            y_true=y_test, y_pred=y_pred,
            title=f"Matriz confusión absoluta — {m_name}",
            output_path=figures_dir / "tabular_best_model_confusion_matrix_absolute.png",
            normalize=False,
        )
        ok_norm = _plot_confusion_matrix(
            y_true=y_test, y_pred=y_pred,
            title=f"Matriz confusión normalizada por fila — {m_name}",
            output_path=figures_dir / "tabular_best_model_confusion_matrix_normalized.png",
            normalize=True,
        )
        if not ok_norm:
            logger.warning("Matriz normalizada omitida (hay clases sin soporte en y_true del test).")

        # Feature importance
        fi = extract_feature_importance(best_est)
        if fi is not None:
            fi.to_csv(tables_dir / "tabular_feature_importance_best_model.csv", index=False)
            logger.info("Guardada importancia de features (%d filas).", len(fi))
        else:
            pd.DataFrame([{"note": f"No fue posible extraer importancia de features para {m_name}."}]).to_csv(
                tables_dir / "tabular_feature_importance_best_model.csv", index=False
            )
            logger.warning("No se pudo extraer feature importance para %s.", m_name)

    # ------------------------------------------------------------------
    # 7. Meta JSON con todo el estado del run
    # ------------------------------------------------------------------
    meta = {
        "dataset_shape": list(df.shape),
        "n_cases": int(df[config.CASE_ID_COLUMN].nunique()),
        "n_classes": int(df[config.TARGET_COLUMN].nunique()),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "leakage_excluded": cls["leakage_excluded"],
        "high_cardinality_excluded": cls["high_cardinality_excluded"],
        "constant_excluded": cls["constant_excluded"],
        "split_info": split_info,
        "cv": {"splitter": cv_name, "n_splits_effective": n_splits_eff},
        "models_results": model_results,
        "winner": winner,
        "args": {
            "n_iter": args.n_iter,
            "n_splits": args.n_splits,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
            "max_cases": args.max_cases,
            "debug": args.debug,
            "models": models,
        },
    }
    with open(tables_dir / "tabular_hyperparameter_search_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Outputs guardados en %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
