"""CLI: búsqueda de hiperparámetros sobre la tarea binaria
`normal_sinus` vs `arrhythmia_or_abnormal`.

Carga `data/processed/binary_rhythm_modeling_dataset.parquet`, hace split
80/20 por `case_id` que garantiza ambas clases binarias, ejecuta
`RandomizedSearchCV` por modelo con CV por grupo
(`StratifiedGroupKFold` o `GroupKFold`), evalúa el test una sola vez al
final, ajusta umbral con Youden J y F1 sobre validación interna del
train, y persiste todos los CSVs/figuras requeridos en `reports/`.

Uso:
    python scripts/06_run_binary_rhythm_model_search.py --debug
    python scripts/06_run_binary_rhythm_model_search.py --n-iter 30 --n-splits 5
    python scripts/06_run_binary_rhythm_model_search.py --models logreg_balanced,xgboost_binary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.binary_search import (  # noqa: E402
    BINARY_NON_FEATURE_METADATA_COLUMNS,
    BINARY_SCORING_METRICS,
    PRIMARY_BINARY_SCORING,
    build_binary_cv_splitter,
    build_binary_model_registry,
    classify_binary_features,
    get_positive_class_score,
    load_binary_modeling_dataset,
    make_binary_group_train_test_split_with_coverage,
    run_binary_search_for_model,
    select_threshold_max_f1,
    select_threshold_youden_j,
)
from src.utils import ensure_dir, get_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers de evaluación
# ---------------------------------------------------------------------------
def _binary_metrics(y_true, y_pred, y_score=None) -> dict:
    """Métricas binarias completas (positiva = arrhythmia_or_abnormal)."""
    pos = config.BINARY_POSITIVE_CLASS
    neg = config.BINARY_NEGATIVE_CLASS

    # Sensitivity = recall positive, Specificity = TNR (recall negative).
    cm = confusion_matrix(y_true, y_pred, labels=[neg, pos])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_abnormal": float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "precision_abnormal": float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "recall_abnormal_sensitivity": sensitivity,
        "specificity_normal": specificity,
        "n_tp": int(tp),
        "n_fp": int(fp),
        "n_fn": int(fn),
        "n_tn": int(tn),
    }
    if y_score is not None and len(set(y_true)) == 2:
        y_true_bin = (np.asarray(y_true) == pos).astype(int)
        try:
            m["roc_auc"] = float(roc_auc_score(y_true_bin, y_score))
            m["average_precision"] = float(average_precision_score(y_true_bin, y_score))
        except Exception:  # noqa: BLE001
            m["roc_auc"] = float("nan")
            m["average_precision"] = float("nan")
    else:
        m["roc_auc"] = float("nan")
        m["average_precision"] = float("nan")
    return m


def _plot_confusion(y_true, y_pred, title: str, output_path: Path, normalize: bool) -> bool:
    labels = [config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS]
    if normalize:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        fmt = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fmt = "d"
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return True


def _plot_roc(y_true_bin, y_score, title, output_path):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    auc = roc_auc_score(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("FPR (1 - specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_pr(y_true_bin, y_score, title, output_path):
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Recall (positive)")
    ax.set_ylabel("Precision (positive)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _extract_feature_importance(estimator) -> pd.DataFrame | None:
    """Importancia/coeficientes del clasificador final del pipeline."""
    try:
        feature_names = estimator.named_steps["preprocessor"].get_feature_names_out()
    except Exception:  # noqa: BLE001
        return None

    clf = estimator.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
        if len(importances) != len(feature_names):
            return None
        return pd.DataFrame({"feature": feature_names, "importance": importances}) \
            .sort_values("importance", ascending=False) \
            .reset_index(drop=True)

    # Wrapper XGB / similares: clf._xgb.feature_importances_
    if hasattr(clf, "_xgb") and hasattr(clf._xgb, "feature_importances_"):
        importances = np.asarray(clf._xgb.feature_importances_, dtype=float)
        if len(importances) != len(feature_names):
            return None
        return pd.DataFrame({"feature": feature_names, "importance": importances}) \
            .sort_values("importance", ascending=False) \
            .reset_index(drop=True)

    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_).ravel()
        if len(coef) != len(feature_names):
            return None
        return pd.DataFrame({"feature": feature_names, "coef": coef,
                             "abs_coef": np.abs(coef)}) \
            .sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return None


def _plot_feature_importance_top(fi_df: pd.DataFrame, output_path: Path, top: int = 20) -> None:
    if fi_df is None or fi_df.empty:
        return
    value_col = "importance" if "importance" in fi_df.columns else "abs_coef"
    top_df = fi_df.head(top)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_df) + 1))
    sns.barplot(x=top_df[value_col], y=top_df["feature"], ax=ax,
                hue=top_df["feature"], palette="viridis", legend=False)
    ax.set_title(f"Top {top} feature importance ({value_col})")
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(registry: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=str, default=None,
                   help=f"Lista separada por coma. Disponibles: {','.join(registry.keys())}.")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=config.RANDOM_SEED)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-cases", type=int, default=None,
                   help="Subsamplear N case_id antes del split (debug).")
    p.add_argument("--debug", action="store_true",
                   help="--max-cases 80, --n-iter 3, --n-splits 2.")
    p.add_argument("--output-dir", type=Path, default=config.REPORTS_DIR)
    return p.parse_args()


def _subsample_cases(df: pd.DataFrame, max_cases: int, random_state: int) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    all_ids = df[config.CASE_ID_COLUMN].drop_duplicates().to_numpy()
    if max_cases >= len(all_ids):
        return df
    chosen = rng.choice(all_ids, size=max_cases, replace=False)
    return df.loc[df[config.CASE_ID_COLUMN].isin(chosen)].copy()


def _select_models(arg: str | None, registry: dict) -> list[str]:
    if not arg:
        return list(registry.keys())
    names = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [m for m in names if m not in registry]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {list(registry.keys())}")
    return names


def main() -> int:
    registry = build_binary_model_registry()
    args = _parse_args(registry)
    logger = get_logger("binary_search")

    if args.debug:
        args.n_iter = 3
        args.n_splits = 2
        if args.max_cases is None:
            args.max_cases = 80
        logger.warning("DEBUG mode: n_iter=3 n_splits=2 max_cases=%s", args.max_cases)

    models = _select_models(args.models, registry)
    figures_dir = ensure_dir(args.output_dir / "figures")
    tables_dir = ensure_dir(args.output_dir / "tables")

    # 1. Carga
    df = load_binary_modeling_dataset()
    logger.info("Dataset: shape=%s cases=%d", df.shape, df[config.CASE_ID_COLUMN].nunique())

    if args.max_cases is not None:
        df = _subsample_cases(df, args.max_cases, args.random_state)
        logger.info("Tras subsample: shape=%s cases=%d", df.shape, df[config.CASE_ID_COLUMN].nunique())

    # 2. Clasificación de columnas
    cls = classify_binary_features(df)
    numeric = cls["numeric_features"]
    categorical = cls["categorical_features"]
    logger.info("Features: %d num + %d cat", len(numeric), len(categorical))
    logger.info("Excluidas leakage: %s", cls["leakage_excluded"])

    pd.DataFrame({
        "feature": numeric + categorical,
        "kind": ["numeric"] * len(numeric) + ["categorical"] * len(categorical),
    }).to_csv(tables_dir / "binary_feature_list_used.csv", index=False)
    pd.DataFrame({
        "column": cls["leakage_excluded"] + cls["high_cardinality_excluded"] +
                  cls["constant_excluded"] + cls["too_missing_excluded"],
        "reason": (["leakage"] * len(cls["leakage_excluded"])
                   + ["high_cardinality"] * len(cls["high_cardinality_excluded"])
                   + ["constant"] * len(cls["constant_excluded"])
                   + ["too_missing"] * len(cls["too_missing_excluded"])),
    }).to_csv(tables_dir / "binary_excluded_columns_leakage.csv", index=False)

    # 3. Split por case_id
    X_df = df[numeric + categorical]
    y = df[config.BINARY_TARGET_COLUMN].to_numpy()
    groups = df[config.CASE_ID_COLUMN].to_numpy()
    train_idx, test_idx, split_info = make_binary_group_train_test_split_with_coverage(
        X_df, y, groups, test_size=args.test_size,
        random_state=args.random_state, max_attempts=500,
    )
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    overlap = set(groups_train) & set(groups_test)
    assert not overlap, "Fuga de case_id detectada."

    split_summary = pd.DataFrame([
        {"metric": "chosen_seed", "value": split_info["chosen_seed"]},
        {"metric": "n_train_groups", "value": len(split_info["train_groups"])},
        {"metric": "n_test_groups", "value": len(split_info["test_groups"])},
        {"metric": "n_train_rows", "value": int(len(train_idx))},
        {"metric": "n_test_rows", "value": int(len(test_idx))},
        {"metric": "actual_test_fraction", "value": round(split_info["actual_test_fraction"], 4)},
        {"metric": "binary_coverage_ok", "value": split_info["binary_coverage_ok"]},
        {"metric": "binary_classes_in_train", "value": ",".join(split_info["binary_classes_in_train"])},
        {"metric": "binary_classes_in_test", "value": ",".join(split_info["binary_classes_in_test"])},
    ])
    split_summary.to_csv(tables_dir / "binary_train_test_split_summary.csv", index=False)

    sup = pd.DataFrame({
        "class": [config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
        "train": [int((y_train == config.BINARY_NEGATIVE_CLASS).sum()),
                  int((y_train == config.BINARY_POSITIVE_CLASS).sum())],
        "test": [int((y_test == config.BINARY_NEGATIVE_CLASS).sum()),
                 int((y_test == config.BINARY_POSITIVE_CLASS).sum())],
    })
    sup["total"] = sup["train"] + sup["test"]
    sup.to_csv(tables_dir / "binary_class_support_train_test.csv", index=False)

    pd.DataFrame([{
        "n_train_cases": len(split_info["train_groups"]),
        "n_test_cases": len(split_info["test_groups"]),
        "n_overlap_cases": len(overlap),
        "no_overlap": len(overlap) == 0,
    }]).to_csv(tables_dir / "binary_case_overlap_check.csv", index=False)

    logger.info("Split: %s", json.dumps(
        {k: v for k, v in split_info.items() if k not in ("train_groups", "test_groups")},
        indent=2, default=str,
    ))

    # 4. CV interna
    cv, cv_name, n_splits_eff = build_binary_cv_splitter(
        groups_train=groups_train, y_train=y_train, n_splits=args.n_splits,
    )
    logger.info("CV: %s | n_splits_eff=%d", cv_name, n_splits_eff)

    # 5. Búsqueda por modelo
    results: list[dict] = []
    fitted: dict[str, object] = {}
    test_scores: dict[str, np.ndarray | None] = {}

    for name in models:
        if name not in registry:
            logger.warning("Modelo %s no registrado (dependencia opcional faltante). Skip.", name)
            continue
        spec = registry[name]
        logger.info("===== %s =====", name)
        t0 = time.time()
        try:
            res = run_binary_search_for_model(
                spec=spec,
                X_train=X_train, y_train=y_train, groups_train=groups_train,
                numeric_features=numeric, categorical_features=categorical,
                cv=cv, n_iter=args.n_iter,
                random_state=args.random_state, n_jobs=args.n_jobs,
            )
            est = res.best_estimator
            fitted[name] = est

            # Score continuo y predicción default
            y_score_test = get_positive_class_score(est, X_test)
            test_scores[name] = y_score_test
            y_pred_default = est.predict(X_test)

            # Métricas con predict default (threshold implícito = 0.5 / argmax)
            m_default = _binary_metrics(y_test, y_pred_default, y_score=y_score_test)

            # Ajuste de umbral solo cuando hay score continuo y modelo no es dummy
            chosen_thr = None
            thr_method = None
            m_at_threshold = m_default
            if y_score_test is not None and name != "dummy_most_frequent":
                # Selección de umbral SOLO sobre train (predicciones CV cross_val_predict)
                from sklearn.model_selection import cross_val_predict
                try:
                    score_train_cv = cross_val_predict(
                        est, X_train, y_train, cv=cv, groups=groups_train,
                        method="predict_proba" if hasattr(est, "predict_proba") else "decision_function",
                        n_jobs=1,
                    )
                    if score_train_cv.ndim == 2:
                        classes = getattr(est, "classes_", None)
                        if classes is None and hasattr(est, "steps"):
                            classes = est[-1].classes_
                        idx = list(classes).index(config.BINARY_POSITIVE_CLASS) if classes is not None else 1
                        score_train_cv = score_train_cv[:, idx]
                    y_train_bin = (y_train == config.BINARY_POSITIVE_CLASS).astype(int)
                    thr_yj, _ = select_threshold_youden_j(y_train_bin, score_train_cv)
                    thr_f1, _ = select_threshold_max_f1(y_train_bin, score_train_cv)
                    # Selección operativa: Youden J (priorizar balance sensitivity/specificity)
                    chosen_thr = thr_yj
                    thr_method = "youden_j (CV en train)"
                    y_pred_at = np.where(y_score_test >= chosen_thr,
                                         config.BINARY_POSITIVE_CLASS,
                                         config.BINARY_NEGATIVE_CLASS)
                    m_at_threshold = _binary_metrics(y_test, y_pred_at, y_score=y_score_test)
                    res.chosen_threshold = chosen_thr
                    res.threshold_method = thr_method
                    res.y_pred_test = y_pred_at
                    # Guardar tabla detallada de threshold
                    pd.DataFrame([{
                        "model": name,
                        "threshold_youden_j": thr_yj,
                        "threshold_max_f1": thr_f1,
                        "chosen_threshold": chosen_thr,
                        "method_used": thr_method,
                    }]).to_csv(tables_dir / f"binary_threshold_analysis_{name}.csv", index=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Threshold tuning falló para %s: %s", name, exc)
                    res.y_pred_test = y_pred_default
            else:
                res.y_pred_test = y_pred_default

            res.test_metrics = {f"test_{k}": v for k, v in m_at_threshold.items()}
            res.y_score_test = y_score_test

            row = {
                "model": res.model,
                "status": res.status,
                "fit_seconds": res.fit_seconds,
                "n_iter_effective": res.n_iter_effective,
                "best_cv_score_primary": res.best_cv_score_primary,
                **res.cv_metrics,
                **res.test_metrics,
                "best_params_json": json.dumps(res.best_params, default=str),
                "chosen_threshold": res.chosen_threshold,
                "threshold_method": res.threshold_method,
            }
            logger.info("  ok | cv_%s=%.3f | test_balanced_accuracy=%.3f | test_f1_abn=%.3f | fit=%.1fs",
                        PRIMARY_BINARY_SCORING,
                        res.cv_metrics.get(f"cv_{PRIMARY_BINARY_SCORING}", float("nan")),
                        res.test_metrics.get("test_balanced_accuracy", float("nan")),
                        res.test_metrics.get("test_f1_abnormal", float("nan")),
                        res.fit_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.error("ERROR en %s: %s", name, exc)
            row = {
                "model": name,
                "status": "error",
                "fit_seconds": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(row)

    # 6. Tablas comparativas
    results_df = pd.DataFrame(results)
    cv_cols = ["model", "status", "fit_seconds", "n_iter_effective", "best_cv_score_primary"] \
              + [f"cv_{m}" for m in BINARY_SCORING_METRICS]
    test_cols = ["model", "status",
                 "test_balanced_accuracy", "test_f1_abnormal",
                 "test_precision_abnormal", "test_recall_abnormal_sensitivity",
                 "test_specificity_normal", "test_accuracy",
                 "test_roc_auc", "test_average_precision",
                 "test_n_tp", "test_n_fp", "test_n_fn", "test_n_tn",
                 "chosen_threshold", "threshold_method"]
    cv_view = results_df.reindex(columns=[c for c in cv_cols if c in results_df.columns])
    test_view = results_df.reindex(columns=[c for c in test_cols if c in results_df.columns])
    cv_view.to_csv(tables_dir / "binary_model_comparison_cv.csv", index=False)
    test_view.to_csv(tables_dir / "binary_model_comparison_test.csv", index=False)
    results_df.reindex(columns=["model", "status", "best_params_json"]) \
        .to_csv(tables_dir / "binary_best_hyperparameters.csv", index=False)

    logger.info("Comparativa test:\n%s",
                test_view.round(3).to_string(index=False))

    # 7. Mejor modelo
    ok = results_df.loc[results_df["status"] == "ok"].copy()
    winner = None
    if not ok.empty:
        ok = ok.dropna(subset=["test_balanced_accuracy"])
        if not ok.empty:
            winner = ok.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()

    if winner is None:
        logger.warning("Sin ganador válido.")
        pd.DataFrame([{"note": "Sin modelos válidos."}]) \
            .to_csv(tables_dir / "binary_best_model_classification_report.csv", index=False)
    else:
        m_name = winner["model"]
        logger.info("Mejor modelo: %s (test_balanced_accuracy=%.3f)",
                    m_name, winner["test_balanced_accuracy"])
        est = fitted[m_name]
        # Reusar predicción del threshold elegido si existe
        y_pred = next((r for r in results if r["model"] == m_name), None)
        # Necesitamos el y_pred_test final del result; re-evaluar:
        for r in results:
            if r["model"] == m_name and r["status"] == "ok":
                # Reconstruir predicción a partir del threshold elegido
                y_score = test_scores.get(m_name)
                thr = r.get("chosen_threshold")
                if y_score is not None and thr is not None and not np.isnan(thr):
                    y_pred_final = np.where(y_score >= thr,
                                            config.BINARY_POSITIVE_CLASS,
                                            config.BINARY_NEGATIVE_CLASS)
                else:
                    y_pred_final = est.predict(X_test)
                break
        else:
            y_pred_final = est.predict(X_test)

        # Classification report por clase
        rep = classification_report(
            y_test, y_pred_final,
            labels=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
            output_dict=True, zero_division=0,
        )
        rep_df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class_or_avg"})
        rep_df.to_csv(tables_dir / "binary_best_model_classification_report.csv", index=False)

        # Matrices de confusión
        cm_abs = pd.DataFrame(
            confusion_matrix(y_test, y_pred_final,
                             labels=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS]),
            index=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
            columns=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
        )
        cm_abs.reset_index(names="true_class").to_csv(
            tables_dir / "binary_confusion_matrix_absolute.csv", index=False
        )
        cm_norm = pd.DataFrame(
            confusion_matrix(y_test, y_pred_final,
                             labels=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
                             normalize="true"),
            index=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
            columns=[config.BINARY_NEGATIVE_CLASS, config.BINARY_POSITIVE_CLASS],
        )
        cm_norm.reset_index(names="true_class").to_csv(
            tables_dir / "binary_confusion_matrix_normalized.csv", index=False
        )

        _plot_confusion(y_test, y_pred_final,
                        f"Matriz absoluta — {m_name}",
                        figures_dir / "binary_confusion_matrix_absolute.png",
                        normalize=False)
        _plot_confusion(y_test, y_pred_final,
                        f"Matriz normalizada — {m_name}",
                        figures_dir / "binary_confusion_matrix_normalized.png",
                        normalize=True)

        # ROC / PR si hay scores
        y_score = test_scores.get(m_name)
        if y_score is not None:
            y_true_bin = (y_test == config.BINARY_POSITIVE_CLASS).astype(int)
            _plot_roc(y_true_bin, y_score, f"ROC — {m_name}",
                      figures_dir / "binary_roc_curve_best_model.png")
            _plot_pr(y_true_bin, y_score, f"Precision-Recall — {m_name}",
                     figures_dir / "binary_precision_recall_curve_best_model.png")

            # Tabla de análisis de threshold completa (varios puntos)
            ths = np.unique(np.quantile(y_score, np.linspace(0.05, 0.95, 19)))
            thr_rows = []
            for t in ths:
                pred = np.where(y_score >= t,
                                config.BINARY_POSITIVE_CLASS, config.BINARY_NEGATIVE_CLASS)
                m = _binary_metrics(y_test, pred, y_score=y_score)
                thr_rows.append({
                    "threshold": float(t),
                    "balanced_accuracy": m["balanced_accuracy"],
                    "f1_abnormal": m["f1_abnormal"],
                    "precision_abnormal": m["precision_abnormal"],
                    "recall_abnormal_sensitivity": m["recall_abnormal_sensitivity"],
                    "specificity_normal": m["specificity_normal"],
                })
            pd.DataFrame(thr_rows).to_csv(
                tables_dir / "binary_threshold_analysis.csv", index=False
            )

        # Feature importance
        fi = _extract_feature_importance(est)
        if fi is not None:
            fi.to_csv(tables_dir / "binary_feature_importance_best_model.csv", index=False)
            _plot_feature_importance_top(fi, figures_dir / "binary_feature_importance_top20.png", top=20)
        else:
            pd.DataFrame([{"note": f"No se pudo extraer feature importance para {m_name}."}]) \
                .to_csv(tables_dir / "binary_feature_importance_best_model.csv", index=False)
            logger.warning("No se pudo extraer feature importance para %s.", m_name)

    # 8. Meta JSON
    meta = {
        "dataset_shape": list(df.shape),
        "n_cases": int(df[config.CASE_ID_COLUMN].nunique()),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "leakage_excluded": cls["leakage_excluded"],
        "high_cardinality_excluded": cls["high_cardinality_excluded"],
        "constant_excluded": cls["constant_excluded"],
        "too_missing_excluded": cls["too_missing_excluded"],
        "split_info": split_info,
        "cv": {"splitter": cv_name, "n_splits_effective": n_splits_eff},
        "models_results": results,
        "winner": winner,
        "args": {
            "n_iter": args.n_iter, "n_splits": args.n_splits, "test_size": args.test_size,
            "random_state": args.random_state, "n_jobs": args.n_jobs,
            "max_cases": args.max_cases, "debug": args.debug, "models": models,
        },
    }
    with open(tables_dir / "binary_hyperparameter_search_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Outputs en %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
