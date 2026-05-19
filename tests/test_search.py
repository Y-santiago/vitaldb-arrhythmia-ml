"""Tests para `src.search` y el split robusto en `src.modeling`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.modeling import (
    assert_no_forbidden_features,
    make_train_test_group_split_with_coverage,
)
from src.search import (
    MODEL_REGISTRY,
    NON_FEATURE_METADATA_COLUMNS,
    PRIMARY_SCORING,
    SCORING_METRICS,
    build_cv_splitter,
    load_feature_dataset,
    prepare_dataset_for_modeling,
)


# ---------------------------------------------------------------------------
# Fixtures sintéticas
# ---------------------------------------------------------------------------
def _toy_features_df(n_per_case: int = 20,
                     case_ids=(1, 2, 3, 4, 5),
                     classes=("N", "A", "B"),
                     seed: int = 7) -> pd.DataFrame:
    """Construye un DataFrame de features sintético con todas las columnas que
    el pipeline real espera (metadata + features numéricas + columnas prohibidas)."""
    rng = np.random.RandomState(seed)
    rows = []
    for cid in case_ids:
        for i in range(n_per_case):
            cls = classes[(cid + i) % len(classes)]
            row = {
                config.CASE_ID_COLUMN: cid,
                config.TARGET_COLUMN: cls,
                config.BEAT_TYPE_COLUMN: "N",     # presente pero PROHIBIDO
                config.SIGNAL_QUALITY_COLUMN: False,
                config.BEAT_TIME_COLUMN: float(i) * 0.8,
                "beat_index": i,
                "start_sample": i * 600,
                "end_sample": i * 600 + 1000,
                "window_seconds": 2.0,
                # features "reales"
                "mean": rng.randn(),
                "std": abs(rng.randn()),
                "rr_prev": rng.uniform(0.6, 1.2),
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# prepare_dataset_for_modeling: bloqueo de columnas prohibidas
# ---------------------------------------------------------------------------
def test_prepare_dataset_excludes_forbidden_and_metadata():
    df = _toy_features_df()
    X, y, groups, feature_cols = prepare_dataset_for_modeling(df)

    # Las columnas prohibidas NO deben aparecer como features.
    for forbidden in config.FORBIDDEN_FEATURE_COLUMNS:
        assert forbidden not in feature_cols, f"{forbidden} debería estar excluida"
    # Los metadatos por ventana tampoco.
    for meta in NON_FEATURE_METADATA_COLUMNS:
        assert meta not in feature_cols
    # Las features reales sí.
    assert "mean" in feature_cols
    assert "std" in feature_cols
    assert "rr_prev" in feature_cols
    assert X.shape == (len(df), len(feature_cols))
    assert len(y) == len(df)
    assert len(groups) == len(df)


def test_prepare_dataset_raises_when_beat_type_is_added_to_features():
    """Verificación dura: si por error pasara una columna prohibida, debe abortar."""
    with pytest.raises(ValueError, match="beat_type"):
        assert_no_forbidden_features(["mean", "std", config.BEAT_TYPE_COLUMN])


def test_prepare_dataset_raises_on_missing_target():
    df = _toy_features_df().drop(columns=[config.TARGET_COLUMN])
    with pytest.raises(KeyError):
        prepare_dataset_for_modeling(df)


# ---------------------------------------------------------------------------
# load_feature_dataset: limpieza de etiquetas inválidas
# ---------------------------------------------------------------------------
def test_load_feature_dataset_drops_nan_and_string_nan(tmp_path):
    df = _toy_features_df()
    # Inyectar etiquetas inválidas
    df.loc[0, config.TARGET_COLUMN] = np.nan
    df.loc[1, config.TARGET_COLUMN] = "nan"
    df.loc[2, config.TARGET_COLUMN] = "  "
    df.loc[3, config.TARGET_COLUMN] = "NONE"
    p = tmp_path / "feats.parquet"
    df.to_parquet(p, index=False)

    cleaned = load_feature_dataset(p)
    assert len(cleaned) == len(df) - 4
    assert cleaned[config.TARGET_COLUMN].notna().all()
    assert (
        ~cleaned[config.TARGET_COLUMN].astype(str).str.strip().str.lower()
        .isin({"nan", "none", ""})
    ).all()


# ---------------------------------------------------------------------------
# make_train_test_group_split_with_coverage
# ---------------------------------------------------------------------------
def test_split_with_coverage_no_group_leak():
    df = _toy_features_df(n_per_case=15, case_ids=tuple(range(1, 8)))
    X, y, groups, _ = prepare_dataset_for_modeling(df)
    tr, te, info = make_train_test_group_split_with_coverage(X, y, groups, test_size=0.25)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert info["chosen_seed"] is not None
    assert info["n_total_classes"] == len(set(y.tolist()))


def test_split_with_coverage_reports_deterministic():
    df = _toy_features_df(n_per_case=15, case_ids=tuple(range(1, 8)))
    X, y, groups, _ = prepare_dataset_for_modeling(df)
    tr1, te1, info1 = make_train_test_group_split_with_coverage(X, y, groups, random_state=42)
    tr2, te2, info2 = make_train_test_group_split_with_coverage(X, y, groups, random_state=42)
    np.testing.assert_array_equal(tr1, tr2)
    np.testing.assert_array_equal(te1, te2)
    assert info1["chosen_seed"] == info2["chosen_seed"]


def test_split_with_coverage_test_size_close_to_target_when_many_groups():
    """Con suficientes grupos, la fracción real debe acercarse al objetivo."""
    df = _toy_features_df(n_per_case=10, case_ids=tuple(range(1, 21)))  # 20 grupos
    X, y, groups, _ = prepare_dataset_for_modeling(df)
    _, _, info = make_train_test_group_split_with_coverage(X, y, groups, test_size=0.2)
    assert abs(info["actual_test_fraction"] - 0.2) < 0.1


def test_split_with_coverage_maximizes_class_overlap():
    """Si existe un split donde todas las clases están en ambos lados, debe encontrarlo."""
    df = _toy_features_df(n_per_case=10, case_ids=tuple(range(1, 11)))  # 10 grupos
    X, y, groups, _ = prepare_dataset_for_modeling(df)
    _, _, info = make_train_test_group_split_with_coverage(X, y, groups, max_attempts=50)
    assert info["n_classes_covered"] == info["n_total_classes"]
    assert info["classes_only_in_train"] == []
    assert info["classes_only_in_test"] == []


def test_split_with_coverage_reports_missing_classes_when_only_few_groups():
    """Con 2 grupos y clases concentradas, debe documentar honestamente la pérdida."""
    rng = np.random.RandomState(0)
    rows = []
    # Caso 1 tiene solo clase 'A'; caso 2 tiene solo clase 'B'.
    for i in range(20):
        rows.append({
            config.CASE_ID_COLUMN: 1,
            config.TARGET_COLUMN: "A",
            config.BEAT_TYPE_COLUMN: "N",
            config.SIGNAL_QUALITY_COLUMN: False,
            config.BEAT_TIME_COLUMN: float(i),
            "beat_index": i, "start_sample": 0, "end_sample": 1,
            "window_seconds": 2.0,
            "feat": rng.randn(),
        })
    for i in range(20):
        rows.append({
            config.CASE_ID_COLUMN: 2,
            config.TARGET_COLUMN: "B",
            config.BEAT_TYPE_COLUMN: "N",
            config.SIGNAL_QUALITY_COLUMN: False,
            config.BEAT_TIME_COLUMN: float(i),
            "beat_index": i, "start_sample": 0, "end_sample": 1,
            "window_seconds": 2.0,
            "feat": rng.randn(),
        })
    df = pd.DataFrame(rows)
    X, y, groups, _ = prepare_dataset_for_modeling(df)
    _, _, info = make_train_test_group_split_with_coverage(X, y, groups, max_attempts=50)
    # Es estructuralmente imposible cubrir A y B en ambos lados con solo 2 grupos.
    assert info["n_classes_covered"] == 0
    # Una de las dos clases debe quedar solo en uno de los lados.
    assert info["classes_only_in_train"] or info["classes_only_in_test"]


# ---------------------------------------------------------------------------
# build_cv_splitter
# ---------------------------------------------------------------------------
def test_build_cv_splitter_returns_object_and_name():
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    y = np.array(["A", "B", "A", "B", "A", "B", "A", "B"])
    cv, name, n_splits = build_cv_splitter(groups, y, n_splits=3)
    assert n_splits == 3
    assert name in {"StratifiedGroupKFold", "GroupKFold"}
    # El iterador debe producir folds disjuntos.
    folds = list(cv.split(np.zeros((len(y), 1)), y, groups=groups))
    assert len(folds) == 3
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_build_cv_splitter_clamps_n_splits_to_n_groups():
    groups = np.array([1, 1, 2, 2])  # solo 2 grupos
    y = np.array(["A", "B", "A", "B"])
    _, _, n_splits = build_cv_splitter(groups, y, n_splits=5)
    assert n_splits == 2


# ---------------------------------------------------------------------------
# Estructura del MODEL_REGISTRY y métricas
# ---------------------------------------------------------------------------
def test_model_registry_has_required_models():
    required = {"logreg", "decision_tree", "random_forest", "xgboost", "linear_svc", "mlp"}
    assert required.issubset(set(MODEL_REGISTRY.keys()))


def test_scoring_metrics_contains_primary():
    assert PRIMARY_SCORING in SCORING_METRICS
    for required in ("f1_macro", "precision_macro", "recall_macro",
                     "accuracy", "balanced_accuracy", "f1_weighted"):
        assert required in SCORING_METRICS


def test_model_specs_can_be_instantiated():
    """Todos los modelos del registro deben producir un pipeline ejecutable."""
    for name, spec in MODEL_REGISTRY.items():
        pipe = spec.pipeline_factory()
        assert pipe is not None, f"Factory de {name} retornó None"
        # Steps del pipeline deben terminar en 'clf'
        steps_names = [s for s, _ in pipe.steps]
        assert steps_names[-1] == "clf", f"Último step de {name} no es 'clf'"
