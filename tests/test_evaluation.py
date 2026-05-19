"""Tests para `src.evaluation` (helpers de soporte por split y matriz con totales)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import (
    class_support_per_split,
    classes_missing_in_train,
    compute_macro_metrics,
    confusion_matrix_with_totals,
    per_class_report,
)


def test_class_support_per_split_basic_counts():
    y_train = np.array(["N", "N", "A", "B", "B"])
    y_test = np.array(["N", "A", "A", "C"])
    df = class_support_per_split(y_train, y_test)

    assert set(df.index) >= {"A", "B", "C", "N", "TOTAL"}
    assert df.loc["N", "train"] == 2
    assert df.loc["N", "test"] == 1
    assert df.loc["A", "train"] == 1
    assert df.loc["A", "test"] == 2
    assert df.loc["B", "train"] == 2
    assert df.loc["B", "test"] == 0  # ausente en test, no NaN
    assert df.loc["C", "train"] == 0  # ausente en train
    assert df.loc["C", "test"] == 1
    assert df.loc["TOTAL", "train"] == len(y_train)
    assert df.loc["TOTAL", "test"] == len(y_test)


def test_classes_missing_in_train_returns_only_test_only():
    y_train = np.array(["N", "A"])
    y_test = np.array(["N", "B", "C"])
    assert classes_missing_in_train(y_train, y_test) == ["B", "C"]


def test_classes_missing_in_train_returns_empty_when_all_present():
    y_train = np.array(["N", "A", "B"])
    y_test = np.array(["N", "A"])
    assert classes_missing_in_train(y_train, y_test) == []


def test_confusion_matrix_with_totals_layout():
    y_true = np.array(["A", "A", "B", "B", "C"])
    y_pred = np.array(["A", "B", "B", "B", "A"])
    cm = confusion_matrix_with_totals(y_true, y_pred)

    # Esquina interior: clases reales x clases predichas
    assert cm.loc["A", "A"] == 1
    assert cm.loc["A", "B"] == 1
    assert cm.loc["B", "B"] == 2
    assert cm.loc["C", "A"] == 1

    # Márgenes
    assert cm.loc["A", "support_true"] == 2
    assert cm.loc["B", "support_true"] == 2
    assert cm.loc["C", "support_true"] == 1
    assert cm.loc["predicted_total", "A"] == 2
    assert cm.loc["predicted_total", "B"] == 3


def test_per_class_report_includes_support():
    y_true = np.array(["A", "A", "B", "B", "B"])
    y_pred = np.array(["A", "B", "B", "B", "B"])
    rep = per_class_report(y_true, y_pred)
    assert "support" in rep.columns
    assert rep.loc["A", "support"] == 2
    assert rep.loc["B", "support"] == 3


def test_compute_macro_metrics_keys():
    y_true = np.array(["A", "B", "A", "B"])
    y_pred = np.array(["A", "B", "B", "B"])
    m = compute_macro_metrics(y_true, y_pred)
    assert set(m.keys()) == {"f1_macro", "recall_macro", "precision_macro", "balanced_accuracy"}
    for v in m.values():
        assert 0.0 <= v <= 1.0
