"""Métricas y reportes de evaluación.

Funciones de soporte para reporte por clase, métricas macro y matriz de
confusión. **No** se calculan ni se imprimen resultados hasta que se invocan
con datos reales.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_macro_metrics(y_true: np.ndarray | pd.Series,
                          y_pred: np.ndarray | pd.Series
                          ) -> dict[str, float]:
    """Calcula métricas macro y balanced accuracy.

    Returns
    -------
    dict[str, float]
        ``f1_macro``, ``recall_macro``, ``precision_macro``,
        ``balanced_accuracy``.
    """
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def per_class_report(y_true: np.ndarray | pd.Series,
                     y_pred: np.ndarray | pd.Series,
                     labels: Iterable[str] | None = None
                     ) -> pd.DataFrame:
    """Reporte por clase (precision, recall, f1, support) como DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_df(y_true: np.ndarray | pd.Series,
                        y_pred: np.ndarray | pd.Series,
                        labels: Iterable[str] | None = None,
                        normalize: str | None = None) -> pd.DataFrame:
    """Devuelve la matriz de confusión como DataFrame indexado por clase.

    Parameters
    ----------
    normalize : {None, "true", "pred", "all"}
        Igual que en :func:`sklearn.metrics.confusion_matrix`.
    """
    label_list = list(labels) if labels is not None else sorted(
        set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique())
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list, normalize=normalize)
    return pd.DataFrame(cm, index=label_list, columns=label_list)
