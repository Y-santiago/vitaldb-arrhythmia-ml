"""Métricas y reportes de evaluación.

Funciones de soporte para reporte por clase, métricas macro, tablas de
soporte por split y matriz de confusión con totales marginales. **No** se
calculan ni se imprimen resultados hasta que se invocan con datos reales.
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


# ---------------------------------------------------------------------------
# Métricas macro
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Soporte por clase y por split
# ---------------------------------------------------------------------------
def class_support_per_split(y_train: np.ndarray | pd.Series,
                            y_test: np.ndarray | pd.Series
                            ) -> pd.DataFrame:
    """Tabla con el conteo de muestras por clase en train y test.

    La unión de etiquetas se ordena para facilitar la lectura. Cualquier
    clase ausente en uno de los dos conjuntos aparece con ``0`` en lugar de
    ``NaN`` para evitar ambigüedad.

    Returns
    -------
    pandas.DataFrame
        Columnas: ``train``, ``test``, ``total``. Index: etiquetas.
        Última fila ``TOTAL`` con la suma por columna.
    """
    s_train = pd.Series(y_train).value_counts()
    s_test = pd.Series(y_test).value_counts()
    labels = sorted(set(s_train.index) | set(s_test.index), key=lambda v: str(v))

    df = pd.DataFrame(
        {
            "train": [int(s_train.get(lbl, 0)) for lbl in labels],
            "test": [int(s_test.get(lbl, 0)) for lbl in labels],
        },
        index=labels,
    )
    df["total"] = df["train"] + df["test"]
    df.loc["TOTAL"] = df.sum(axis=0)
    return df


def classes_missing_in_train(y_train: np.ndarray | pd.Series,
                             y_test: np.ndarray | pd.Series
                             ) -> list:
    """Etiquetas que aparecen en test pero no en train.

    Un clasificador no puede aprender estas clases: su recall será 0 por
    construcción y arrastrarán la métrica macro hacia abajo. El notebook
    debe mostrarlas explícitamente para evitar interpretaciones equivocadas.
    """
    train_set = set(pd.Series(y_train).unique())
    test_set = set(pd.Series(y_test).unique())
    return sorted(test_set - train_set, key=lambda v: str(v))


# ---------------------------------------------------------------------------
# Reporte por clase
# ---------------------------------------------------------------------------
def per_class_report(y_true: np.ndarray | pd.Series,
                     y_pred: np.ndarray | pd.Series,
                     labels: Iterable[str] | None = None
                     ) -> pd.DataFrame:
    """Reporte por clase (precision, recall, f1, support) como DataFrame.

    La columna ``support`` indica cuántas muestras reales tiene cada clase
    en ``y_true``. Si una clase tiene ``support=0`` su F1 no es interpretable.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


# ---------------------------------------------------------------------------
# Matrices de confusión
# ---------------------------------------------------------------------------
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
        set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique()),
        key=lambda v: str(v),
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list, normalize=normalize)
    return pd.DataFrame(cm, index=label_list, columns=label_list)


def confusion_matrix_with_totals(y_true: np.ndarray | pd.Series,
                                 y_pred: np.ndarray | pd.Series,
                                 labels: Iterable[str] | None = None
                                 ) -> pd.DataFrame:
    """Matriz de confusión en conteos absolutos + márgenes.

    Estructura del DataFrame devuelto:

        * Filas = clases reales (``y_true``).
        * Columnas = clases predichas (``y_pred``).
        * Columna extra ``support_true``: total de muestras reales por clase
          (suma de la fila).
        * Fila extra ``predicted_total``: total de predicciones por clase
          (suma de la columna).

    Esta tabla es más fácil de auditar que una matriz normalizada cuando hay
    clases con muy pocas muestras o ausentes en train.
    """
    label_list = list(labels) if labels is not None else sorted(
        set(pd.Series(y_true).unique()) | set(pd.Series(y_pred).unique()),
        key=lambda v: str(v),
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    df = pd.DataFrame(cm, index=label_list, columns=label_list).astype(int)
    df["support_true"] = df.sum(axis=1)
    df.loc["predicted_total"] = df.sum(axis=0)
    return df
