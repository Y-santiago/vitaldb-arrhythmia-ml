"""Utilidades transversales del proyecto."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y la variable de entorno PYTHONHASHSEED.

    Parameters
    ----------
    seed : int
        Valor de semilla a aplicar.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "vitaldb_arrhythmia_ml",
               level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger configurado con formato uniforme.

    Idempotente: no añade handlers duplicados si se llama varias veces.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Crea el directorio `path` (y los padres) si no existe. Devuelve el `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(directory: str | os.PathLike[str],
               patterns: Iterable[str] = ("*",)) -> list[Path]:
    """Lista archivos del directorio que coincidan con alguno de los patrones glob.

    No es recursivo. Si el directorio no existe devuelve una lista vacía.
    """
    base = Path(directory)
    if not base.exists():
        return []
    matched: list[Path] = []
    for pattern in patterns:
        matched.extend(sorted(base.glob(pattern)))
    return matched
