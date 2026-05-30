"""Generate Model B evaluation reports from training artifacts.

The model evaluation itself is performed once inside `train_model_b.py` after
hyperparameter search. This script reads those saved artifacts and writes the
human-readable Markdown reports.

Usage:
    python model_b_pipeline/evaluate_model_b.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_b_pipeline import config_model_b as cfg  # noqa: E402
from model_b_pipeline.utils_model_b import ensure_dir  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _markdown_table(df: pd.DataFrame | None, columns: list[str] | None = None, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_No disponible en esta corrida._"
    table = df.copy()
    if columns is not None:
        table = table[[col for col in columns if col in table.columns]]
    table = table.head(max_rows)
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def _metric(metadata: dict, name: str) -> str:
    value = (metadata.get("test_metrics") or {}).get(name)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summarize_previous_broad_model() -> str:
    broad_test = _read_csv(cfg.PROJECT_ROOT / "reports" / "tables" / "binary_model_comparison_test.csv")
    broad_features = _read_csv(cfg.PROJECT_ROOT / "reports" / "tables" / "binary_feature_list_used.csv")
    if broad_test is None:
        return (
            "No se encontro `reports/tables/binary_model_comparison_test.csv`; "
            "la comparacion numerica contra el modelo amplio queda pendiente."
        )
    if "balanced_accuracy" in broad_test.columns:
        broad_test = broad_test.sort_values("balanced_accuracy", ascending=False)
    feature_count = len(broad_features) if broad_features is not None else "NA"
    cols = [
        "model",
        "balanced_accuracy",
        "accuracy",
        "precision_abnormal",
        "recall_abnormal_sensitivity",
        "specificity_normal",
        "f1_abnormal",
        "roc_auc",
        "average_precision",
    ]
    return (
        f"Existe una corrida previa del modelo binario amplio con {feature_count} "
        "features candidatas registradas. Mejor fila disponible:\n\n"
        f"{_markdown_table(broad_test, cols, max_rows=1)}\n\n"
        "La comparacion debe leerse con cuidado si aquella corrida fue `--debug` "
        "o uso un split distinto."
    )


def _write_model_report(metadata: dict) -> Path:
    audit = _read_csv(cfg.TABLES_DIR / "model_b_dataset_audit.csv")
    class_rows = _read_csv(cfg.TABLES_DIR / "model_b_class_distribution_rows.csv")
    class_cases = _read_csv(cfg.TABLES_DIR / "model_b_class_distribution_cases.csv")
    case_composition = _read_csv(cfg.TABLES_DIR / "model_b_case_composition.csv")
    split = _read_csv(cfg.TABLES_DIR / "model_b_train_test_split_summary.csv")
    support = _read_csv(cfg.TABLES_DIR / "model_b_class_support_train_test.csv")
    cv = _read_csv(cfg.TABLES_DIR / "model_b_model_comparison_cv.csv")
    test = _read_csv(cfg.TABLES_DIR / "model_b_model_comparison_test.csv")
    params = _read_csv(cfg.TABLES_DIR / "model_b_best_hyperparameters.csv")
    cm_abs = _read_csv(cfg.TABLES_DIR / "model_b_confusion_matrix_absolute.csv")
    threshold = _read_csv(cfg.TABLES_DIR / "model_b_threshold_analysis.csv")

    winner = metadata.get("winning_model", "NA")
    report = f"""# Model B Report - Binary Rhythm Classification

## 1. Objetivo

Crear un flujo independiente para clasificar `normal_sinus` vs `arrhythmia_or_abnormal` usando `rhythm_binary` como target, con split 80/20 por `case_id` y sin reactivar ECG crudo ni multiclase.

## 2. Justificacion del Modelo B

Modelo B reduce el pipeline binario amplio a 25 variables originales: dinamica RR/local, contexto clinico basal y cinco laboratorios preoperatorios. La meta es un modelo mas limpio, defendible e interpretable, no maximizar metricas con variables administrativas o intraoperatorias.

## 3. Variables Usadas

Se usaron exactamente estas 25 variables:

`{", ".join(cfg.FEATURES_MODEL_B)}`

Agrupos generales: RR/temporales, clinicas basicas y laboratorios preoperatorios seleccionados.

## 4. Variables Excluidas

No se usaron `case_id`, `rhythm_binary`, `rhythm_label`, `beat_type`, identificadores, variables administrativas, desenlaces hospitalarios, texto diagnostico/procedimiento ni variables intraoperatorias. `case_id` solo se uso para el split por grupo.

## 5. Auditoria del Dataset

{_markdown_table(audit, max_rows=20)}

Distribucion por filas:

{_markdown_table(class_rows)}

Distribucion por casos:

{_markdown_table(class_cases)}

Composicion de casos:

{_markdown_table(case_composition)}

## 6. Split por `case_id`

{_markdown_table(split, max_rows=20)}

Soporte por clase:

{_markdown_table(support)}

El archivo `model_b_case_overlap_check.csv` confirma ausencia de overlap entre train y test.

## 7. Modelos Evaluados

Modelos principales: `dummy_most_frequent`, `logreg_balanced`, `sgd_log_loss`, `hist_gradient_boosting`. `random_forest_balanced` queda disponible con `--include-random-forest`.

## 8. Hiperparametros

Busqueda con `RandomizedSearchCV`, CV interna por grupo (`StratifiedGroupKFold` con fallback a `GroupKFold`), `groups = case_id`, y `refit = balanced_accuracy`.

{_markdown_table(params, max_rows=30)}

## 9. Metricas CV

{_markdown_table(cv, [
    "model",
    "best_cv_balanced_accuracy",
    "cv_mean_accuracy",
    "cv_mean_precision_abnormal",
    "cv_mean_recall_abnormal_sensitivity",
    "cv_mean_specificity_normal",
    "cv_mean_f1_abnormal",
    "elapsed_seconds",
], max_rows=20)}

Modelo ganador por CV: `{winner}`.

## 10. Metricas Test

La seleccion del modelo ganador se hizo por CV en train. El test se evaluo despues.

{_markdown_table(test, [
    "model",
    "prediction_rule",
    "balanced_accuracy",
    "accuracy",
    "precision_abnormal",
    "recall_abnormal_sensitivity",
    "specificity_normal",
    "f1_abnormal",
    "roc_auc",
    "average_precision",
], max_rows=20)}

Metricas finales del pipeline persistido con el umbral elegido:

| metric | value |
|---|---:|
| balanced_accuracy | {_metric(metadata, "balanced_accuracy")} |
| accuracy | {_metric(metadata, "accuracy")} |
| precision_abnormal | {_metric(metadata, "precision_abnormal")} |
| recall_abnormal_sensitivity | {_metric(metadata, "recall_abnormal_sensitivity")} |
| specificity_normal | {_metric(metadata, "specificity_normal")} |
| f1_abnormal | {_metric(metadata, "f1_abnormal")} |
| roc_auc | {_metric(metadata, "roc_auc")} |
| average_precision | {_metric(metadata, "average_precision")} |

## 11. Matriz de Confusion

{_markdown_table(cm_abs)}

Figuras:

- `reports/model_b/figures/model_b_confusion_matrix_absolute.png`
- `reports/model_b/figures/model_b_confusion_matrix_normalized.png`

## 12. Threshold Usado

Umbral final:

```json
{json.dumps(metadata.get("threshold_used", {}), indent=2)}
```

Analisis de umbrales guardado:

{_markdown_table(threshold, max_rows=20)}

## 13. Comparacion contra Modelo Amplio Anterior

{_summarize_previous_broad_model()}

## 14. Limitaciones

- La unidad de analisis son filas/latidos, mientras que el split evita fuga por caso.
- Muchos casos son mixtos, por lo que el contexto clinico puede repetirse con etiquetas distintas dentro del mismo caso.
- El umbral se eligio en train; cualquier ajuste operativo posterior debe validarse en un conjunto externo o una nueva particion.
- No es un modelo clinico listo para uso asistencial.

## 15. Recomendaciones

- Ejecutar una corrida full si esta salida fue `--debug`.
- Revisar estabilidad por subgrupos y por composicion de caso.
- Comparar contra el binario amplio usando el mismo split si se requiere una comparacion estricta.
- Mantener Modelo B sin variables intraoperatorias/administrativas salvo analisis secundarios desactivados por defecto.
"""
    path = cfg.REPORTS_MODEL_B_DIR / "MODEL_B_REPORT.md"
    ensure_dir(path.parent)
    path.write_text(report, encoding="utf-8")
    return path


def _write_next_steps(metadata: dict) -> Path:
    debug = metadata.get("debug")
    winner = metadata.get("winning_model", "NA")
    next_steps = f"""# Next Steps - Model B

## Que se creo

- Carpeta independiente `model_b_pipeline/` con configuracion, auditoria, builder, entrenamiento, evaluacion, utilidades y tests.
- Dataset reducido local `data/processed/model_b_dataset.parquet` con `case_id`, `rhythm_binary` y las 25 features de Modelo B.
- Reportes locales en `reports/model_b/`.
- Artefactos locales en `models/model_b/`: pipeline `.joblib`, columnas y metadata.

## Scripts para correr

```bash
python model_b_pipeline/audit_model_b_dataset.py
python model_b_pipeline/build_model_b_dataset.py
python model_b_pipeline/train_model_b.py --debug
python model_b_pipeline/evaluate_model_b.py
```

Corrida full inicial:

```bash
python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5
python model_b_pipeline/evaluate_model_b.py
```

## Resultados de esta corrida

- Debug: `{debug}`
- Modelo ganador por CV: `{winner}`
- Balanced accuracy test final: `{_metric(metadata, "balanced_accuracy")}`
- Sensitivity anormal: `{_metric(metadata, "recall_abnormal_sensitivity")}`
- Specificity normal: `{_metric(metadata, "specificity_normal")}`
- F1 anormal: `{_metric(metadata, "f1_abnormal")}`

## Que fallo

Si este archivo existe, `evaluate_model_b.py` pudo leer la metadata de entrenamiento. Revisar la salida de consola o los tests si alguna tabla/figura esperada falta.

## Que falta

- Correr full run si esta corrida fue debug.
- Revisar si el umbral Youden J es operativo o si conviene fijar sensibilidad minima.
- Comparar Modelo B contra el modelo amplio anterior con el mismo split si se necesita una conclusion metodologica fuerte.

## Recomendacion para la siguiente iteracion

Mantener Modelo B como baseline interpretable de 25 variables. La siguiente mejora deberia enfocarse en validacion y calibracion, no en agregar variables de leakage o intraoperatorias al modelo principal.
"""
    path = cfg.REPORTS_MODEL_B_DIR / "NEXT_STEPS_MODEL_B.md"
    ensure_dir(path.parent)
    path.write_text(next_steps, encoding="utf-8")
    return path


def run_evaluation_report() -> tuple[Path, Path]:
    metadata_path = cfg.MODELS_MODEL_B_DIR / "model_b_metadata.json"
    metadata = _read_json(metadata_path)
    if metadata is None:
        raise FileNotFoundError(
            "Missing Model B metadata. Run "
            "python model_b_pipeline/train_model_b.py --debug first."
        )
    return _write_model_report(metadata), _write_next_steps(metadata)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    report_path, next_steps_path = run_evaluation_report()
    print(f"Wrote {report_path}")
    print(f"Wrote {next_steps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
