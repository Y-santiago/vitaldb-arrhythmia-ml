"""Generate human-readable Model B reports from saved training artifacts.

`train_model_b.py` performs model selection and the single final test
evaluation. This script reads the generated tables and JSON artifacts, then
refreshes Markdown reports for humans and app builders.

Usage:
    python model_b_pipeline/evaluate_model_b.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_b_pipeline import config_model_b as cfg  # noqa: E402
from model_b_pipeline.train_model_b import build_model_b_registry  # noqa: E402
from model_b_pipeline.utils_model_b import ensure_dir  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any] | None:
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


def _metric(metadata: dict[str, Any], name: str) -> str:
    value = (metadata.get("test_metrics") or {}).get(name)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _registry_search_space_table() -> pd.DataFrame:
    registry = build_model_b_registry(include_random_forest=True, include_xgboost=True)
    rows = []
    for model_name, spec in registry.items():
        params = spec["params"]
        rows.append({
            "model": model_name,
            "description": spec.get("description", ""),
            "hyperparameters": "; ".join(f"{key}={value}" for key, value in params.items()),
        })
    return pd.DataFrame(rows)


def _summarize_previous_broad_model() -> str:
    broad_test = _read_csv(cfg.PROJECT_ROOT / "reports" / "tables" / "binary_model_comparison_test.csv")
    broad_features = _read_csv(cfg.PROJECT_ROOT / "reports" / "tables" / "binary_feature_list_used.csv")
    if broad_test is None:
        return (
            "No se encontro `reports/tables/binary_model_comparison_test.csv`; "
            "la comparacion numerica contra el modelo amplio queda pendiente."
        )
    feature_count = len(broad_features) if broad_features is not None else "NA"
    sort_col = "balanced_accuracy" if "balanced_accuracy" in broad_test.columns else None
    if sort_col:
        broad_test = broad_test.sort_values(sort_col, ascending=False)
    return (
        f"Existe una corrida previa del modelo binario amplio con {feature_count} "
        "features candidatas registradas. Mejor fila disponible:\n\n"
        f"{_markdown_table(broad_test, max_rows=1)}\n\n"
        "La comparacion debe leerse con cuidado si aquella corrida fue `--debug` "
        "o uso un split distinto. Para una comparacion estricta, repetir ambos "
        "flujos con el mismo split."
    )


def _artifact_line(metadata: dict[str, Any], key: str, fallback: str) -> str:
    return str((metadata.get("artifacts") or {}).get(key, fallback)).replace("\\", "/")


def _write_model_report(metadata: dict[str, Any], threshold: dict[str, Any] | None) -> Path:
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
    threshold_table = _read_csv(cfg.TABLES_DIR / "model_b_threshold_analysis.csv")
    failures = _read_csv(cfg.TABLES_DIR / "model_b_model_failures.csv")
    registry_space = _registry_search_space_table()

    winner = metadata.get("winning_model", "NA")
    debug = metadata.get("debug", "NA")
    selection_metric = metadata.get("selection_metric", "balanced_accuracy")
    pipeline_path = _artifact_line(metadata, "pipeline", "models/model_b/model_b_best_pipeline.joblib")
    feature_path = _artifact_line(metadata, "feature_columns", "models/model_b/model_b_feature_columns.json")
    metadata_path = _artifact_line(metadata, "metadata", "models/model_b/model_b_metadata.json")
    threshold_path = _artifact_line(metadata, "threshold", "models/model_b/model_b_threshold.json")

    report = f"""# Model B Report - Binary Rhythm Classification

## Resumen Ejecutivo

- Corrida debug: `{debug}`
- Modelos solicitados: `{", ".join(metadata.get("models_requested", []))}`
- Modelos exitosos: `{", ".join(metadata.get("models_succeeded", []))}`
- Modelo ganador por CV: `{winner}`
- Metrica de seleccion: `{selection_metric}`
- Valor CV de seleccion: `{metadata.get("selection_metric_cv_value", "NA")}`
- Test evaluado una sola vez al final para el ganador.

## 1. Objetivo

Crear un flujo independiente para clasificar `normal_sinus` vs `arrhythmia_or_abnormal` usando `rhythm_binary`, con split 80/20 por `case_id`, sin ECG crudo, sin multiclase y sin variables administrativas/intraoperatorias como predictores.

## 2. Que Archivo Correr

Orden recomendado:

```bash
python model_b_pipeline/audit_model_b_dataset.py
python model_b_pipeline/build_model_b_dataset.py
python model_b_pipeline/train_model_b.py --debug
python model_b_pipeline/evaluate_model_b.py
```

Para corrida full ligera:

```bash
python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5 --models logreg_balanced sgd_log_loss hist_gradient_boosting
python model_b_pipeline/evaluate_model_b.py
```

Tambien se puede correr desde VS Code con las configuraciones en `.vscode/launch.json`.

## 3. Este Flujo Hace Busqueda de Hiperparametros?

Si. `train_model_b.py` ejecuta `RandomizedSearchCV` por cada modelo candidato usando validacion cruzada por `case_id`. El ganador se selecciona por el promedio de `{selection_metric}` en CV sobre train. El test no se usa para seleccionar modelo, hiperparametros, threshold ni semilla; se evalua una sola vez al final para el pipeline ganador.

## 4. Variables Usadas

Se usaron exactamente estas 25 variables originales:

`{", ".join(cfg.FEATURES_MODEL_B)}`

`case_id` se usa solo para split/CV por grupo, nunca como predictor.

## 5. Variables Excluidas

No se usaron `case_id`, `rhythm_binary`, `rhythm_label`, `beat_type`, identificadores, variables administrativas, desenlaces hospitalarios, texto diagnostico/procedimiento ni variables intraoperatorias.

## 6. Auditoria del Dataset

{_markdown_table(audit, max_rows=20)}

Distribucion por filas:

{_markdown_table(class_rows)}

Distribucion por casos:

{_markdown_table(class_cases)}

Composicion de casos:

{_markdown_table(case_composition)}

## 7. Split por `case_id`

{_markdown_table(split, max_rows=20)}

Soporte por clase:

{_markdown_table(support)}

## 8. Modelos Comparados y Espacio de Busqueda

{_markdown_table(registry_space, max_rows=10)}

`random_forest_balanced` solo corre con `--include-random-forest`. `xgboost_binary` solo corre con `--include-xgboost`.

## 9. Mejores Hiperparametros Encontrados

{_markdown_table(params, max_rows=40)}

## 10. Metricas CV

{_markdown_table(cv, max_rows=20)}

## 11. Metricas Test del Ganador

La tabla de test contiene el ganador seleccionado por CV. No es una tabla para elegir modelo.

{_markdown_table(test, max_rows=5)}

Metricas finales del pipeline persistido:

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

## 12. Matriz de Confusion

{_markdown_table(cm_abs)}

Figuras:

- `reports/model_b/figures/model_b_confusion_matrix_absolute.png`
- `reports/model_b/figures/model_b_confusion_matrix_normalized.png`

## 13. Threshold Usado

Archivo: `{threshold_path}`

```json
{json.dumps(threshold or metadata.get("threshold_used", {}), indent=2)}
```

Analisis de thresholds sobre train:

{_markdown_table(threshold_table, max_rows=20)}

## 14. Donde Quedo el Modelo para la App?

- Pipeline completo: `{pipeline_path}`
- Columnas requeridas: `{feature_path}`
- Metadata: `{metadata_path}`
- Threshold operativo: `{threshold_path}`

La app debe cargar el `.joblib`; ese pipeline ya incluye imputacion, escalado, one-hot encoding y clasificador. La app no debe reconstruir preprocesamiento por separado.

Ejemplo:

```python
from model_b_pipeline.predict_model_b import predict_model_b_dataframe

predicciones = predict_model_b_dataframe(df_entrada)
```

Ver tambien `reports/model_b/MODEL_B_APP_USAGE.md`.

## 15. Modelos Fallidos

{_markdown_table(failures, max_rows=20)}

## 16. Comparacion contra Modelo Amplio Anterior

{_summarize_previous_broad_model()}

## 17. Limitaciones

- Esta salida puede ser debug si `debug=True`; no interpretar como rendimiento final.
- La unidad de analisis son filas/latidos, mientras que el split evita fuga por caso.
- Muchos casos son mixtos, por lo que variables clinicas se repiten con etiquetas distintas dentro del mismo caso.
- El umbral se selecciona sobre train. Cualquier ajuste operativo posterior requiere validacion externa o nueva particion.
- No es un modelo clinico listo para uso asistencial.

## 18. Recomendaciones

- Ejecutar full run si esta corrida fue debug.
- Mantener Modelo B con sus 25 variables para conservar interpretabilidad.
- Comparar contra el modelo amplio usando el mismo split si se necesita una conclusion estricta.
"""
    path = cfg.REPORTS_MODEL_B_DIR / "MODEL_B_REPORT.md"
    ensure_dir(path.parent)
    path.write_text(report, encoding="utf-8")
    return path


def _write_next_steps(metadata: dict[str, Any]) -> Path:
    next_steps = f"""# Next Steps - Model B

## Que se creo o actualizo

- Registro explicito de modelos en `build_model_b_registry()`.
- Busqueda de hiperparametros por modelo con `RandomizedSearchCV`.
- Seleccion del ganador por CV en train, usando `{metadata.get("selection_metric", "balanced_accuracy")}`.
- Evaluacion de test una sola vez para el ganador.
- Artefactos para app en `models/model_b/`.
- Utilidad de inferencia en `model_b_pipeline/predict_model_b.py`.
- Configuraciones de VS Code en `.vscode/launch.json`.

## Como correr desde cero

```bash
python model_b_pipeline/audit_model_b_dataset.py
python model_b_pipeline/build_model_b_dataset.py
python model_b_pipeline/train_model_b.py --debug
python model_b_pipeline/evaluate_model_b.py
```

## Como correr desde VS Code

Abrir Run and Debug y elegir:

1. `Model B - Audit`
2. `Model B - Build dataset`
3. `Model B - Train DEBUG`
4. `Model B - Evaluate reports`

Para una corrida mas seria, usar `Model B - Train FULL light`.

## Resultado de esta corrida

- Debug: `{metadata.get("debug")}`
- Modelos exitosos: `{", ".join(metadata.get("models_succeeded", []))}`
- Modelo ganador por CV: `{metadata.get("winning_model")}`
- Balanced accuracy test final: `{_metric(metadata, "balanced_accuracy")}`
- Sensitivity anormal: `{_metric(metadata, "recall_abnormal_sensitivity")}`
- Specificity normal: `{_metric(metadata, "specificity_normal")}`
- F1 anormal: `{_metric(metadata, "f1_abnormal")}`

## Artefactos para app

- `models/model_b/model_b_best_pipeline.joblib`
- `models/model_b/model_b_feature_columns.json`
- `models/model_b/model_b_metadata.json`
- `models/model_b/model_b_threshold.json`

## Que falta

- Ejecutar full run si esta corrida fue debug.
- Revisar calibracion y estabilidad por subgrupo.
- Decidir si el threshold `youden_j_train` es adecuado para el objetivo operativo o si se requiere sensibilidad minima.

## Si algo falla

Revisar:

- `reports/model_b/tables/model_b_model_failures.csv`
- `reports/model_b/tables/model_b_cv_results_all.csv`
- salida de consola de `train_model_b.py`
"""
    path = cfg.REPORTS_MODEL_B_DIR / "NEXT_STEPS_MODEL_B.md"
    ensure_dir(path.parent)
    path.write_text(next_steps, encoding="utf-8")
    return path


def _write_app_usage() -> Path:
    usage = """# Model B App Usage

## Archivos que usa la app

Despues de entrenar, la app debe leer:

- `models/model_b/model_b_best_pipeline.joblib`
- `models/model_b/model_b_feature_columns.json`
- `models/model_b/model_b_metadata.json`
- `models/model_b/model_b_threshold.json`

El `.joblib` ya contiene preprocesamiento completo y clasificador. No reconstruyas imputadores, escaladores ni one-hot encoders en la app.

## Ejemplo minimo

```python
import pandas as pd
from model_b_pipeline.predict_model_b import predict_model_b_dataframe

df = pd.read_parquet("data/processed/model_b_dataset.parquet")
pred = predict_model_b_dataframe(df)
print(pred.head())
```

## Columnas de entrada requeridas

El dataframe de entrada debe contener estas 25 columnas:

```python
[
    "rr_prev", "rr_next", "hr_inst_from_rr_prev", "position_in_case",
    "rr_prev_rolling_mean_5", "rr_prev_rolling_std_5",
    "rr_prev_rolling_mean_20", "rr_prev_rolling_std_20",
    "rr_rmssd_5", "rr_rmssd_20", "rr_pnn50_5", "rr_pnn50_20",
    "local_hr_mean_5", "local_hr_mean_20", "age", "sex", "bmi",
    "asa", "preop_htn", "preop_dm", "preop_hb", "preop_na",
    "preop_k", "preop_gluc", "preop_cr",
]
```

Puede tener columnas extra; la utilidad las ignora.

## Salida

`predict_model_b_dataframe(df)` devuelve:

- `model_b_score_arrhythmia_or_abnormal`
- `model_b_prediction`
- `model_b_prediction_default_threshold`
- `model_b_threshold_used`
- `model_b_threshold_method`

Si faltan columnas, lanza un `ValueError` con la lista exacta de columnas faltantes.
"""
    path = cfg.REPORTS_MODEL_B_DIR / "MODEL_B_APP_USAGE.md"
    ensure_dir(path.parent)
    path.write_text(usage, encoding="utf-8")
    return path


def run_evaluation_report() -> tuple[Path, Path, Path]:
    metadata_path = cfg.MODELS_MODEL_B_DIR / "model_b_metadata.json"
    threshold_path = cfg.MODELS_MODEL_B_DIR / "model_b_threshold.json"
    metadata = _read_json(metadata_path)
    if metadata is None:
        raise FileNotFoundError(
            "Missing Model B metadata. Run "
            "python model_b_pipeline/train_model_b.py --debug first."
        )
    threshold = _read_json(threshold_path)
    return _write_model_report(metadata, threshold), _write_next_steps(metadata), _write_app_usage()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    report_path, next_steps_path, app_usage_path = run_evaluation_report()
    print(f"Wrote {report_path}")
    print(f"Wrote {next_steps_path}")
    print(f"Wrote {app_usage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
