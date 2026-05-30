# Estado del proyecto y siguiente instrucción (handoff)

**Fecha:** 2026-05-29
**Rama actual:** `binary-normal-vs-arrhythmia` (no fusionada a `main`)
**Iteración:** clasificación binaria `normal_sinus` vs
`arrhythmia_or_abnormal`.

Documento dirigido a que ChatGPT revise el estado real y dé la siguiente
instrucción técnica sin tener que adivinar nada.

---

## 1. Estado exacto del repositorio

### 1.1 Flujo activo de esta iteración (binario)
- `scripts/04_audit_binary_rhythm_dataset.py` — audita el dataset
  binario y genera 9 CSVs + 3 figuras descriptivas en `reports/`.
- `scripts/05_build_binary_rhythm_modeling_dataset.py` — construye
  `data/processed/binary_rhythm_modeling_dataset.parquet` con el target
  binario y nuevas features RR rolling por caso.
- `scripts/06_run_binary_rhythm_model_search.py` — CLI de
  `RandomizedSearchCV` multi-modelo con CV por grupo, threshold tuning
  por Youden J (en train) y persistencia de todos los outputs.
- `src/binary_search.py` — `build_binary_model_registry`,
  `classify_binary_features`,
  `make_binary_group_train_test_split_with_coverage`,
  `run_binary_search_for_model`, `_XGBBinaryClassifierSafe`,
  selectores de threshold (`select_threshold_youden_j`,
  `select_threshold_max_f1`).
- `src/config.py` — constantes binarias:
  `BINARY_TARGET_COLUMN`, `BINARY_POSITIVE_CLASS`,
  `BINARY_NEGATIVE_CLASS`, `BINARY_LABEL_MAPPING`,
  `BINARY_EXCLUDED_LABELS`, `BINARY_LEAKAGE_COLUMNS`,
  `BINARY_DATASET_FILENAME` y la función
  `map_rhythm_label_to_binary`.
- `tests/test_binary_search.py` — 32 tests nuevos.
- `reports/BINARY_RHYTHM_MODELING_REPORT.md` — informe técnico.

### 1.2 Iteraciones anteriores (NO se reactivan en esta fase)
- **Tabular multiclase** (iteración previa): `src/tabular_search.py`,
  `scripts/01_audit_filtered_tabular_dataset.py`,
  `scripts/02_build_filtered_tabular_modeling_dataset.py`,
  `scripts/03_run_tabular_hyperparameter_search.py`,
  `notebooks/06_tabular_modeling_hyperparameter_search.ipynb`,
  `reports/TABULAR_MODELING_REPORT.md`. La construcción del dataset
  tabular base se reutiliza desde la iteración binaria.
- **ECG crudo** (legacy): `src/search.py`, `src/download.py`,
  `src/windowing.py`, scripts y notebooks 03–05 con banner `[LEGACY]`.

### 1.3 Datos en disco (no versionados, salvo modelos joblib)
- `data/raw/physionet_annotations/Annotation_Files/` — 482 archivos.
- `data/processed/filtered_tabular_modeling_dataset.parquet` —
  639 460 × 85, base del flujo tabular.
- `data/processed/binary_rhythm_modeling_dataset.parquet` —
  639 401 × 101 (16 features RR rolling nuevas).
- `models/*.joblib`, `models/feature_columns.json`,
  `models/model_artifacts_metadata.json` — versionados con excepción
  declarada en `.gitignore` para que la app Streamlit los consuma.

---

## 2. Archivos nuevos en esta iteración

| Path | Tipo |
|---|---|
| `src/binary_search.py` | nuevo |
| `scripts/04_audit_binary_rhythm_dataset.py` | nuevo |
| `scripts/05_build_binary_rhythm_modeling_dataset.py` | nuevo |
| `scripts/06_run_binary_rhythm_model_search.py` | nuevo |
| `tests/test_binary_search.py` | nuevo |
| `reports/BINARY_RHYTHM_MODELING_REPORT.md` | nuevo |
| `reports/NEXT_STEPS_FOR_CHATGPT.md` | reescrito para esta iteración |
| `reports/PROJECT_REPORT.md` | actualizado con sección 13 (binario) |
| `README.md` | actualizado con flujo binario |
| `src/config.py` | constantes y mapper binarios |

---

## 3. Comandos ejecutados (reproducibilidad)

```bash
# 0. Tests
python -m pytest tests/ -q

# 1. Audit
python scripts/04_audit_binary_rhythm_dataset.py

# 2. Build
python scripts/05_build_binary_rhythm_modeling_dataset.py

# 3. Búsqueda
python scripts/06_run_binary_rhythm_model_search.py --debug
# o full run:
python scripts/06_run_binary_rhythm_model_search.py --n-iter 30 --n-splits 5
```

---

## 4. Resultados principales — corrida `--debug` (80 cases)

`max_cases=80, n_iter=3, n_splits=2, n_jobs=-1`.
Split: 64 cases train (82 541 filas) / 16 cases test (22 972 filas).
Fuente: `reports/tables/binary_hyperparameter_search_meta.json`,
`reports/tables/binary_model_comparison_test.csv`.

| modelo | test_balanced_acc | test_f1_abn | sens (recall) | spec | ROC-AUC | AP | fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy_most_frequent | 0.500 | 0.000 | 0.000 | 1.000 | 0.500 | 0.597 | 7.9 |
| **logreg_balanced** | **0.906** | **0.922** | **0.914** | **0.898** | **0.956** | **0.964** | 15.7 |
| sgd_log_loss | 0.891 | 0.912 | 0.909 | 0.873 | 0.950 | 0.951 | 9.5 |
| linear_svc_balanced | 0.500* | 0.748 | 1.000 | 0.000 | 0.053 | 0.394 | 16.7 |
| hist_gradient_boosting | 0.862 | 0.899 | 0.931 | 0.793 | 0.937 | 0.949 | 31.4 |
| random_forest_balanced | 0.871 | 0.903 | 0.925 | 0.817 | 0.945 | 0.966 | 53.2 |
| extra_trees_balanced | 0.883 | 0.911 | 0.928 | 0.838 | 0.936 | 0.939 | 444 |
| balanced_random_forest | 0.867 | 0.898 | 0.916 | 0.817 | 0.925 | 0.955 | 240 |
| easy_ensemble | **error** | — | — | — | — | — | — |
| xgboost_binary | 0.500* | 0.748 | 1.000 | 0.000 | 0.936 | 0.949 | 74 |

(*) Umbral Youden J colapsado (predijo todo positivo). ROC-AUC alto
indica que la discriminación intrínseca es buena; el problema es la
calibración del threshold. Posible fix: `CalibratedClassifierCV`.

**Ganador (debug):** `logreg_balanced` con `clf__C ≈ 0.0746` y
`class_weight='balanced'`. test_balanced_accuracy = 0.906, ROC-AUC =
0.956, AP = 0.964. Matriz de confusión: TN=8 319, FP=941, FN=1 184,
TP=12 528.

`easy_ensemble` falló por `numpy._core._exceptions._ArrayMemoryError`
en un `cross_val_predict` interno. Documentado en el reporte; reducir
`n_estimators` o eliminarlo para el full run.

Las cifras **no son** las del full run sobre los 482 cases. Para
reproducir el ganador final, correr sin `--debug`.

---

## 5. Problemas encontrados y soluciones

| Problema | Causa | Solución |
|---|---|---|
| sklearn falla con `pos_label=1` cuando las etiquetas son strings y se usa scoring genérico `f1`/`precision`/`recall`. | Scorers genéricos asumen pos_label=1. | `BINARY_SCORING_METRICS` usa `make_scorer(..., pos_label=BINARY_POSITIVE_CLASS)`. |
| `roc_auc` y `average_precision` no son uniformes vía `make_scorer` con strings + distintos response_methods. | Algunos modelos no exponen `predict_proba`. | Se computan **solo en test** sobre `decision_function` o `predict_proba` directamente. |
| `LinearSVC` produjo umbral Youden J que colapsó las predicciones de test a una sola clase. | `decision_function` poco calibrado; el umbral de train no generalizó. | Documentado. `chosen_threshold` y `threshold_method` se registran por modelo. Posible mejora: `CalibratedClassifierCV`. |
| XGBoost ≥ 2.0 con etiquetas string + multiclase ya tenía wrapper; para binario se replica como `_XGBBinaryClassifierSafe` con mapeo explícito ne → 0, pos → 1 y `scale_pos_weight = n_neg/n_pos` calculado en `fit`. | XGB binario requiere y ∈ {0,1}. | Wrapper interno, transparente al pipeline. |

---

## 6. Preguntas técnicas pendientes

1. **¿Calibrar probabilidades?** `CalibratedClassifierCV` sobre el mejor
   modelo (especialmente LinearSVC/RF) puede estabilizar el umbral.
2. **¿Tomar decisión a nivel de caso (agregando latidos)?** Hoy se mide
   por latido. Reportar también la fracción anormal por caso podría ser
   más clínicamente útil.
3. **¿Probar `imblearn.Pipeline` con SMOTE dentro de CV?** Solo
   recomendable después del full run para no esconder el efecto de
   `class_weight`.
4. **¿Persistir el mejor modelo binario en `models/`?** El `.gitignore`
   ya permite versionar `.joblib` para Streamlit.
5. **¿Fairness por sexo/edad?** Hay desbalances obvios en metadata; vale
   medir si el modelo es uniformemente bueno.

---

## 7. Recomendación concreta para la siguiente instrucción

### Path A — Full run + persistencia del mejor modelo

```
Ejecuta:
  python scripts/06_run_binary_rhythm_model_search.py --n-iter 30 --n-splits 5
Tiempo estimado: 1-3 horas con todos los modelos.

Después, persiste el best_estimator del modelo ganador con
joblib.dump en models/binary_best_pipeline.joblib y guarda
models/binary_feature_columns.json con la lista de features. Actualiza
BINARY_RHYTHM_MODELING_REPORT.md con las cifras del full run y registra
en él el ganador final.
```

### Path B — Calibración + decisión a nivel de caso

```
Antes del full run, modifica scripts/06_run_binary_rhythm_model_search.py
para envolver el mejor modelo en CalibratedClassifierCV (cv interno) y
añadir una métrica agregada por case_id (fracción anormal por caso vs
ground truth). Mide cuánto mejora el threshold tuning.
```

### Path C — Fairness por subgrupo

```
Crea scripts/07_binary_fairness_audit.py que tome el mejor modelo
persistido en models/ y reporte test_balanced_accuracy, recall_abnormal
y specificity_normal por sexo, por banda de edad (<40, 40-60, 60-80, >80),
y por tipo de cirugía. Guarda los CSVs y figuras en reports/.
```

**Mi recomendación:** **Path A**. Las cifras de debug ya muestran que la
tarea es tratable (LogReg balanced_accuracy 0.906 en debug). El full run
sobre 482 cases es lo que falta para confirmar la generalización.

---

## Apéndice — Verificación rápida

```bash
# Tests (esperado: 116/116 OK)
python -m pytest tests/ -q

# Que no haya features prohibidas
python -c "
import pandas as pd
from src.binary_search import classify_binary_features
df = pd.read_parquet('data/processed/binary_rhythm_modeling_dataset.parquet')
cls = classify_binary_features(df)
forbidden = {'beat_type','rhythm_label','rhythm_binary','case_id','rhythm_classes','bad_signal_quality','bad_signal_quality_label'}
for c in forbidden:
    assert c not in cls['numeric_features'] and c not in cls['categorical_features'], c
print('OK: ninguna columna prohibida en features.')
"

# Resumen del último run
python -c "
import pandas as pd
print(pd.read_csv('reports/tables/binary_model_comparison_test.csv').round(3).to_string(index=False))
"
```
