# Estado del proyecto y siguiente instrucción (handoff)

**Fecha:** 2026-05-27
**Rama:** `main`
**Iteración actual:** pivot a modelado tabular (sin ECG crudo).

Documento dirigido a que ChatGPT revise el estado real y dé la siguiente
instrucción técnica sin tener que adivinar nada.

---

## 1. Estado exacto del repositorio

### 1.1 Flujo activo (tabular)
- `scripts/01_audit_filtered_tabular_dataset.py` — audita anotaciones +
  metadata y produce 5 CSVs descriptivos en `reports/tables/`.
- `scripts/02_build_filtered_tabular_modeling_dataset.py` — construye
  `data/processed/filtered_tabular_modeling_dataset.parquet` (639 460
  filas × 85 columnas, 482 cases, 10 clases).
- `scripts/03_run_tabular_hyperparameter_search.py` — CLI de
  `RandomizedSearchCV` multi-modelo con CV por grupo. Genera todos los
  CSVs/PNGs requeridos.
- `src/preprocessing.py::build_tabular_preprocessor` — `ColumnTransformer`
  con imputación + escalado para numéricas y imputación + OHE con
  `handle_unknown="ignore"` para categóricas.
- `src/modeling.py::make_train_test_group_split_with_coverage` — split
  80/20 por `case_id` con búsqueda de cobertura de clases.
- `src/tabular_search.py` — orquestación: `classify_features`,
  `build_pipeline_for_model`, `MODEL_PARAM_DISTRIBUTIONS`,
  `run_search_for_model`, `evaluate_on_test`, `extract_feature_importance`.
- `notebooks/06_tabular_modeling_hyperparameter_search.ipynb` — wrapper
  interactivo del mismo flujo.

### 1.2 Línea legacy (ECG crudo) — pausada
Marcados con banner `[LEGACY — ...]` en su docstring:
- `scripts/01_download_all_available_ecg.py`
- `scripts/02_build_features_all_windows.py`
- `scripts/03_run_hyperparameter_search.py`
- `src/search.py`
- Notebooks `03`, `04`, `05`, `06_full_modeling_hyperparameter_search.ipynb`
- `reports/MODELING_REPORT.md` (informe de la corrida ECG previa)

NO se ha eliminado nada de este flujo: queda como referencia histórica.

### 1.3 Datos en disco (no versionados)
- `data/raw/physionet_annotations/Annotation_Files/` — 482 archivos.
- `data/raw/physionet_annotations/metadata.csv` — 482 × 79.
- `data/raw/vitaldb_waveforms/` — 3 `.npy` legacy (case_1001, case_1002,
  case_1018). No se usan en la fase tabular.
- `data/processed/filtered_tabular_modeling_dataset.parquet` — 639 460 ×
  85, generado por `scripts/02_build_filtered_tabular_modeling_dataset.py`.

---

## 2. Archivos nuevos en esta iteración

### Nuevos
- `scripts/01_audit_filtered_tabular_dataset.py`
- `scripts/02_build_filtered_tabular_modeling_dataset.py`
- `scripts/03_run_tabular_hyperparameter_search.py`
- `src/tabular_search.py`
- `notebooks/06_tabular_modeling_hyperparameter_search.ipynb`
- `tests/test_tabular_search.py`
- `reports/TABULAR_MODELING_REPORT.md`
- `reports/NEXT_STEPS_FOR_CHATGPT.md` (este archivo)

### Modificados
- `src/config.py` — añade `TABULAR_LEAKAGE_COLUMNS`,
  `TABULAR_MAX_CATEGORY_CARDINALITY`, `TABULAR_OHE_MIN_FREQUENCY`,
  `TABULAR_DATASET_FILENAME`.
- `src/preprocessing.py` — añade `build_tabular_preprocessor`.
- `src/search.py` — banner `[LEGACY]`.
- `scripts/01_download_all_available_ecg.py` — banner `[LEGACY]`.
- `scripts/02_build_features_all_windows.py` — banner `[LEGACY]`.
- `scripts/03_run_hyperparameter_search.py` — banner `[LEGACY]`.
- `README.md` — pivot metodológico, estructura actualizada, flujo
  recomendado tabular.

---

## 3. Comandos ejecutados (reproducibilidad)

```bash
# Tests (deben pasar)
python -m pytest tests/

# Pipeline tabular:
python scripts/01_audit_filtered_tabular_dataset.py
python scripts/02_build_filtered_tabular_modeling_dataset.py

# Búsqueda de hiperparámetros — debug (≈ 5-10 min):
python scripts/03_run_tabular_hyperparameter_search.py --debug

# Full run (puede tardar horas con n_iter=30 y los 6 modelos):
python scripts/03_run_tabular_hyperparameter_search.py --n-iter 30 --n-splits 5
```

---

## 4. Resultados principales

### 4.1 Auditoría del dataset (reproducible y firme)
Fuente: `reports/tables/tabular_dataset_audit.csv`.

| métrica | valor |
|---|---:|
| filas antes filtros | 676 250 |
| filas después filtros | 639 460 |
| cases (antes y después) | 482 |
| clases `rhythm_label` | 10 |
| features numéricas candidatas | 54 |
| features categóricas candidatas | 17 |

Clases dominantes: `N` (61 %), `AFIB/AFL` (25 %). Minoritarias críticas:
`AVB` (10 cases), `Unclassifiable` (5 cases).

### 4.2 Split (sobre dataset completo, `random_state=42`)
- `chosen_seed=42`, 10/10 clases cubiertas en train y test.
- `train`: 385 cases / 510 287 filas.
- `test`: 97 cases / 129 173 filas.
- `actual_test_fraction = 0.202`.

### 4.3 Modelos — corrida `--debug` ejecutada (60 cases, n_iter=3, n_splits=2)

Resultados reales del run documentado en
`reports/tables/tabular_hyperparameter_search_meta.json`:

| modelo | test_f1_macro | test_bal_acc | test_accuracy | fit_seconds |
|---|---:|---:|---:|---:|
| logreg | 0.151 | 0.272 | 0.412 | 82.8 |
| decision_tree | 0.078 | 0.112 | 0.166 | 12.9 |
| random_forest | 0.144 | 0.157 | 0.566 | 57.7 |
| xgboost | 0.085 | 0.113 | 0.560 | 97.4 |
| **linear_svc** | **0.189** | **0.356** | 0.354 | 149.6 |
| mlp | 0.126 | 0.126 | 0.344 | 83.7 |

Total wall-clock: ~9 minutos.

### 4.4 Mejor modelo (debug)
`linear_svc` con `clf__C ≈ 56.7` y `class_weight="balanced"`.

Por clase en test (debug, 16 085 filas en test):

| clase | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| AFIB/AFL | 0.405 | 0.843 | 0.547 | 4 070 |
| SND | 0.445 | 0.999 | 0.616 | 699 |
| N | 0.665 | 0.141 | 0.233 | 9 075 |
| Patterned Ventricular Ectopy | 0.371 | 0.211 | 0.269 | 1 295 |
| AVB, Patterned Atrial Ectopy, WAP/MAT | 0 | 0 | 0 | 465+362+15 |
| SVTA | 0.006 | 1.000 | 0.012 | 19 |
| VT | 1.000 | 0.012 | 0.023 | 85 |

Detalle completo en `reports/tables/tabular_best_model_classification_report.csv` y
`reports/tables/tabular_confusion_matrix_absolute.csv`. La matriz
muestra que LinearSVC con `class_weight="balanced"` está sobre-prediciendo
SVTA, sacrificando precision a cambio de recall.

---

## 5. Problemas encontrados y soluciones aplicadas

| Problema | Causa raíz | Solución |
|---|---|---|
| `caseid` (97 % NaN) en el merge | Un archivo (Annotation_file_2453) tiene la columna `caseid` en lugar de `bad_signal_quality_label`. | Añadido a `TABULAR_LEAKAGE_COLUMNS`. |
| `age` venía como string | Algunas filas usan `>89` para anonimización. | Coerción `>89 → 89` en `02_build_*`. |
| XGBoost ≥ 2.0 rechaza folds con clases no consecutivas | `LabelEncoder` global crea gaps cuando un fold pierde clases. | Wrapper `_XGBClassifierSafe` (re-encoding por fit). |
| MLP + early stopping crashea con etiquetas string | `np.isnan(y_pred)` sobre strings. | `early_stopping=False`. |
| Constantes ocultas tras filtros (`bad_signal_quality`, `caseid`, `casestart`, `airway`) | Tras aplicar filtros una columna puede quedar con un único valor. | Drop automático en `02_build_*` y detección en `classify_features`. |
| Debug `n_jobs=1` demasiado lento | Modelos RF/MLP no paralelos. | `n_jobs=-1` también en debug. |

---

## 6. Preguntas técnicas pendientes

1. **¿Aumentar `--n-iter` para full run, o ya es suficiente con 30?** Con 6
   modelos × 30 iter × 5 folds = 900 fits sobre 510k filas, el costo es
   significativo. ¿Vale 50? ¿100?
2. **¿Encoding de `dx` / `opname`?** Hoy descartadas por alta cardinalidad.
   ¿Probar target encoding por fold? ¿Embeddings de texto preentrenados
   (off-scope clínico)?
3. **¿Persistir el `best_estimator_` con `joblib`?** Hoy se vuelve a
   ajustar en cada corrida; persistirlo aceleraría inferencia.
4. **¿Manejo de desbalance más allá de `class_weight`?** ¿Probar
   `imbalanced-learn` (SMOTE, BalancedRandomForest, BalancedBagging)?
5. **¿Reintroducir el flujo ECG crudo como complemento?** Ahora que el
   tabular está estable, ¿vale la pena reactivar las features ECG para
   un ensemble?
6. **¿Reportar varianza entre folds además del promedio?**

---

## 7. Recomendación concreta para la siguiente instrucción

**Path A — Full run sobre los 482 cases**

```
Ejecuta:
  python scripts/03_run_tabular_hyperparameter_search.py --n-iter 30 --n-splits 5

Tiempo estimado: varias horas según hardware. Revisa al final
reports/tables/tabular_model_comparison_test.csv y actualiza
reports/TABULAR_MODELING_REPORT.md con las cifras del full run
sustituyendo las del debug.
```

**Path B — Mejora del feature engineering antes del full run**

```
Añade a scripts/02_build_filtered_tabular_modeling_dataset.py:
  * mean/std/rmssd de RR sobre ventana móvil de N latidos por caso
    (con N parametrizable; default N=20)
  * codificación numérica de `dx` / `opname` por target encoding
    estimado SOLO en train tras el split externo (fit fuera del
    pipeline interno para evitar fuga)
Luego corre nuevamente la búsqueda en debug y full.
```

**Path C — Persistir el modelo y construir un script de inferencia**

```
Tras el full run, persiste el mejor estimator con joblib.dump en
models/best_tabular.joblib (ignorado por git). Crea un script
scripts/04_predict_tabular.py que cargue ese modelo y prediga sobre
un nuevo parquet con las mismas columnas.
```

### Mi recomendación inicial

**Path A** primero (es la corrida limpia que esta iteración prometió y
todavía no se ha hecho), seguido de **Path C** si Path A da un baseline
razonable. **Path B** queda como mejora futura de feature engineering.

---

## Apéndice — Comandos de verificación

```bash
# Tests
python -m pytest tests/ -q

# Verificar que el dataset no contiene columnas prohibidas como features
python -c "
import pandas as pd
from src.tabular_search import classify_features
df = pd.read_parquet('data/processed/filtered_tabular_modeling_dataset.parquet')
cls = classify_features(df)
forbidden = {'beat_type','rhythm_label','case_id','rhythm_classes','bad_signal_quality','bad_signal_quality_label','subjectid','death_inhosp','icu_days','adm','dis'}
for c in forbidden:
    assert c not in cls['numeric_features'] and c not in cls['categorical_features'], c
print('OK: ninguna columna prohibida en features.')
"

# Inspeccionar resultados de la última corrida
python -c "
import pandas as pd
print(pd.read_csv('reports/tables/tabular_model_comparison_test.csv').round(3).to_string(index=False))
"
```
