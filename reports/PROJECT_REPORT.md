# Informe del proyecto — VitalDB Arrhythmia ML

**Fecha del informe:** 2026-05-27
**Rama:** `main`
**Commits relevantes hasta este informe:** `f5140f2` → `6657569` → `5d296e8` →
`3b22cab` → (commit de la iteración tabular).

---

## 0. Pivot metodológico (iteración tabular, 2026-05-27)

A partir de esta iteración el proyecto pasa de “clasificación desde
segmentos de ECG crudo” a “**clasificación multiclase de `rhythm_label`
usando datos tabulares filtrados de anotaciones y metadatos del dataset
VitalDB Arrhythmia**”. El nuevo flujo:

- usa los 482 archivos de anotaciones y `metadata.csv` ya en disco;
- **no** descarga señal ECG desde VitalDB;
- **no** construye ventanas de señal;
- une anotaciones + metadata por `case_id`;
- añade features temporales locales por latido (`rr_prev`, `rr_next`,
  `hr_inst_from_rr_prev`, `position_in_case`);
- aplica `ColumnTransformer(Imputer+Scaler / Imputer+OneHotEncoder)`;
- corre `RandomizedSearchCV` con CV por grupo sobre 6 modelos.

La línea ECG cruda anterior (notebooks `03–05`, `06_full_...ipynb`,
módulos `src/search.py`, scripts ECG) queda marcada como `legacy` con
banner explícito en cada archivo. No se elimina; se preserva como
referencia histórica.

Reporte específico de la fase tabular: `reports/TABULAR_MODELING_REPORT.md`.
Reporte específico de la fase legacy ECG: `reports/MODELING_REPORT.md`.
Estado para handoff: `reports/NEXT_STEPS_FOR_CHATGPT.md`.

---

## 1. Resumen ejecutivo

Proyecto académico y exploratorio de machine learning para la clasificación
multiclase de `rhythm_label` sobre datos tabulares filtrados del dataset
*VitalDB Arrhythmia Database 1.0.0* (PhysioNet).

A la fecha del informe el repositorio tiene:

- Pipeline de auditoría + construcción + búsqueda de hiperparámetros
  reproducible vía scripts CLI.
- Notebook interactivo equivalente (`notebooks/06_tabular_modeling_hyperparameter_search.ipynb`).
- Cobertura de tests sobre split por grupo, bloqueo de columnas
  prohibidas, preprocesamiento tabular y estructura de tablas de
  resultados (84 tests verdes).
- Reportes y CSVs/PNGs persistidos en `reports/` para cada corrida.

No se reportan métricas “finales”: la búsqueda se ejecuta tanto en modo
`--debug` (subset de cases) como en modo completo; los CSVs reflejan la
corrida más reciente. Cualquier comparación debe consultar el
`tabular_hyperparameter_search_meta.json` para saber qué `n_iter`,
`n_splits` y `max_cases` se usaron.

---

## 2. Contexto y problema

- **Tarea:** clasificación multiclase supervisada de `rhythm_label`.
- **Unidad de análisis:** ventana temporal centrada en cada latido anotado.
- **Restricción crítica:** `beat_type` no se utiliza como predictor en ningún
  experimento; solo se conserva para análisis descriptivo.
- **Exclusiones:** registros con `bad_signal_quality` y la clase `Noise`.
- **Validación:** split estricto por `case_id` (`GroupKFold` /
  `GroupShuffleSplit`). Nunca aleatorio por ventana o latido.

---

## 3. Estructura del repositorio

```
vitaldb-arrhythmia-ml/
├── README.md
├── LICENSE                       # MIT genérico, sin nombres
├── .gitignore                    # protege data/, models/, *.csv, *.parquet, etc.
├── requirements.txt
├── environment.yml               # Python 3.11
├── pyproject.toml
├── data/
│   ├── raw/
│   │   ├── physionet_annotations/    # paquete PhysioNet (no versionado)
│   │   └── vitaldb_waveforms/        # ECG descargado de VitalDB (no versionado)
│   ├── interim/                       # no versionado
│   └── processed/                     # no versionado
├── notebooks/
│   ├── 01_download_and_structure.ipynb
│   ├── 02_eda_annotations.ipynb
│   ├── 03_ecg_loading_and_visualization.ipynb
│   ├── 04_windowing_and_feature_engineering.ipynb
│   └── 05_baseline_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── download.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── features.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── PROJECT_REPORT.md         # este archivo
├── models/                       # no versionado
└── tests/
    ├── conftest.py
    ├── test_data_loading.py
    ├── test_windowing.py
    ├── test_features.py
    ├── test_modeling.py
    └── test_evaluation.py
```

Las carpetas `data/`, `models/`, `reports/figures/*`, `reports/tables/*` están
protegidas por `.gitignore`. La estructura se conserva en git mediante archivos
`.gitkeep`.

---

## 4. Decisiones metodológicas implementadas

Todas estas reglas están codificadas en `src/config.py` o se hacen cumplir
mediante asserts en `src/modeling.py`.

| Regla | Implementación |
|---|---|
| Target = `rhythm_label` | `config.TARGET_COLUMN = "rhythm_label"` |
| `beat_type` prohibido como predictor | Listado en `FORBIDDEN_FEATURE_COLUMNS`. `assert_no_forbidden_features` aborta en runtime. |
| Excluir `Noise` | `EXCLUDED_RHYTHM_LABELS = ("Noise",)` + `preprocessing.exclude_rhythm_labels`. |
| Excluir `bad_signal_quality` | `preprocessing.exclude_bad_signal_quality` (acepta bool o string). |
| Split por `case_id` | `modeling.make_group_split` / `make_group_kfold` usan `GroupShuffleSplit` / `GroupKFold`. |
| Rutas relativas | `config.PROJECT_ROOT = Path(__file__).resolve().parents[1]` y derivadas. |
| No subir datos al repo | `.gitignore` bloquea `*.csv`, `*.parquet`, `*.pkl`, `*.npy`, `*.h5`, `*.joblib`, `data/raw/*`, `data/interim/*`, `data/processed/*`, `models/*`. Verificado con `git check-ignore`. |

---

## 5. Componentes implementados

### 5.1 Módulos en `src/`

| Módulo | Responsabilidad principal |
|---|---|
| `config.py` | Rutas relativas, nombres de columnas, parámetros por defecto (`DEFAULT_ECG_FS_HZ`, `DEFAULT_WINDOW_SECONDS`, `RANDOM_SEED`, regex de nombre de archivo de anotación). |
| `data_loading.py` | `load_metadata`, `load_annotations_for_case`, `load_all_annotations`, `merge_metadata_and_annotations`. Inyecta `case_id` desde el nombre del archivo porque los CSV de anotaciones no contienen esa columna. |
| `download.py` | `load_ecg_from_vitaldb(case_id, track_name, sampling_rate_hz)` y `save_ecg_npy`. Cache local en `data/raw/vitaldb_waveforms/case_<id>.npy`. |
| `preprocessing.py` | `exclude_rhythm_labels`, `exclude_bad_signal_quality`, `validate_columns`, `drop_exact_duplicates`, `apply_basic_filters`. |
| `windowing.py` | `WindowSpec` (dataclass) + `build_windows_for_case`. Ventanas centradas en cada latido, sobrelapamiento opcional, descarte de ventanas que se salen de la señal, propagación de NaN como `None` en la etiqueta. |
| `features.py` | `compute_time_features` (15 estadísticas por ventana), `compute_time_features_batch`, `compute_rr_intervals`, `compute_rr_features` (incluye `rr_rmssd`, `rr_pnn50`). |
| `modeling.py` | Pipelines baseline `SimpleImputer → StandardScaler → Clf`. `safe_n_splits` recorta `n_splits` al número de grupos disponibles. `assert_no_forbidden_features` bloquea features prohibidas. |
| `evaluation.py` | `compute_macro_metrics`, `per_class_report` con soporte, `class_support_per_split`, `classes_missing_in_train`, `confusion_matrix_with_totals`. |
| `utils.py` | `set_seed`, `get_logger`, `ensure_dir`, `list_files`. |

### 5.2 Notebooks

Todos parten de `notebooks/` y agregan la raíz al `sys.path` para importar
`src` sin instalar el paquete.

| Notebook | Propósito |
|---|---|
| `01_download_and_structure.ipynb` | Verifica que el paquete de PhysioNet esté en disco; lista archivos; carga `metadata.csv` y explora la carpeta de anotaciones. |
| `02_eda_annotations.ipynb` | Carga todas las anotaciones (inyectando `case_id` desde el nombre), las une con metadata, aplica filtros base y muestra distribución de `rhythm_label`, conteos por caso, distribución de `bad_signal_quality` y análisis descriptivo *complementario* de `beat_type`. |
| `03_ecg_loading_and_visualization.ipynb` | Descarga ECG desde VitalDB para un subconjunto reducido de `case_id` (cache en `.npy`) y visualiza tramos. |
| `04_windowing_and_feature_engineering.ipynb` | Itera sobre **todos** los `case_<id>.npy` presentes en disco (variable `MAX_CASES` para limitar en pruebas), construye ventanas y genera `data/processed/features_baseline.parquet`. |
| `05_baseline_modeling.ipynb` | Diagnóstico de clases por caso, split por grupo, Pipeline completo (`Imputer + Scaler + Clf` con `class_weight="balanced"`), métricas macro, reporte por clase con `support`, matriz de confusión absoluta con totales fila/columna, validación cruzada con `safe_n_splits`. |

### 5.3 Tests

| Archivo | Cobertura |
|---|---|
| `test_data_loading.py` | Validación de columnas, exclusión de `Noise`, exclusión de `bad_signal_quality` (bool y string), `drop_exact_duplicates`, `apply_basic_filters`, merge metadata + anotaciones, parser de `case_id` desde nombre de archivo, desambiguación por nombre exacto, `load_all_annotations` con inyección de `case_id`. |
| `test_windowing.py` | Shape correcta de ventanas, descarte de centros fuera de rango, validación de `overlap`, ventanas auxiliares por sobrelapamiento, `NaN` en `rhythm_label` → `None` (no la cadena `"nan"`). |
| `test_features.py` | Presencia de todas las features temporales, manejo de ventana constante (finitos), batch shape, intervalos RR básicos, RR sobre input vacío, bloqueo de `beat_type` / `rhythm_label` / `case_id` como features. |
| `test_modeling.py` | `safe_n_splits` recorta y aborta con grupos insuficientes, `make_group_kfold` produce folds disjuntos, estructura de pipelines (`imputer → scaler → clf` para LR; `imputer → clf` para RF), Pipeline tolera NaN, `assert_no_forbidden_features` menciona la columna ofensora en el error. |
| `test_evaluation.py` | `class_support_per_split` con 0 explícito para clases ausentes, `classes_missing_in_train`, layout de `confusion_matrix_with_totals` (incluyendo márgenes), `per_class_report` incluye `support`, `compute_macro_metrics` devuelve las claves esperadas en `[0, 1]`. |

---

## 6. Datos

### 6.1 Origen

- **Anotaciones y metadata:** descarga manual desde PhysioNet a
  `data/raw/physionet_annotations/`.
- **Señal ECG cruda:** se obtiene de VitalDB en runtime usando la librería
  `vitaldb`, indexando por `case_id`. Se cachea en
  `data/raw/vitaldb_waveforms/case_<id>.npy`.

### 6.2 Formato real verificado

- Archivos de anotación: `Annotation_file_<case_id>.csv` (singular, con
  guion bajo antes del entero).
- Columnas internas: `time_second`, `beat_type`, `rhythm_label`,
  `bad_signal_quality`, `bad_signal_quality_label`.
- **`case_id` no aparece como columna en los CSV de anotaciones**; vive
  solo en el nombre del archivo.
- `metadata.csv`: 482 filas × 79 columnas (verificado en la versión cargada
  para los smoke tests).

### 6.3 Estado actual del cache local

A la fecha del informe hay `.npy` descargados solo para 3 casos
(`case_1001`, `case_1002`, `case_1018`). El parquet de features generado a
partir de ellos contiene 3373 ventanas válidas (3374 antes de filtrar 1 fila
con etiqueta nula) y 15 features temporales por ventana.

Distribución de clases observada en ese parquet:

| `rhythm_label` | total ventanas | nº de casos donde aparece |
|---|---:|---:|
| `N` | 2609 | 3 |
| `Patterned Ventricular Ectopy` | 455 | 1 (solo `case_1001`) |
| `SVTA` | 268 | 2 (`case_1002`, `case_1018`) |
| `VT` | 41 | 1 (solo `case_1018`) |

Esta concentración por caso es la causa estructural por la que el baseline
actual produce métricas macro muy bajas con split por grupo: hay clases que
nunca aparecen en train cuando su único caso queda en test.

---

## 7. Cambios aplicados (cronológico)

### 7.1 `f5140f2` — *Initial project structure for ECG rhythm classification*

- Andamiaje completo del repositorio (carpetas, archivos base).
- README técnico con secciones de problema, fuente, objetivo, instalación,
  descarga, advertencias académicas, flujo, evaluación y limitaciones.
- `.gitignore` robusto verificado con `git check-ignore` contra `*.csv`,
  `*.npy`, `*.parquet`, `*.joblib`, `*.pkl`, `models/*`, `reports/figures/*`,
  `.venv/`, `__pycache__/`.
- `requirements.txt` y `environment.yml` (Python 3.11) con dependencias
  iniciales.
- Módulos `src/` con docstrings y firmas estables.
- Notebooks `01` a `05` con secciones markdown y celdas base.
- Tests iniciales (`test_data_loading.py`, `test_windowing.py`,
  `test_features.py`).
- Repositorio privado creado en GitHub y `main` empujado.

### 7.2 `6657569` — *Fix annotation loading: derive case_id from filename and rename beat_time to time_second*

**Síntoma:** `notebooks/02_eda_annotations.ipynb` levantaba `KeyError` al
hacer el merge contra `metadata.csv` porque las anotaciones no traían la
columna `case_id`.

**Hallazgo:** los CSV de anotaciones (`Annotation_file_<id>.csv`) no incluyen
`case_id` como columna; el identificador vive en el nombre del archivo.
Además, la columna del tiempo del latido se llama `time_second` (no
`beat_time`).

**Cambios:**
- `config.py`: `BEAT_TIME_COLUMN = "time_second"`. Nuevo
  `ANNOTATION_FILENAME_REGEX = r"^Annotations?_file_(\d+)\.csv$"`
  (singular / plural, case-insensitive).
- `data_loading.py`: parser de filename, inyección de `case_id` por archivo
  y match por nombre exacto (evita que `case_id=1` colisione con `10`,
  `100`, `1001`). `load_all_annotations(case_ids=None)` ahora también
  inyecta `case_id`.
- Tests: parametrización del parser, desambiguación por nombre exacto,
  carga de archivos no existentes, concatenación múltiple.

### 7.3 `5d296e8` — *Rework baseline modeling: full Pipeline + group-aware validation*

**Síntoma reportado:** baseline con métricas demasiado bajas y matrices de
confusión ilegibles.

**Diagnóstico:**
1. Con solo 3 casos y clases concentradas en un único caso, cualquier split
   por grupo deja al menos una clase sin ejemplos en train → recall 0 por
   construcción → `f1_macro` hundido.
2. `GroupKFold(n_splits=5)` con 3 grupos disponibles falla con
   `ValueError: n_splits=5 > n_groups=3`.
3. La matriz de confusión normalizada por `true` produce filas/columnas
   NaN cuando una clase no aparece en `y_true`, ilegible visualmente.
4. Una fila con `rhythm_label` NaN se propagaba como la cadena literal
   `"nan"`, generando una "clase fantasma" en los reportes.
5. El pipeline original no usaba `Pipeline` de sklearn con `SimpleImputer`,
   por lo que cualquier NaN en features causaría crash.

**Cambios:**

`src/modeling.py`:
- Pipelines reescritos a `SimpleImputer(strategy="median") →
  StandardScaler → LogisticRegression` y `SimpleImputer → RandomForest`
  (RF no necesita scaler).
- `safe_n_splits(n_splits, groups)`: recorta `n_splits` al número de
  grupos disponibles y aborta con < 2 grupos.

`src/evaluation.py`:
- `class_support_per_split(y_train, y_test)`: tabla de soporte por clase
  en train/test con 0 explícito (no NaN).
- `classes_missing_in_train(y_train, y_test)`: lista de clases que solo
  aparecen en test.
- `confusion_matrix_with_totals(y_true, y_pred)`: matriz absoluta con
  columna `support_true` (totales por fila) y fila `predicted_total`
  (totales por columna).

`src/windowing.py`:
- NaN en `rhythm_label` se propaga como `None` en `WindowSpec.label`
  usando `pd.isna()`, no como la cadena `"nan"`.

`notebooks/04_windowing_and_feature_engineering.ipynb`:
- Itera sobre **todos** los `case_<id>.npy` en `data/raw/vitaldb_waveforms/`
  en lugar de hardcodear `head(3)`. Variable `MAX_CASES` al inicio para
  limitar en pruebas.
- Salta casos sin anotaciones con warning explícito.
- Outputs anteriores limpiados.

`notebooks/05_baseline_modeling.ipynb` — **reescrito completo**:

1. Setup
2. Carga + limpieza defensiva (NaN reales y string `"nan"` / `"none"` /
   vacíos)
3. Diagnóstico por caso × clase (crosstab + número de casos por clase +
   warning si alguna clase aparece en ≤ 1 caso)
4. Preparación X, y, groups + `assert_no_forbidden_features`
5. Split por `case_id` + soporte por clase en train/test + warning de
   clases ausentes en train
6. Definición del Pipeline (impresión de su estructura)
7. Fit + predicción + métricas macro tabuladas en un único DataFrame
8. `classification_report` por clase con `support`
9. Matriz de confusión absoluta con totales fila/columna + heatmap en
   conteos enteros (sin normalización)
10. `GroupKFold` con `safe_n_splits` + métricas por fold + agregados
    `mean`/`std`/`min`/`max`
11. Notas de interpretación

Tests añadidos:
- `tests/test_modeling.py` (8 tests): `safe_n_splits`,
  `make_group_kfold`, estructura del Pipeline, tolerancia a NaN.
- `tests/test_evaluation.py` (6 tests): soporte por split, clases
  faltantes, matriz con totales, `support` en reporte por clase.
- `tests/test_windowing.py`: nuevo test para NaN → None en label.

---

## 8. Estado actual del baseline

El notebook 05 está listo para ejecutarse y produce ahora salidas
**interpretables** aunque las métricas en términos absolutos sigan siendo
bajas, dado el cache actual de 3 casos. La sección 3 del notebook expone
explícitamente qué clases están concentradas en un único caso, y la sección
5 expone qué clases no aparecen en train tras el split.

No se reportan cifras de F1/recall/precision en este informe porque
cualquier número que se reporte sobre 3 casos no es representativo de lo
que dará el baseline con un dataset completo. Las cifras solo cobrarán
sentido tras descargar más casos en `nb03` y regenerar el parquet en
`nb04`.

---

## 9. Limitaciones conocidas

- **Tamaño del cache local.** Solo 3 casos descargados; no permite un
  baseline honesto. La sección 3 de nb05 lo señala explícitamente.
- **Desbalance estructural por caso.** Con anotaciones validadas pero
  pacientes intraoperatorios, clases minoritarias (`VT`,
  `Patterned Ventricular Ectopy`) aparecen en pocos casos.
- **Features muy básicas.** 15 estadísticas temporales sobre la ventana
  cruda, sin filtrado pasa-banda, sin features morfológicas, sin features
  espectrales, sin features RR locales por ventana. Suficiente para
  validar el pipeline; insuficiente para discriminación clínica.
- **Sin manejo avanzado de desbalance.** El baseline usa
  `class_weight="balanced"` y nada más. No hay SMOTE, no hay focal loss,
  no hay ensemble.
- **Sin búsqueda de hiperparámetros.** Por diseño: este es solo el
  baseline.
- **No clínico.** El proyecto es exclusivamente académico; no validado
  para uso médico.

---

## 10. Próximos pasos sugeridos

Antes de iterar en modelado:

1. Descargar más `case_id` en `notebooks/03_ecg_loading_and_visualization.ipynb`.
   Verificar al final que el número de archivos en
   `data/raw/vitaldb_waveforms/` es consistente con lo esperado.
2. Re-ejecutar `notebooks/04_windowing_and_feature_engineering.ipynb` con
   `MAX_CASES = None` para regenerar `data/processed/features_baseline.parquet`
   con todos los casos disponibles.
3. Volver a correr `notebooks/05_baseline_modeling.ipynb` completo. Revisar
   primero la sección 3 (diagnóstico por caso) y la sección 5 (soporte por
   split) **antes** de mirar las métricas.

Tareas abiertas de modelado (para fases posteriores):

- Filtrado de señal (pasa-banda ~0.5–40 Hz, remoción de línea de base) antes
  de extraer features.
- Features adicionales: morfológicas (amplitud R, ancho QRS si hay detección
  de fiducial points), RR locales por ventana (intervalo al latido previo y
  siguiente, ratios), espectrales (energía en bandas Welch).
- Probar XGBoost (`src.modeling.build_xgb_pipeline` ya existe; requiere
  pasar `sample_weight` para balanceo multiclase).
- Búsqueda de hiperparámetros con `GridSearchCV` o `RandomizedSearchCV`
  envueltos en un `GroupKFold` interno (validación anidada por caso).
- Reportes persistidos en `reports/figures/` y `reports/tables/` (excluidos
  por `.gitignore` para no inflar el repo).

---

## 11. Trazabilidad

- **Repositorio:** privado en GitHub (rama `main`).
- **Commits hasta la fecha:**
  - `f5140f2` — Initial project structure for ECG rhythm classification
  - `6657569` — Fix annotation loading: derive case_id from filename and
    rename beat_time to time_second
  - `5d296e8` — Rework baseline modeling: full Pipeline + group-aware
    validation
  - `3b22cab` — Add modeling and hyperparameter search infrastructure
    (línea ECG completa, hoy `legacy`)
  - *(commit de la iteración tabular)* — Pivot a modelado tabular
    sobre anotaciones + metadata; ECG marcado `legacy`.
- **Tests:** ejecutables con `pytest tests/`. 84 tests verdes a la fecha
  (64 previos + 20 nuevos en `tests/test_tabular_search.py`).
- **Reproducibilidad:** semilla por defecto en `config.RANDOM_SEED = 42`.
  Aplicada en scripts y notebooks vía `set_seed`.
- **Datos:** ninguno versionado. Las rutas reales se construyen desde
  `config.PROJECT_ROOT` (rutas relativas).

---

## 12. Iteración tabular (2026-05-27)

Nueva fase. Reemplaza al baseline ECG como flujo principal.

### 12.1 Nuevos componentes

| Path | Propósito |
|---|---|
| `scripts/01_audit_filtered_tabular_dataset.py` | Audita anotaciones+metadata. Produce `tabular_dataset_audit.csv`, `tabular_class_distribution.csv`, `tabular_cases_per_class.csv`, `tabular_missing_values.csv`, `tabular_columns_classification.csv`. |
| `scripts/02_build_filtered_tabular_modeling_dataset.py` | Construye `data/processed/filtered_tabular_modeling_dataset.parquet`. Coerce `age` (`>89 → 89`), añade RR locales, descarta constantes. |
| `scripts/03_run_tabular_hyperparameter_search.py` | CLI de `RandomizedSearchCV` multi-modelo. Acepta `--debug`, `--max-cases`, `--models`, `--n-iter`, `--n-splits`. |
| `src/tabular_search.py` | Registro de modelos, classify_features, build_pipeline_for_model, run_search_for_model, evaluate_on_test, extract_feature_importance. Incluye wrapper `_XGBClassifierSafe`. |
| `src/preprocessing.build_tabular_preprocessor` | `ColumnTransformer(Imputer+Scaler / Imputer+OneHotEncoder(handle_unknown='ignore'))`. |
| `src/modeling.make_train_test_group_split_with_coverage` | 80/20 por `case_id` con cobertura de clases (reuso de la iteración previa). |
| `notebooks/06_tabular_modeling_hyperparameter_search.ipynb` | Notebook interactivo. |
| `tests/test_tabular_search.py` | 20 tests sobre clasificación de columnas, bloqueo de prohibidas, split, preprocesador, estructura de pipelines. |
| `reports/TABULAR_MODELING_REPORT.md` | Informe técnico del flujo tabular. |
| `reports/NEXT_STEPS_FOR_CHATGPT.md` | Handoff actualizado. |

### 12.2 Decisiones de leakage

`src/config.py::TABULAR_LEAKAGE_COLUMNS` excluye del set de predictores:
`rhythm_label`, `beat_type`, `rhythm_classes`, `bad_signal_quality`,
`bad_signal_quality_label`, `case_id`, `caseid` (variante con typo del
archivo 2453), `subjectid`, `source_file`, `icu_days`, `death_inhosp`,
`adm`, `dis`.

### 12.3 Auditoría del dataset (cifras reales y reproducibles)

| métrica | valor |
|---|---:|
| filas antes filtros | 676 250 |
| filas después filtros | 639 460 |
| cases antes / después | 482 / 482 |
| n clases | 10 |
| numéricas candidatas | 54 |
| categóricas candidatas | 17 |
| excluidas por leakage | 11 |
| excluidas por alta cardinalidad | 3 (`dx`, `opname`, `age` antes de coerción) |

Distribución global de clases (`reports/tables/tabular_class_distribution.csv`):
N (61 %), AFIB/AFL (25 %), Patterned Ventricular Ectopy (3.7 %), SND
(3.5 %), Patterned Atrial Ectopy (3.1 %), WAP/MAT (1.6 %), SVTA (1.0 %),
AVB (0.7 %), VT (0.2 %), Unclassifiable (0.01 %).

### 12.4 Split sobre dataset completo

Con `random_state=42`, primer intento exitoso:
- 10 / 10 clases cubiertas en train y test.
- 385 cases en train (510 287 filas) / 97 cases en test (129 173 filas).
- `actual_test_fraction = 0.202`.

### 12.5 Estado de la corrida

El CLI produce todos los CSVs/PNGs requeridos en `reports/tables/` y
`reports/figures/`. La corrida que persistió las cifras de la última
sección 5 del `TABULAR_MODELING_REPORT.md` es la documentada en
`reports/tables/tabular_hyperparameter_search_meta.json` (consultar
`args.n_iter`, `args.n_splits`, `args.max_cases`, `args.debug` antes de
interpretar).

### 12.6 Línea ECG (legacy)

Banner `[LEGACY]` añadido en:
- `src/search.py`
- `scripts/01_download_all_available_ecg.py`
- `scripts/02_build_features_all_windows.py`
- `scripts/03_run_hyperparameter_search.py`

Los notebooks `03-05` y `06_full_modeling_hyperparameter_search.ipynb`
quedan disponibles para inspección histórica. Los `.npy` (case_1001,
case_1002, case_1018) cacheados en `data/raw/vitaldb_waveforms/` no se
usan en el flujo activo.
