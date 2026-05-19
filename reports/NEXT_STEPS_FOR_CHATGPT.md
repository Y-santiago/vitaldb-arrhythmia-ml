# Estado para la siguiente instrucción (handoff)

**Fecha del handoff:** 2026-05-19
**Rama actual:** `main`
**Último commit antes del handoff:** ver `git log -1 --oneline`.

Este documento resume objetivamente qué se hizo en la iteración de
modelado e hiperparámetros, qué quedó pendiente y qué decisiones técnicas
hay que tomar antes de continuar.

---

## 1. Estado exacto del repositorio

```
vitaldb-arrhythmia-ml/
├── src/                    # módulos del paquete (importables como `from src import ...`)
│   ├── config.py           # rutas, columnas, semillas, regex de archivos
│   ├── data_loading.py     # carga metadata + anotaciones (inyecta case_id desde filename)
│   ├── download.py         # wrapper a vitaldb.load_case y persistencia a .npy
│   ├── preprocessing.py    # filtros base (Noise, bad_signal_quality)
│   ├── windowing.py        # construcción de ventanas alrededor de cada latido
│   ├── features.py         # 15 features temporales + 4 RR locales por latido
│   ├── modeling.py         # split por grupo, safe_n_splits, pipelines baseline
│   ├── search.py           # MODEL_REGISTRY + orquestación de RandomizedSearchCV
│   ├── evaluation.py       # métricas macro, soporte por split, matrices con totales
│   └── utils.py            # logger, set_seed, ensure_dir, list_files
├── scripts/
│   ├── 01_download_all_available_ecg.py        # descarga ECG masiva
│   ├── 02_build_features_all_windows.py        # genera 3 parquets (1.2/2.0/5.0 s)
│   └── 03_run_hyperparameter_search.py         # CLI de la búsqueda completa
├── notebooks/
│   ├── 01_download_and_structure.ipynb         # local (no commiteado)
│   ├── 02_eda_annotations.ipynb                # local (no commiteado)
│   ├── 03_ecg_loading_and_visualization.ipynb  # local (no commiteado)
│   ├── 04_windowing_and_feature_engineering.ipynb
│   ├── 05_baseline_modeling.ipynb
│   └── 06_full_modeling_hyperparameter_search.ipynb   # nuevo
├── reports/
│   ├── MODELING_REPORT.md                      # nuevo (informe técnico)
│   ├── NEXT_STEPS_FOR_CHATGPT.md               # este archivo
│   ├── PROJECT_REPORT.md                       # estado previo
│   ├── tables/                                 # CSVs reales producidos (ignorados por git)
│   └── figures/                                # PNGs reales producidos (ignorados por git)
├── tests/
│   ├── test_data_loading.py
│   ├── test_windowing.py
│   ├── test_features.py
│   ├── test_modeling.py
│   ├── test_evaluation.py
│   └── (pendiente: tests de split-with-coverage y search)
├── data/
│   ├── raw/physionet_annotations/              # paquete PhysioNet (en disco, no en git)
│   ├── raw/vitaldb_waveforms/                  # 3 archivos `.npy` (en disco, no en git)
│   ├── interim/                                # vacío
│   └── processed/                              # 3 parquets generados (no en git)
└── models/                                     # vacío
```

**Estado de tests:** `pytest tests/` corrió **50/50 OK** antes de la
iteración. Los tests nuevos para la fase de modelado están pendientes y
se añaden en Fase G (ver §5).

**Estado de git:** los notebooks 01-03 tienen modificaciones locales
(outputs de ejecuciones del usuario) que NO se versionan por defecto.
Los nuevos archivos `src/search.py`, `scripts/*.py`, `notebooks/06...ipynb`,
`reports/MODELING_REPORT.md`, `reports/NEXT_STEPS_FOR_CHATGPT.md` están
listos para commit.

---

## 2. Archivos nuevos y modificados

### Nuevos

| Path | Propósito |
|---|---|
| `src/search.py` | Registro de modelos, `WindowRunConfig`, `run_one_window`, ensamblaje de tablas de resultados. |
| `scripts/__init__.py` | Marker package. |
| `scripts/01_download_all_available_ecg.py` | CLI de descarga masiva desde VitalDB con tolerancia a fallos. |
| `scripts/02_build_features_all_windows.py` | CLI para generar los 3 parquets por tamaño de ventana. |
| `scripts/03_run_hyperparameter_search.py` | CLI de la búsqueda; produce todos los CSVs y PNGs requeridos. |
| `notebooks/06_full_modeling_hyperparameter_search.ipynb` | Notebook interactivo equivalente al CLI 03. |
| `reports/MODELING_REPORT.md` | Informe técnico con las 12 secciones obligatorias. |
| `reports/NEXT_STEPS_FOR_CHATGPT.md` | Este archivo. |

### Modificados

| Path | Cambio |
|---|---|
| `src/config.py` | Sin cambios funcionales en esta iteración (revisado). |
| `src/modeling.py` | Removido `multi_class="auto"` deprecado de LogReg. Nueva función `make_train_test_group_split_with_coverage(X, y, groups, test_size, random_state, max_attempts)` que itera semillas para maximizar cobertura de clases sin usar métricas de desempeño. |
| `src/features.py` | Añadida `compute_per_beat_rr_features(beat_times)` que retorna `rr_prev`, `rr_next`, `rr_mean_local`, `rr_ratio` por latido. |

---

## 3. Comandos ejecutados

Reproducibilidad de esta corrida (puede repetirse desde cero):

```bash
# Verificar dependencias (vitaldb, xgboost, sklearn, pandas, etc. ya instaladas)
python -c "import sklearn, xgboost, vitaldb, pandas, numpy"

# 1) Descarga de ECG (en debug = solo los 3 que estaban cacheados)
python scripts/01_download_all_available_ecg.py --case-ids 1001,1002,1018

# 2) Generación de features para los 3 tamaños de ventana
python scripts/02_build_features_all_windows.py --debug

# 3) Búsqueda de hiperparámetros (debug: n_iter=3, n_splits=2, n_jobs=1)
python scripts/03_run_hyperparameter_search.py --debug

# Tests
python -m pytest tests/
```

Outputs generados (los archivos están en disco pero ignorados por git):

- `data/processed/features_w1p2s.parquet`, `features_w2p0s.parquet`,
  `features_w5p0s.parquet` (3373 × 27 cada uno).
- `reports/tables/download_status.csv`, `full_model_comparison.csv`,
  `full_model_comparison_by_window.csv`, `best_hyperparameters.csv`,
  `class_support_train_test_by_window.csv`, `classes_missing_by_split.csv`,
  `test_classification_report_best_model.csv`,
  `test_confusion_matrix_best_model_with_totals.csv`,
  `hyperparameter_search_meta.json`.
- `reports/figures/confusion_matrix_best_model_absolute.png`,
  `confusion_matrix_best_model_normalized.png`.

---

## 4. Resultados principales (de la corrida real)

Cohorte: 3 case_id (1001, 1002, 1018). Modo `--debug`.

### Mejor modelo global

| campo | valor |
|---|---|
| modelo | `xgboost` |
| ventana | 5.0 s |
| `test_f1_macro` | 0.386 |
| `test_balanced_accuracy` | 0.647 |
| `test_accuracy` | 0.905 |
| `cv_f1_macro` | 0.286 |
| best_params | `n_estimators=400, max_depth=10, learning_rate=0.01, subsample=1.0, colsample_bytree=0.8, min_child_weight=5` |

### Mejor por ventana (`test_f1_macro`)

| ventana | mejor modelo | test_f1_macro |
|---:|---|---:|
| 1.2 s | logreg | 0.322 |
| 2.0 s | decision_tree | 0.375 |
| 5.0 s | xgboost | 0.386 |

### Split usado (igual en las 3 ventanas)

- `chosen_seed = 44`
- `train_groups = [1001, 1002]`, `test_groups = [1018]`
- `actual_test_fraction = 0.314` (objetivo 0.20; imposible con solo 3 grupos)
- Clases en train: `N, Patterned Ventricular Ectopy, SVTA`
- Clases en test: `N, SVTA, VT`
- **`VT` no aparece en train** → recall 0 obligatorio para esa clase
- **`Patterned Ventricular Ectopy` no aparece en test** → no se mide su F1

### Matriz de confusión del ganador (test, absoluta)

|                | pred N | pred SVTA | pred VT | support_true |
|---             |---:    |---:       |---:     |---:          |
| **N** (1008)   | 948    | 60        | 0       | 1008         |
| **SVTA** (11)  | 0      | 11        | 0       | 11           |
| **VT** (41)    | 9      | 32        | 0       | 41           |

---

## 5. Problemas encontrados y cómo se resolvieron

| Problema | Causa raíz | Resolución aplicada |
|---|---|---|
| `XGBClassifier` lanzaba `Invalid classes inferred from unique values of y. Expected: [0 1], got [0 2]` durante CV. | XGBoost ≥ 2.0 valida estrictamente que las clases sean enteros consecutivos; cuando un fold de GroupKFold tiene un subconjunto de clases, el `LabelEncoder` externo deja gaps. | Wrapper `_XGBClassifierSafe` en `src/search.py` que reencoda labels dentro de su `fit` y revierte en `predict`. |
| `MLPClassifier` fallaba con `TypeError: ufunc 'isnan' not supported` cuando `early_stopping=True`. | Bug conocido de sklearn con etiquetas string + early stopping. | `early_stopping=False` en `_mlp_factory`. Trade-off: más `max_iter`. |
| Test fraction = 0.314 en vez de 0.20. | Con 3 grupos, `GroupShuffleSplit(test_size=0.2)` redondea a 1 grupo → el grupo más grande arrastra la fracción. | Documentado. Se resuelve con más casos. |
| Clases ausentes en train o test. | Cohorte chica con clases concentradas en pocos casos. | `make_train_test_group_split_with_coverage` selecciona la mejor cobertura posible y reporta clases faltantes. Estructural; no resoluble sin más datos. |
| `multi_class="auto"` deprecation warning en LogReg (sklearn ≥ 1.5). | Comportamiento default es ahora `multinomial`. | Removido el argumento de `_logreg_factory`. |

---

## 6. Preguntas técnicas pendientes

1. **¿Reintento automático de errores de descarga?** Hoy `scripts/01...py`
   registra errores en `download_status.csv` y continúa. ¿Conviene
   añadir un modo `--retry-errors` que recorra solo las filas con
   `status="error"` y reintente?
2. **¿`scale_pos_weight` manual para XGBoost multiclase?** XGBoost no
   expone `class_weight="balanced"` para multiclase. Una opción es pasar
   `sample_weight = compute_sample_weight("balanced", y_train)` en
   `fit`. Esto rompe la interfaz uniforme con sklearn pero podría dar
   un boost en recall de clases minoritarias.
3. **¿`SGDClassifier(loss="hinge")` además de LinearSVC?** Más rápido
   con dataset grande, soporta `partial_fit` para online learning. ¿Vale
   añadirlo al registro?
4. **¿Filtrado pasa-banda antes de features?** Sin él, las features
   estadísticas están contaminadas por baseline wander y ruido de red.
   ¿Aceptable añadirlo como un step adicional dentro del Pipeline o se
   prefiere como preprocesamiento offline en `scripts/02...py`?
5. **¿Persistir el mejor estimador a disco?** Hoy no se guarda; cada
   notebook/script lo regenera. ¿Se quiere `joblib.dump` del
   `best_estimator_` global en `models/` (ignorado por git)?
6. **¿Reporte intermedio sobre folds de CV?** Hoy se guarda solo el
   promedio del mejor candidato. Si interesa la varianza, se puede
   guardar el dataframe completo de `cv_results_` por modelo.

---

## 7. Recomendación concreta para la siguiente instrucción

**El bloqueante principal es la cohorte: 3 casos no permiten estimar
generalización**. El siguiente paso debería ser uno de estos dos
caminos, y conviene decidir cuál antes de pedir trabajo nuevo.

### Camino A — Descargar todo y correr el full search

Costo: descarga lenta (~480 casos × tiempo variable por caso) + run
largo de la búsqueda (~`30 × 6 × 3 × 5 = 2700` fits, varias horas según
hardware).

Instrucción sugerida para la siguiente iteración:

> "Ejecuta `python scripts/01_download_all_available_ecg.py`. Revisa
> `reports/tables/download_status.csv` y reporta cuántos casos quedaron
> en `status="error"`. Luego ejecuta `python scripts/02_build_features_all_windows.py`
> sin `--debug`. Finalmente ejecuta
> `python scripts/03_run_hyperparameter_search.py --n-iter 30 --n-splits 5`.
> Cuando termine, sustituye las cifras del `MODELING_REPORT.md` con las
> del full run y elimina las notas de modo `--debug`."

### Camino B — Refinar el pipeline antes del full run

Costo: bajo en cómputo, alto en código. Permite que el full run incluya
mejores features.

Instrucción sugerida:

> "Antes de correr la cohorte completa, añade un step de filtrado
> pasa-banda 0.5–40 Hz en `scripts/02_build_features_all_windows.py`
> usando `scipy.signal.butter` + `filtfilt`. Añade tres features
> espectrales por ventana (energía en [0–5 Hz], [5–15 Hz], [15–40 Hz])
> con `scipy.signal.welch`. Implementa también detección simple de pico
> R por ventana y deriva amplitud y polaridad. Actualiza los tests."

### Mi recomendación

Iría por **Camino A** primero. Las features actuales ya son útiles para
calibrar el orden relativo de los modelos; el cuello de botella
metodológico ahora es la cantidad de casos, no la calidad de las
features. Después del full run, decidir features con datos reales sobre
la mesa.

---

## Apéndice — Comandos útiles para verificar el handoff

```bash
# Tests
python -m pytest tests/ -q

# Reproducir la corrida documentada
python scripts/03_run_hyperparameter_search.py --debug

# Solo un modelo, una ventana
python scripts/03_run_hyperparameter_search.py --debug --models xgboost --windows 5.0

# Inspeccionar resultados
python -c "import pandas as pd; print(pd.read_csv('reports/tables/full_model_comparison.csv').round(3).to_string(index=False))"

# Verificar que no haya features prohibidas en los parquets
python -c "
import pandas as pd
from src.modeling import assert_no_forbidden_features
from src.config import FORBIDDEN_FEATURE_COLUMNS
for w in ['1p2s','2p0s','5p0s']:
    df = pd.read_parquet(f'data/processed/features_w{w}.parquet')
    forbidden_in_df = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in df.columns]
    print(f'{w}: columnas {len(df.columns)}; prohibidas presentes (esperado, no se usan como features): {forbidden_in_df}')
"
```
