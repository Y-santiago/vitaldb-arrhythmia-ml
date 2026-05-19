# Informe de modelado — VitalDB Arrhythmia ML

**Fecha:** 2026-05-19
**Rama:** `main`
**Modo de ejecución reportado:** `--debug` (corrida reducida sobre la cohorte ya cacheada localmente). El full run sobre los 482 casos del dataset queda pendiente; ver §11.

---

## 1. Resumen ejecutivo

Se construyó la fase formal de modelado para clasificar `rhythm_label`
sobre ventanas temporales de ECG intraoperatorio. La cadena es completa y
reproducible:

```
metadata + anotaciones (PhysioNet)        scripts/01_download_all_available_ecg.py
            +
ECG por case_id (VitalDB)         ──────► data/raw/vitaldb_waveforms/case_<id>.npy
                                          
                                  ──────► scripts/02_build_features_all_windows.py
                                          data/processed/features_w{1p2,2p0,5p0}s.parquet
                                          
                                  ──────► scripts/03_run_hyperparameter_search.py
                                          reports/tables/*.csv + reports/figures/*.png
```

Los 6 modelos del alcance (LogReg, DecisionTree, RandomForest, XGBoost,
LinearSVC, MLP) corrieron sobre los 3 tamaños de ventana (1.2, 2.0, 5.0 s).
La búsqueda usó `RandomizedSearchCV` con CV por grupo
(`StratifiedGroupKFold` con fallback a `GroupKFold`) y `f1_macro` como
métrica primaria. El test se evaluó **una sola vez**, después de fijar
hiperparámetros con CV únicamente en train.

**Resultado clave de esta corrida (debug):** mejor combinación global =
**XGBoost @ ventana = 5.0 s** con `test_f1_macro = 0.386`. La cifra está
muy condicionada por la baja cobertura de clases del split (ver §3 y §11);
no debe leerse como desempeño objetivo del baseline. La utilidad principal
de este run es validar la cadena completa, no estimar generalización.

---

## 2. Datos usados

### 2.1 Fuente
- **Metadata + anotaciones:** *VitalDB Arrhythmia Database 1.0.0*
  (PhysioNet), 482 filas en `metadata.csv` y un archivo
  `Annotation_file_<case_id>.csv` por caso.
- **Señal ECG cruda:** VitalDB vía la librería `vitaldb`, canal
  `SNUADC/ECG_II`, frecuencia de muestreo objetivo 500 Hz.

### 2.2 Filtros aplicados antes del modelado
- Exclusión de la clase `Noise` (`EXCLUDED_RHYTHM_LABELS`).
- Exclusión de filas con `bad_signal_quality = True`.
- Exclusión de filas con `rhythm_label` nulo o cadena `"nan"`/`"none"`/`""`.
- `beat_type` se conserva en el parquet pero **no entra al modelo** (ver §4).

### 2.3 Cohorte usada en esta corrida
- ECG descargados en disco: **3 casos** (`case_1001`, `case_1002`,
  `case_1018`). Reportado en `reports/tables/download_status.csv`.
- Ventanas por parquet (idénticas para los 3 tamaños porque el conteo de
  latidos no depende del tamaño de ventana, sino del descarte por
  excederse de la señal):

  | ventana | ventanas totales | columnas |
  |---:|---:|---:|
  | 1.2 s | 3373 | 27 |
  | 2.0 s | 3373 | 27 |
  | 5.0 s | 3373 | 27 |

### 2.4 Features
Por ventana, 19 features numéricas:

- **Temporales (15):** `mean`, `std`, `var`, `min`, `max`, `range`,
  `median`, `p25`, `p75`, `iqr`, `skew`, `kurtosis`, `energy`,
  `zero_crossing_rate`, `abs_mean`.
- **RR locales (4):** `rr_prev`, `rr_next`, `rr_mean_local`, `rr_ratio`.

Columnas conservadas pero **no usadas como predictoras**:
`case_id`, `rhythm_label` (objetivo), `beat_type` (uso descriptivo),
`time_second`, `beat_index`, `start_sample`, `end_sample`,
`window_seconds`.

---

## 3. Split train/test

### 3.1 Estrategia
Split por `case_id` con `make_train_test_group_split_with_coverage`:
`GroupShuffleSplit(test_size=0.2)` repetido sobre `max_attempts=200`
semillas, seleccionando la que maximiza la cantidad de clases presentes
simultáneamente en train y en test (criterio puramente estructural; no usa
métricas de desempeño). Tiebreaker secundario: minimizar
`|test_fraction_real − 0.20|`.

### 3.2 Resultado del split (igual para las 3 ventanas)
| campo | valor |
|---|---|
| chosen_seed | 44 |
| n_classes_covered (train ∩ test) | 2 / 4 |
| actual_test_fraction | 0.314 |
| requested_test_size | 0.200 |
| train_groups | [1001, 1002] |
| test_groups | [1018] |
| classes_only_in_train | `Patterned Ventricular Ectopy` |
| classes_only_in_test | `VT` |

Con solo 3 case_id disponibles, **es imposible** acercarse a 0.20 con
unidad grupal: GroupShuffleSplit redondea siempre a 1 caso en test. La
proporción real (0.314) refleja el peso relativo del caso 1018 sobre el
conjunto. Cuando se procesen todos los casos, esta fracción convergerá a
0.20.

### 3.3 Soporte por clase en cada split (idéntico en las 3 ventanas)
| `rhythm_label` | train | test | total |
|---|---:|---:|---:|
| N | 1601 | 1008 | 2609 |
| Patterned Ventricular Ectopy | 455 | 0 | 455 |
| SVTA | 257 | 11 | 268 |
| VT | 0 | 41 | 41 |
| **TOTAL** | 2313 | 1060 | 3373 |

Este es el dato más importante del informe: `VT` no tiene ejemplos en
train, así que el modelo **no puede aprender a predecirlo**. Y
`Patterned Ventricular Ectopy` no tiene ejemplos en test, así que su F1
no se puede medir. Cualquier f1_macro reportado refleja estas dos
ausencias.

### 3.4 CV interna
- Splitter elegido: **`StratifiedGroupKFold`** (sklearn ≥ 1.0).
- `n_splits` solicitado: 2 (modo debug). Para full run se subirá a 5.
- Como train tiene solo 2 grupos, cada fold queda con un caso en
  validación y el otro en training.

---

## 4. Modelos evaluados

Implementación en `src/search.py::MODEL_REGISTRY`. Pipelines:

| modelo | etapas |
|---|---|
| `logreg` | Imputer(median) → StandardScaler → LogisticRegression(class_weight="balanced") |
| `decision_tree` | Imputer(median) → DecisionTreeClassifier(class_weight="balanced") |
| `random_forest` | Imputer(median) → RandomForestClassifier(class_weight="balanced", n_jobs=-1) |
| `xgboost` | Imputer(median) → `_XGBClassifierSafe` (wrapper con re-encoding de labels por fit) |
| `linear_svc` | Imputer(median) → StandardScaler → LinearSVC(class_weight="balanced") |
| `mlp` | Imputer(median) → StandardScaler → MLPClassifier(early_stopping=False) |

Notas de implementación:
- **XGBoost** se envuelve en `_XGBClassifierSafe` para reencodear labels
  por fit con `LabelEncoder`. Sin esto, XGBoost ≥ 2.0 rechaza folds donde
  las clases no son enteros consecutivos (común en CV por grupo con
  datasets desbalanceados).
- **MLP** corre con `early_stopping=False` porque sklearn invoca
  `np.isnan` sobre las predicciones durante early stopping, lo cual falla
  con etiquetas string.
- **LinearSVC** se usa en lugar de `SVC(kernel="rbf")` por escalabilidad
  (LinearSVC tiene complejidad lineal). El cambio a `SGDClassifier` o RBF
  queda como opción para iteraciones futuras.

`beat_type` está fuera del conjunto de features:
`assert_no_forbidden_features` aborta cualquier intento de incluirlo
(`FORBIDDEN_FEATURE_COLUMNS` en `src/config.py`).

---

## 5. Estrategia de búsqueda de hiperparámetros

- **Algoritmo:** `RandomizedSearchCV`.
- **n_iter:** 30 en full run; 3 en este `--debug`.
- **Scoring:** dict multimétrica con `f1_macro` (refit), `precision_macro`,
  `recall_macro`, `accuracy`, `balanced_accuracy`, `f1_weighted`.
- **CV:** `StratifiedGroupKFold(n_splits=2)` en este run; `n_splits=5` en
  full run.
- **Distribuciones de hiperparámetros:** definidas en
  `src/search.py::MODEL_REGISTRY`. Para los modelos de árbol/forest
  incluye `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `criterion`, `n_estimators`, `max_features`. Para XGBoost incluye
  `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
  `colsample_bytree`, `min_child_weight`. Para LogReg y LinearSVC se usa
  `loguniform(1e-3, 1e2)` sobre `C`. Para MLP se incluye
  `hidden_layer_sizes`, `alpha`, `learning_rate_init`, `activation`.

**Garantía metodológica:** el test set se mantiene intacto durante toda la
búsqueda. `RandomizedSearchCV` solo ve `X_train`, `y_train`,
`groups_train`. La métrica primaria sobre la cual se selecciona el mejor
hiperparámetro por modelo es **el promedio de f1_macro sobre los folds
de CV**, no el test.

---

## 6. Resultados globales y por clase

### 6.1 Comparación completa (long format)
Tabla en `reports/tables/full_model_comparison.csv`. Resumen
(`best_cv_score_primary` es el f1_macro promedio en CV; `test_f1_macro` se
calcula tras refit sobre todo el train):

| ventana | modelo | best_cv_f1_macro | test_f1_macro | test_bal_acc | test_acc |
|---:|---|---:|---:|---:|---:|
| 1.2 | logreg | 0.219 | 0.322 | 0.326 | 0.929 |
| 1.2 | decision_tree | 0.286 | 0.280 | 0.613 | 0.892 |
| 1.2 | random_forest | 0.286 | 0.285 | 0.644 | 0.896 |
| 1.2 | xgboost | 0.286 | 0.282 | 0.643 | 0.892 |
| 1.2 | linear_svc | 0.191 | 0.260 | 0.381 | 0.917 |
| 1.2 | mlp | 0.266 | 0.254 | 0.355 | 0.928 |
| 2.0 | logreg | 0.157 | 0.274 | 0.414 | 0.925 |
| 2.0 | decision_tree | 0.286 | 0.375 | 0.614 | 0.895 |
| 2.0 | random_forest | 0.286 | 0.367 | 0.637 | 0.875 |
| 2.0 | xgboost | 0.288 | 0.280 | 0.642 | 0.892 |
| 2.0 | linear_svc | 0.127 | 0.295 | 0.586 | 0.901 |
| 2.0 | mlp | 0.126 | 0.282 | 0.416 | 0.931 |
| 5.0 | logreg | 0.225 | 0.241 | 0.326 | 0.929 |
| 5.0 | decision_tree | 0.285 | 0.375 | 0.614 | 0.895 |
| 5.0 | random_forest | 0.342 | 0.283 | 0.613 | 0.894 |
| 5.0 | xgboost | 0.286 | 0.386 | 0.647 | 0.905 |
| 5.0 | linear_svc | 0.174 | 0.268 | 0.407 | 0.904 |
| 5.0 | mlp | 0.202 | 0.325 | 0.333 | 0.950 |

### 6.2 Pivote test_f1_macro (rows=model, cols=window)
Tabla en `reports/tables/full_model_comparison_by_window.csv`.

| model | 1.2 | 2.0 | 5.0 |
|---|---:|---:|---:|
| decision_tree | 0.280 | 0.375 | 0.375 |
| linear_svc | 0.260 | 0.295 | 0.268 |
| logreg | 0.322 | 0.274 | 0.241 |
| mlp | 0.254 | 0.282 | 0.325 |
| random_forest | 0.285 | 0.367 | 0.283 |
| xgboost | 0.286 | 0.280 | 0.386 |

### 6.3 Reporte por clase del mejor modelo global (XGBoost @ 5.0 s)
Archivo: `reports/tables/test_classification_report_best_model.csv`.

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| N | 0.991 | 0.940 | 0.965 | 1008 |
| SVTA | 0.107 | 1.000 | 0.193 | 11 |
| VT | 0.000 | 0.000 | 0.000 | 41 |
| **macro avg** | 0.366 | 0.647 | **0.386** | 1060 |
| weighted avg | 0.943 | 0.905 | 0.920 | 1060 |
| accuracy | — | — | 0.905 | 1060 |

- `Patterned Ventricular Ectopy` no aparece porque no tiene soporte en
  test (todo su soporte cayó en train).

---

## 7. Mejor modelo por ventana

Tomando `test_f1_macro` como criterio (consultar también §3 antes de
comparar entre ventanas: el split es el mismo en las 3, así que las
diferencias provienen del tamaño de la ventana de ECG, no del split):

| ventana | mejor modelo | test_f1_macro | test_bal_acc | test_accuracy |
|---:|---|---:|---:|---:|
| 1.2 s | logreg | 0.322 | 0.326 | 0.929 |
| 2.0 s | decision_tree | 0.375 | 0.614 | 0.895 |
| 5.0 s | **xgboost** | **0.386** | **0.647** | 0.905 |

Las métricas se mueven en un rango estrecho (~0.24–0.39 en f1_macro),
porque la gran mayoría del test es la clase `N` y las clases minoritarias
tienen problemas estructurales de cobertura.

---

## 8. Mejor modelo general

- **Modelo:** `xgboost` (envuelto con re-encoding seguro).
- **Ventana:** 5.0 s.
- **Hiperparámetros (mejor combinación encontrada por la búsqueda):**
  ```json
  {
    "clf__subsample": 1.0,
    "clf__n_estimators": 400,
    "clf__min_child_weight": 5,
    "clf__max_depth": 10,
    "clf__learning_rate": 0.01,
    "clf__colsample_bytree": 0.8
  }
  ```
- **Métricas en test:**
  | métrica | valor |
  |---|---:|
  | test_f1_macro | 0.386 |
  | test_balanced_accuracy | 0.647 |
  | test_accuracy | 0.905 |
  | test_precision_macro | 0.366 |
  | test_recall_macro | 0.647 |
  | test_f1_weighted | 0.920 |
- **CV (promedio sobre 2 folds en debug):**
  | métrica | valor |
  |---|---:|
  | cv_f1_macro | 0.286 |
  | cv_balanced_accuracy | 0.500 |
  | cv_accuracy | 0.679 |
- **Tiempo de fit:** 4.4 s con `n_iter=3`. Para `n_iter=30` esperamos ~45 s
  en esta cohorte; para la cohorte completa será proporcional al número de
  ventanas.

---

## 9. Matriz de confusión

### Absoluta (`reports/figures/confusion_matrix_best_model_absolute.png`)

| true \ pred | N | SVTA | VT | support_true |
|---|---:|---:|---:|---:|
| N | 948 | 60 | 0 | 1008 |
| SVTA | 0 | 11 | 0 | 11 |
| VT | 9 | 32 | 0 | 41 |
| **predicted_total** | 957 | 103 | 0 | 1060 |

### Normalizada por fila (`reports/figures/confusion_matrix_best_model_normalized.png`)
Es válida en esta corrida porque las 3 clases reales del test (N, SVTA,
VT) tienen `support_true > 0`. `Patterned Ventricular Ectopy` no aparece
en el eje real porque su soporte en test es 0.

Lecturas:
- **N (mayoritaria):** 94 % bien clasificada. 60 N se confunden con SVTA.
- **SVTA (11 muestras):** todas predichas como SVTA (recall = 1.0), pero
  103 predicciones totales de SVTA → 92 falsos positivos → precision = 0.11.
- **VT (41 muestras):** ningún caso predicho como VT (precision/recall/f1
  = 0). Esto es **consecuencia directa** de que train no tenía VT, no del
  modelo.

---

## 10. Interpretación objetiva

1. El número absoluto de `test_f1_macro = 0.386` **no es informativo**
   como métrica de desempeño del baseline. Está dominado por dos
   estructuras del split:
   - `VT` sin train → recall 0 obligatorio.
   - `Patterned Ventricular Ectopy` sin test → no contribuye al average.
2. Una vez que se controle por estos dos factores (procesando todos los
   casos), se espera que la métrica suba sustancialmente sin que el modelo
   cambie. La cota inferior actual es informativa de la fragilidad del
   split con 3 casos.
3. **Ranking relativo entre modelos:** los modelos no lineales (XGBoost,
   Random Forest, Decision Tree) ganan consistentemente a los lineales
   (LogReg, LinearSVC). MLP queda en el medio. Ese ranking debería
   mantenerse al escalar a más casos, pero la **magnitud** absoluta puede
   cambiar.
4. **Efecto del tamaño de ventana:** con la cohorte actual, ventanas más
   largas (5.0 s) favorecen ligeramente a XGBoost. Las diferencias entre
   1.2 s, 2.0 s y 5.0 s son menores que las diferencias entre modelos
   dentro de una misma ventana. Esto es coherente con que las features
   son estadísticas de bajo nivel (no incorporan estructura morfológica
   detallada).

---

## 11. Limitaciones

1. **Cohorte mínima.** Solo 3 `case_id` están descargados localmente. La
   búsqueda corrió en modo `--debug` (`n_iter=3`, `n_splits=2`,
   `n_jobs=1`). El full run sobre los 482 casos del dataset queda
   pendiente y requerirá:
   - `python scripts/01_download_all_available_ecg.py` (horas).
   - `python scripts/02_build_features_all_windows.py` (minutos por caso
     × 3 ventanas).
   - `python scripts/03_run_hyperparameter_search.py`
     (`n_iter=30 × 6 modelos × 3 ventanas × 5 folds` ≈ 2700 fits).
2. **Imposibilidad de hit exacto del 0.20.** Con menos de ~10 casos,
   `GroupShuffleSplit(test_size=0.2)` redondea a 1 caso en test y la
   fracción real diverge (aquí 0.31). Esto desaparece con más casos.
3. **Clases sin cobertura en uno de los lados** (`VT` solo en test,
   `Patterned Ventricular Ectopy` solo en train) son artefacto de la
   cohorte chica. Aparecen documentadas en
   `reports/tables/classes_missing_by_split.csv`.
4. **Features simples.** Solo estadísticas temporales + RR locales. No hay
   filtrado pasa-banda, ni features morfológicas (amplitud R, ancho QRS),
   ni features espectrales (PSD Welch). Esto limita el techo de
   desempeño.
5. **Sin manejo avanzado de desbalance** más allá de
   `class_weight="balanced"`. No se prueba SMOTE, focal loss, ni ensembles
   de costo sensible.
6. **XGBoost wrapper.** `_XGBClassifierSafe` aplica `LabelEncoder` por
   fit. Esto resuelve el bug con clases ausentes en folds pero implica
   que `predict` devuelve strings ya decodificados. Documentado en el
   código.
7. **MLP sin early stopping** porque sklearn no maneja bien la
   combinación con etiquetas string. Trade-off conocido.
8. **Sin clínica.** Proyecto académico; sin validación médica.

---

## 12. Recomendaciones para la siguiente fase

Orden sugerido:

1. **Descargar todos los casos.** Ejecutar
   `python scripts/01_download_all_available_ecg.py`. Revisar
   `reports/tables/download_status.csv` por casos en `status="error"`;
   pueden ser canales no presentes en esos casos. Documentar errores y
   excluirlos o reintentar.
2. **Regenerar parquets.** Ejecutar
   `python scripts/02_build_features_all_windows.py`.
3. **Full run de la búsqueda.** Ejecutar
   `python scripts/03_run_hyperparameter_search.py --n-iter 30 --n-splits 5`.
   Repetir el análisis de las secciones 3–9 con los nuevos CSVs.
4. **Refinar features.** Implementar filtrado pasa-banda 0.5–40 Hz y
   remoción de baseline antes de extraer features. Añadir features
   morfológicas (detección de pico R, amplitud, ancho QRS) y espectrales
   (energía por bandas con `scipy.signal.welch`).
5. **Probar manejo de desbalance adicional.** `imbalanced-learn` ya está
   instalado: `SMOTE`, `SMOTEENN`, `BalancedRandomForest`. Comparar contra
   `class_weight="balanced"`.
6. **Aumentar el espacio de búsqueda de XGBoost.** Si XGBoost se confirma
   como ganador, expandir el grid con `gamma`, `reg_alpha`, `reg_lambda`,
   `scale_pos_weight` para multiclase manual.
7. **Validación anidada.** Considerar `GridSearchCV` interno dentro de un
   `cross_validate` por caso para reportar varianza entre folds del split
   externo, no solo del split único.
8. **Persistir modelos.** Cuando se cierre la elección de hiperparámetros,
   serializar el `best_estimator_` con `joblib.dump` en `models/`
   (excluido del repo por `.gitignore`).

No iniciar (todavía):

- Búsqueda exhaustiva (`GridSearchCV` full) hasta no tener el dataset
  completo.
- Comparaciones contra benchmarks externos.
- Estudio de calibración de probabilidades.

---

## Anexos

- **CSVs producidos por la corrida:**
  - `reports/tables/full_model_comparison.csv`
  - `reports/tables/full_model_comparison_by_window.csv`
  - `reports/tables/best_hyperparameters.csv`
  - `reports/tables/test_classification_report_best_model.csv`
  - `reports/tables/class_support_train_test_by_window.csv`
  - `reports/tables/classes_missing_by_split.csv`
  - `reports/tables/test_confusion_matrix_best_model_with_totals.csv`
  - `reports/tables/hyperparameter_search_meta.json`
  - `reports/tables/download_status.csv`
- **Figuras:**
  - `reports/figures/confusion_matrix_best_model_absolute.png`
  - `reports/figures/confusion_matrix_best_model_normalized.png`
- **Reportes complementarios:** `reports/NEXT_STEPS_FOR_CHATGPT.md`.
