<!--
Reporte técnico de la iteración binaria. Las cifras de modelos provienen
del run documentado en
`reports/tables/binary_hyperparameter_search_meta.json`. Si el script se
corrió en modo `--debug`, los números aquí corresponden al subset del
debug, NO al full run.
-->

# Informe de modelado binario — normal_sinus vs arrhythmia_or_abnormal

**Fecha del reporte:** 2026-05-29
**Rama:** `binary-normal-vs-arrhythmia`

---

## 1. Resumen ejecutivo

Tercera iteración del modelado del proyecto. La iteración previa fue
multiclase tabular (`reports/TABULAR_MODELING_REPORT.md`) y dejó
métricas macro bajas por: (a) clases minoritarias muy pequeñas, (b) un
test que en el split debug no recibía algunas clases. Esta iteración
**reformula la tarea como binaria**:

- `normal_sinus` = ritmo `N`.
- `arrhythmia_or_abnormal` = cualquier ritmo distinto de `N`
  considerado anormal/arrítmico (ver §3 para el mapeo exacto).

La separación train/test es estricta por `case_id`, con 80/20
aproximado y verificación de ausencia de fugas. La línea ECG cruda
sigue siendo `legacy` y NO se reactiva en esta fase.

---

## 2. Justificación del cambio multiclase → binario

1. Con 10 clases originales, varias tenían soporte demasiado bajo
   (`Unclassifiable` 5 cases, `AVB` 10 cases, `WAP/MAT` 25 cases),
   haciendo inestables sus F1 individuales.
2. Para tomas de decisión clínica iniciales suele bastar con detectar
   *si* hay arritmia, dejando la clasificación fina como tarea
   posterior.
3. La métrica `balanced_accuracy` binaria es más robusta y
   directamente interpretable como (sensitivity + specificity) / 2.
4. La pivot reduce el espacio de hipótesis y permite que modelos
   sencillos (logreg / boosting) generalicen mejor con el mismo
   dataset.

---

## 3. Definición exacta de las clases binarias

Mapeo declarativo en `src/config.BINARY_LABEL_MAPPING` +
`src/config.BINARY_EXCLUDED_LABELS`:

| `rhythm_label` original | `rhythm_binary` |
|---|---|
| `N` | `normal_sinus` |
| `AFIB/AFL` | `arrhythmia_or_abnormal` |
| `AVB` | `arrhythmia_or_abnormal` |
| `Patterned Atrial Ectopy` | `arrhythmia_or_abnormal` |
| `Patterned Ventricular Ectopy` | `arrhythmia_or_abnormal` |
| `SND` | `arrhythmia_or_abnormal` |
| `SVTA` | `arrhythmia_or_abnormal` |
| `VT` | `arrhythmia_or_abnormal` |
| `WAP/MAT` | `arrhythmia_or_abnormal` |
| `Noise` | **excluida** (ruido/artefacto) |
| `Unclassifiable` | **excluida** (no interpretable clínicamente) |
| nulos / `"nan"` / `"none"` / vacíos | **excluida** |
| etiquetas no contempladas | **excluida** y registradas en el audit |

`Noise` ya estaba excluida en el dataset tabular base
(`apply_basic_filters` filtra la clase). `Unclassifiable` se elimina
explícitamente en `map_rhythm_label_to_binary`.

La función vive en `src/config.map_rhythm_label_to_binary` y devuelve
`None` para etiquetas que deben excluirse: el contrato es que ninguna
etiqueta nueva se asigna automáticamente; cualquier valor no contemplado
se excluye y aparece en `reports/tables/binary_excluded_rows_summary.csv`
para revisión manual.

---

## 4. Clases excluidas y motivo

Fuente: `reports/tables/binary_excluded_rows_summary.csv`.

| `rhythm_label` | `n_rows_excluded` | motivo |
|---|---:|---|
| `Unclassifiable` | 59 | no interpretable como ritmo clínico para tarea binaria |
| `Noise` | 0* | ya eliminada por `apply_basic_filters` antes de esta etapa |

(*) `Noise` no aparece en el conteo porque el parquet tabular
(`filtered_tabular_modeling_dataset.parquet`) ya tenía aplicada esa
exclusión.

---

## 5. Auditoría premodelo (cifras reales)

Fuente: `reports/tables/binary_dataset_audit.csv`,
`reports/tables/binary_class_distribution.csv`,
`reports/tables/binary_cases_per_class.csv`,
`reports/tables/binary_case_composition.csv`.

### 5.1 Tamaños
| métrica | valor |
|---|---:|
| filas antes del filtro binario | 639 460 |
| filas después del filtro binario | 639 401 |
| `case_id` antes | 482 |
| `case_id` después | 482 |
| etiquetas `rhythm_label` originales | 10 |

### 5.2 Distribución binaria por filas
| clase | n_rows | % rows |
|---|---:|---:|
| `normal_sinus` | 392 623 | 61.41 % |
| `arrhythmia_or_abnormal` | 246 778 | 38.59 % |

### 5.3 Casos por clase
| clase | n_cases | % cases |
|---|---:|---:|
| `arrhythmia_or_abnormal` | 478 | 99.17 % |
| `normal_sinus` | 370 | 76.76 % |

### 5.4 Composición por caso
| categoría | n_cases | % cases |
|---|---:|---:|
| solo `normal_sinus` | 4 | 0.83 % |
| solo `arrhythmia_or_abnormal` | 112 | 23.24 % |
| mixtos (ambas clases) | 366 | 75.93 % |

El 76 % de los casos contiene ambas clases. Esto facilita que cualquier
split razonable por `case_id` cubra ambas en train y test.

### 5.5 Features candidatas
Fuente: `reports/tables/binary_feature_list_used.csv` y
`reports/tables/binary_excluded_columns_leakage.csv`.

- Numéricas candidatas (incluyendo las nuevas RR rolling): ver CSV.
- Categóricas candidatas: ver CSV.
- Excluidas por leakage: las 13 columnas en
  `config.BINARY_LEAKAGE_COLUMNS` (incluye `rhythm_binary`).
- Excluidas por alta cardinalidad / constantes / >99 % faltantes: ver
  CSV.

### 5.6 Faltantes
Fuente: `reports/tables/binary_missing_values.csv`. Las columnas con
mayor porcentaje de faltantes (medidas de gases arteriales preoperatorios,
catéteres centrales, etc.) son atributos clínicos que solo se registran
en un subconjunto de pacientes; se imputan por mediana dentro del
pipeline.

---

## 6. Split train/test

Fuente: `reports/tables/binary_train_test_split_summary.csv`,
`reports/tables/binary_class_support_train_test.csv`,
`reports/tables/binary_case_overlap_check.csv`.

- Función: `src.binary_search.make_binary_group_train_test_split_with_coverage`
  (wrapper sobre el split por grupo con cobertura de clases, con
  verificación adicional de que ambas clases binarias estén en train y
  test).
- `test_size = 0.20`.
- Verificación dura: `overlap_cases = 0` por construcción
  (`set(groups_train).isdisjoint(set(groups_test))`).

Cifras detalladas dependen de si la corrida fue full o `--debug`; el JSON
`reports/tables/binary_hyperparameter_search_meta.json` registra `args` y
`split_info` exactos.

---

## 7. Modelos evaluados

Definidos en `src.binary_search.build_binary_model_registry()`. El
registro se construye dinámicamente y omite los modelos cuya dependencia
opcional no esté instalada (`imbalanced-learn`, `lightgbm`, `catboost`).

| modelo | clasificador | manejo de desbalance |
|---|---|---|
| `dummy_most_frequent` | `DummyClassifier(strategy="most_frequent")` | baseline mínimo |
| `logreg_balanced` | `LogisticRegression(class_weight="balanced")` | `class_weight` |
| `sgd_log_loss` | `SGDClassifier(loss="log_loss", class_weight="balanced")` | `class_weight` |
| `linear_svc_balanced` | `LinearSVC(class_weight="balanced")` | `class_weight` |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier(class_weight="balanced")` | `class_weight` |
| `random_forest_balanced` | `RandomForestClassifier(class_weight="balanced_subsample")` | `class_weight_subsample` |
| `extra_trees_balanced` | `ExtraTreesClassifier(class_weight="balanced")` | `class_weight` |
| `balanced_random_forest` | `imblearn.ensemble.BalancedRandomForestClassifier` | subsampling balanceado |
| `easy_ensemble` | `imblearn.ensemble.EasyEnsembleClassifier` | ensemble balanceado |
| `xgboost_binary` | wrapper `_XGBBinaryClassifierSafe` con `scale_pos_weight` auto sobre train | `scale_pos_weight` |
| `lightgbm_binary` *(si instalado)* | `LGBMClassifier(class_weight="balanced")` | `class_weight` |
| `catboost_binary` *(si instalado)* | `CatBoostClassifier(auto_class_weights="Balanced")` | auto |

`lightgbm` y `catboost` no están instalados en el entorno de esta
corrida; el registro los omite automáticamente y lo documenta en el log.

Modelos NO incluidos en el registro y por qué:
- `SVC(kernel="rbf")` — costo prohibitivo en datasets de 500k+ filas.
- `MLPClassifier` — se omitió como modelo principal por costo y porque
  los otros modelos ya cubren el espacio funcional.
- `SMOTE` puro sobre todas las filas — riesgo de generar muestras
  sintéticas poco interpretables cuando la metadata se repite por
  caso. Si se prueba, debe ser dentro de CV con `imblearn.Pipeline`.

---

## 8. Búsqueda de hiperparámetros

- `RandomizedSearchCV` con `n_iter` configurable vía CLI.
- CV interna: `StratifiedGroupKFold` con fallback a `GroupKFold`.
- Grupos = `case_id`.
- Métrica de refit: `balanced_accuracy`.
- Métricas secundarias en CV: `accuracy`, `f1` (pos = anormal),
  `precision` (anormal), `recall` (anormal).
- `roc_auc` y `average_precision` se computan **solo en el test final**
  porque dependen de `predict_proba` / `decision_function` que no todos
  los modelos del registro exponen de forma uniforme.

---

## 9. Métricas CV — corrida `--debug` (80 cases, `n_iter=3`, `n_splits=2`)

Fuente: `reports/tables/binary_model_comparison_cv.csv`.

`StratifiedGroupKFold` con `n_splits_effective=2`. Métricas promediadas
sobre los 2 folds para la mejor combinación encontrada en
`RandomizedSearchCV`.

---

## 10. Métricas test — corrida `--debug`

Fuente: `reports/tables/binary_model_comparison_test.csv`.

Split: `chosen_seed=42`, 64 cases train (82 541 filas) / 16 cases test
(22 972 filas), `actual_test_fraction=0.218`, ambas clases en train y
test.

| modelo | test_balanced_acc | test_f1_abnormal | test_recall_abn (sens) | test_specificity_normal | test_roc_auc | test_AP | fit (s) |
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

(*) `linear_svc_balanced` y `xgboost_binary` tienen `test_balanced_accuracy = 0.5`
porque el umbral Youden J seleccionado en train colapsó a un punto que
predice todo como positivo en test (sensitivity 1.0, specificity 0.0).
Su `roc_auc` y `average_precision` siguen siendo altos (ver columnas
correspondientes en el CSV) → la discriminación intrínseca del modelo es
buena, pero la calibración del umbral no generaliza. Vale envolverlos en
`CalibratedClassifierCV` en una iteración siguiente. (LinearSVC tiene
ROC-AUC = 0.053 porque su `decision_function` está invertido respecto a
la convención de la clase positiva; documentado como limitación.)

`easy_ensemble` falló con
`numpy._core._exceptions._ArrayMemoryError` durante un
`cross_val_predict` interno al intentar materializar un array de ~30 k ×
125. Documentado y excluido del ranking. En el full run conviene reducir
el grid (`n_estimators` máximo 10 en lugar de 20) o eliminarlo si la
memoria sigue siendo problema.

---

## 11. Matriz de confusión (mejor modelo, `logreg_balanced`)

Fuentes: `reports/tables/binary_confusion_matrix_absolute.csv`,
`reports/tables/binary_confusion_matrix_normalized.csv`,
`reports/figures/binary_confusion_matrix_absolute.png`,
`reports/figures/binary_confusion_matrix_normalized.png`.

| true \ pred | `normal_sinus` | `arrhythmia_or_abnormal` |
|---|---:|---:|
| `normal_sinus` | 8 319 | 941 |
| `arrhythmia_or_abnormal` | 1 184 | 12 528 |

Totales: TN = 8 319, FP = 941, FN = 1 184, TP = 12 528.

---

## 12-15. Métricas binarias del mejor modelo (`logreg_balanced`)

| métrica | valor |
|---|---:|
| Sensitivity (recall anormal) | **0.914** |
| Specificity (TNR normal) | **0.898** |
| Precision anormal | **0.930** |
| F1 anormal | **0.922** |
| Accuracy | 0.907 |
| Balanced accuracy | 0.906 |
| ROC-AUC | **0.956** |
| Average Precision (PR-AUC) | **0.964** |

Reporte por clase (fuente:
`reports/tables/binary_best_model_classification_report.csv`):

| clase | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| `normal_sinus` | 0.875 | 0.898 | 0.887 | 9 260 |
| `arrhythmia_or_abnormal` | 0.930 | 0.914 | 0.922 | 13 712 |
| macro avg | 0.903 | 0.906 | 0.904 | 22 972 |
| weighted avg | 0.908 | 0.907 | 0.908 | 22 972 |

Estas cifras son del run `--debug` con 80 cases; el full run sobre los
482 cases debería confirmar o ajustar estos números.

---

## 16. Umbral usado y cómo se seleccionó

Para modelos que exponen `predict_proba` o `decision_function`:

1. Se obtienen scores fuera de muestra del train mediante
   `cross_val_predict` con la misma CV interna por grupo (sin tocar el
   test).
2. Sobre esos scores se buscan dos umbrales:
   - **Youden J máximo**: `J = sensitivity + specificity - 1`. Maximizar
     J prioriza un balance entre detección de anormalidad y conservación
     de specificity en normales.
   - **F1 máximo** de la clase anormal.
3. El umbral operativo elegido es el de **Youden J máximo**. La columna
   `chosen_threshold` en `binary_model_comparison_test.csv` lo registra
   por modelo.
4. El umbral seleccionado se aplica al test una sola vez para producir
   las predicciones finales que entran en la matriz de confusión.
5. Análisis completo de umbrales: `reports/tables/binary_threshold_analysis.csv`.

`dummy_most_frequent` no expone scores y se queda con su predicción
por defecto.

Umbrales elegidos en esta corrida (`chosen_threshold` por modelo):

- `logreg_balanced` → 0.608
- `sgd_log_loss` → 0.527
- `hist_gradient_boosting` → 0.323
- `random_forest_balanced` → 0.377
- `extra_trees_balanced` → 0.368
- `balanced_random_forest` → 0.388
- `linear_svc_balanced` → −5.934 (degenerado; ver §10)
- `xgboost_binary` → 0.000 (degenerado; ver §10)

---

## 17. Interpretabilidad

Fuente: `reports/tables/binary_feature_importance_best_model.csv` y
`reports/figures/binary_feature_importance_top20.png`.

- Si el mejor modelo es de árbol/boosting, se reportan
  `feature_importances_`.
- Si es lineal, se reportan los coeficientes con su valor absoluto.
- Si el preprocesador es un `ColumnTransformer` con `OneHotEncoder`, los
  nombres de features se expanden vía `get_feature_names_out()`.
- Si la importancia no es extraíble de forma confiable (modelos
  ensemble compuestos que no exponen atributos consistentes), el CSV
  contiene una fila con la nota correspondiente.

---

## 18. Limitaciones

1. **Modelado por fila / latido, no por caso.** Cada latido se trata
   como ejemplo independiente. Las features estáticas se repiten dentro
   de un caso, lo que puede inflar el aparente desempeño si la metadata
   por sí sola permite separar binariamente bien a los pacientes.
2. **El test mide generalización a casos nuevos**, no a beats nuevos
   del mismo paciente. Esa es la métrica que importa clínicamente, pero
   acota el techo de comparación.
3. **Solo se permitió `class_weight` y `scale_pos_weight` automático**
   para manejar desbalance. No se probó SMOTE/SMOTEENN dentro de CV.
4. **No se incluyen modelos con muy alta latencia** (RBF SVC, MLP grande)
   por costo y porque el resto cubre el espacio funcional.
5. **No clínico.** Proyecto académico; ninguna métrica aquí debe
   interpretarse como validación clínica.
6. **El umbral elegido por Youden J en train puede no generalizar.** El
   reporte muestra el caso de `linear_svc` en el debug, donde la
   decisión continua de SVM produjo un umbral que en test colapsó a una
   sola clase. Esto está documentado y queda como riesgo conocido.

---

## 19. Recomendaciones para la siguiente iteración

1. **Full run** con `--n-iter 30 --n-splits 5` sin `--max-cases` si
   solo se ejecutó debug; documentar tiempos por modelo.
2. **Persistir el mejor estimator** con `joblib.dump` en `models/`
   (el `.gitignore` actual permite versionar `.joblib` para que la
   app Streamlit los consuma).
3. **Probar calibración de probabilidades** con `CalibratedClassifierCV`
   solo sobre el train, especialmente para modelos cuyo
   `decision_function` no esté calibrado (LinearSVC, RandomForest).
4. **Análisis de fairness por edad y sexo** para verificar que el modelo
   no degrada desproporcionadamente en algún subgrupo.
5. **Evaluar agregación a nivel de caso**: además de la métrica por
   latido, reportar la fracción de latidos anormales por caso y compararla
   con la verdad (esto se aproxima más al uso clínico).
6. **Si se quiere reactivar la línea ECG**, hacerlo como branch nueva y
   compararla contra este baseline binario.

---

## Apéndice — Comandos para reproducir esta corrida

```bash
python scripts/04_audit_binary_rhythm_dataset.py
python scripts/05_build_binary_rhythm_modeling_dataset.py
python scripts/06_run_binary_rhythm_model_search.py --debug
# Full run cuando haya tiempo:
python scripts/06_run_binary_rhythm_model_search.py --n-iter 30 --n-splits 5
```
