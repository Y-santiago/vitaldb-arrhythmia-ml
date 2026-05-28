<!--
Este reporte se actualiza con los CSVs reales que produce
`scripts/03_run_tabular_hyperparameter_search.py`. Si se ejecuta en modo
`--debug`, los números reportados aquí corresponden al subconjunto de
casos del debug y NO al dataset completo.
-->

# Informe de modelado tabular — VitalDB Arrhythmia ML

**Fecha del reporte:** 2026-05-27
**Rama:** `main`

---

## 1. Resumen ejecutivo

Pivot metodológico de esta iteración: el modelado se hace sobre datos
tabulares filtrados (anotaciones + metadata + features temporales locales
por latido), sin ECG crudo. El flujo basado en señal ECG queda como
línea exploratoria histórica.

Estado del pipeline:

1. ✅ Audit del dataset (`scripts/01_audit_filtered_tabular_dataset.py`).
2. ✅ Construcción del dataset modelable
   (`scripts/02_build_filtered_tabular_modeling_dataset.py`).
3. ✅ Split 80/20 por `case_id` con cobertura
   (`make_train_test_group_split_with_coverage`).
4. ✅ Preprocesador tabular `ColumnTransformer`
   (`src/preprocessing.build_tabular_preprocessor`).
5. ✅ Pipeline `RandomizedSearchCV` multi-modelo
   (`src/tabular_search.py` + `scripts/03_run_tabular_hyperparameter_search.py`).
6. ✅ Evaluación, reportes y figuras (`reports/tables/` y `reports/figures/`).
7. ✅ Corrida `--debug` (60 cases, n_iter=3, n_splits=2) ejecutada
   end-to-end. **Pendiente:** full run (`--n-iter 30 --n-splits 5`) sobre
   los 482 cases.

El número exacto de iteraciones (`--n-iter`) usado en cada corrida queda
registrado en `reports/tables/tabular_hyperparameter_search_meta.json`.
Cualquier comparación entre corridas debe consultar ese archivo.

> **Cifras reportadas en este informe.** Las tablas y la matriz de
> confusión de las secciones 6–9 provienen del run `--debug` con
> `max_cases=60, n_iter=3, n_splits=2`. **No son los números finales**;
> son la prueba de que el pipeline corre extremo a extremo. El full run
> sobre los 482 cases producirá cifras distintas (esperablemente más
> altas y más estables).

---

## 2. Datos usados

### 2.1 Origen
- Anotaciones: `data/raw/physionet_annotations/Annotation_Files/Annotation_file_<case_id>.csv`
  (482 archivos en disco).
- Metadata: `data/raw/physionet_annotations/metadata.csv` (482 filas × 79 columnas).
- **No se usa señal ECG cruda.** El módulo `src/download.py` y los
  notebooks `03`–`05` ECG están marcados como `legacy`.

### 2.2 Resultados de la auditoría
Fuente: `reports/tables/tabular_dataset_audit.csv` y
`reports/tables/tabular_class_distribution.csv`.

| métrica | valor |
|---|---:|
| filas antes de filtros | 676 250 |
| filas después de filtros | 639 460 |
| cases antes de filtros | 482 |
| cases después de filtros | 482 |
| n clases `rhythm_label` | 10 |
| columnas totales en el merge | 85 |
| columnas numéricas candidatas | 54 |
| columnas categóricas candidatas | 17 |
| columnas excluidas por leakage | 11 |
| columnas excluidas por alta cardinalidad | 3 |

Distribución global de clases (fuente:
`reports/tables/tabular_class_distribution.csv`):

| `rhythm_label` | filas | % filas | cases con la clase | % cases |
|---|---:|---:|---:|---:|
| N | 392 623 | 61.40 % | 370 | 76.76 % |
| AFIB/AFL | 158 480 | 24.78 % | 111 | 23.03 % |
| Patterned Ventricular Ectopy | 23 902 | 3.74 % | 109 | 22.61 % |
| SND | 22 224 | 3.48 % | 66 | 13.69 % |
| Patterned Atrial Ectopy | 19 946 | 3.12 % | 85 | 17.64 % |
| WAP/MAT | 10 047 | 1.57 % | 25 | 5.19 % |
| SVTA | 6 413 | 1.00 % | 109 | 22.61 % |
| AVB | 4 193 | 0.66 % | 10 | 2.08 % |
| VT | 1 573 | 0.25 % | 87 | 18.05 % |
| Unclassifiable | 59 | 0.01 % | 5 | 1.04 % |

### 2.3 Filtros aplicados
1. Exclusión de la clase `Noise` (`EXCLUDED_RHYTHM_LABELS`).
2. Exclusión de filas con `bad_signal_quality=True`.
3. Exclusión de filas con `rhythm_label` nulo, `"nan"`, `"none"` o vacío.

### 2.4 Features derivadas por fila (dentro del caso)
- `rr_prev` — intervalo (s) hasta el latido previo.
- `rr_next` — intervalo (s) hasta el latido siguiente.
- `hr_inst_from_rr_prev` — frecuencia instantánea (bpm) a partir de `rr_prev`.
- `position_in_case` — posición relativa entre 0 y 1 dentro de la
  duración del caso.

### 2.5 Coerciones de tipo
- `age` venía como string porque algunas filas usaban `>89` para
  anonimizar pacientes ≥ 89 años. Se coercionó a numérico tratando
  `>89` como `89.0` (afectó 4 028 filas).

---

## 3. Split train/test

Fuente: `reports/tables/tabular_train_test_split_summary.csv` y
`reports/tables/tabular_class_support_train_test.csv`.

- Función: `make_train_test_group_split_with_coverage`
  (`GroupShuffleSplit` con búsqueda de cobertura de clases sobre 200
  semillas).
- `test_size = 0.20`.

Sobre el **dataset completo** (482 cases) con `random_state=42`:

| campo | valor |
|---|---|
| chosen_seed | 42 |
| n_classes_covered | 10 / 10 |
| actual_test_fraction | 0.202 |
| n_train_groups | 385 |
| n_test_groups | 97 |
| n_train_rows | 510 287 |
| n_test_rows | 129 173 |
| classes_only_in_train | (ninguna) |
| classes_only_in_test | (ninguna) |

Para el debug a 60 cases (corrida documentada en este reporte) las cifras
están en `reports/tables/tabular_train_test_split_summary.csv` y pueden
diferir; el debug es solo para validar el pipeline.

---

## 4. Modelos evaluados

Definidos en `src/tabular_search.py::TABULAR_PARAM_DISTRIBUTIONS`.
Pipeline = `ColumnTransformer(Imputer+Scaler / Imputer+OneHotEncoder)`
→ clasificador con `class_weight="balanced"` cuando aplica.

| modelo | clasificador | manejo de desbalance |
|---|---|---|
| `logreg` | `LogisticRegression(solver="lbfgs", max_iter=3000)` | `class_weight="balanced"` |
| `decision_tree` | `DecisionTreeClassifier` | `class_weight="balanced"` |
| `random_forest` | `RandomForestClassifier(n_jobs=-1)` | `class_weight="balanced"` |
| `xgboost` | `XGBClassifier` envuelto en `_XGBClassifierSafe` (re-encoding por fold) | sin `class_weight` (XGB multiclase no lo soporta nativamente) |
| `linear_svc` | `LinearSVC(max_iter=5000, dual="auto")` | `class_weight="balanced"` |
| `mlp` | `MLPClassifier(max_iter=200, early_stopping=False)` | (sin balanceo nativo; vía OHE+scale) |

---

## 5. Estrategia de búsqueda

- `RandomizedSearchCV` con `n_iter` configurable vía CLI.
- Scoring multimétrica (refit por `f1_macro`):
  `f1_macro`, `precision_macro`, `recall_macro`, `accuracy`,
  `balanced_accuracy`, `f1_weighted`.
- CV interna por grupo: `StratifiedGroupKFold` cuando es viable;
  `GroupKFold` como fallback. `n_splits` configurable (recortado al
  número de grupos disponibles).
- `groups = case_id` en `RandomizedSearchCV.fit`.
- Test congelado: se evalúa una sola vez al final.

---

## 6. Resultados globales y por clase (corrida `--debug` con 60 cases)

Cifras reales producidas por la corrida documentada en
`reports/tables/tabular_hyperparameter_search_meta.json`
(`max_cases=60`, `n_iter=3`, `n_splits=2`).

Split de la corrida:
- `chosen_seed=82` (tras explorar 200 semillas, por ausencia de la clase
  `Unclassifiable` que solo aparece en 5 cases del dataset completo).
- 48 cases en train / 12 en test (`63 204` / `16 085` filas).
- 9 / 10 clases cubiertas. `Unclassifiable` quedó solo en train.

### 6.1 Comparativa por modelo (test)
Fuente: `reports/tables/tabular_model_comparison_test.csv`.

| modelo | test_f1_macro | test_balanced_acc | test_recall_macro | test_precision_macro | test_accuracy |
|---|---:|---:|---:|---:|---:|
| logreg | 0.151 | 0.272 | 0.272 | 0.172 | 0.412 |
| decision_tree | 0.078 | 0.112 | 0.112 | 0.101 | 0.166 |
| random_forest | 0.144 | 0.157 | 0.157 | 0.165 | 0.566 |
| xgboost | 0.085 | 0.113 | 0.113 | 0.174 | 0.560 |
| **linear_svc** | **0.189** | **0.356** | **0.356** | **0.321** | 0.354 |
| mlp | 0.126 | 0.126 | 0.126 | 0.209 | 0.344 |

### 6.2 Comparativa CV (mejor combinación por modelo)
Fuente: `reports/tables/tabular_model_comparison_cv.csv`.

| modelo | cv_f1_macro | cv_balanced_acc | cv_accuracy | fit_seconds |
|---|---:|---:|---:|---:|
| logreg | 0.065 | 0.066 | 0.348 | 82.8 |
| decision_tree | 0.128 | 0.148 | 0.466 | 12.9 |
| random_forest | 0.107 | 0.127 | 0.566 | 57.7 |
| xgboost | 0.101 | 0.129 | 0.497 | 97.4 |
| linear_svc | 0.075 | 0.130 | 0.247 | 149.6 |
| mlp | 0.101 | 0.128 | 0.489 | 83.7 |

**Lectura:** RandomForest y XGBoost tienen la mejor `accuracy` (≈ 56 %)
pero `f1_macro` mediocre porque predicen mayoritariamente la clase
dominante. LinearSVC tiene la mejor `f1_macro` y `recall_macro` porque
con `class_weight="balanced"` predice activamente clases minoritarias,
sacrificando precision en algunas (ver sección 9).

---

## 7. Mejor modelo por ventana

En el flujo tabular **no hay tamaño de ventana**: cada fila es un latido,
no un segmento de señal. Esta sección es N/A para el flujo activo y se
deja documentada explícitamente para evitar confusión con el reporte
legacy `MODELING_REPORT.md`.

---

## 8. Mejor modelo general (corrida `--debug`)

**Modelo:** `linear_svc`
**Métrica principal en test:** `test_f1_macro = 0.189`

Hiperparámetros elegidos:
```json
{"clf__C": 56.69849511478853}
```

Métricas completas en test (fuente:
`reports/tables/tabular_best_model_classification_report.csv`):

| clase | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| AFIB/AFL | 0.405 | 0.843 | 0.547 | 4 070 |
| AVB | 0.000 | 0.000 | 0.000 | 465 |
| N | 0.665 | 0.141 | 0.233 | 9 075 |
| Patterned Atrial Ectopy | 0.000 | 0.000 | 0.000 | 362 |
| Patterned Ventricular Ectopy | 0.371 | 0.211 | 0.269 | 1 295 |
| SND | 0.445 | 0.999 | 0.616 | 699 |
| SVTA | 0.006 | 1.000 | 0.012 | 19 |
| VT | 1.000 | 0.012 | 0.023 | 85 |
| WAP/MAT | 0.000 | 0.000 | 0.000 | 15 |
| **macro avg** | 0.321 | 0.356 | **0.189** | 16 085 |
| weighted avg | 0.532 | 0.354 | 0.318 | 16 085 |
| accuracy | — | — | 0.354 | 16 085 |

`Unclassifiable` no aparece en el reporte porque su soporte en test es 0.

---

## 9. Matriz de confusión (corrida `--debug`)

Fuente: `reports/tables/tabular_confusion_matrix_absolute.csv`,
`reports/figures/tabular_best_model_confusion_matrix_absolute.png` y
`reports/figures/tabular_best_model_confusion_matrix_normalized.png`.

Filas = clase real (con `support_true`), columnas = clase predicha:

| true \ pred | AFIB/AFL | AVB | N | PAE | PVE | SND | SVTA | VT | WAP/MAT | sup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AFIB/AFL | 3 429 | 0 | 635 | 1 | 5 | 0 | 0 | 0 | 0 | 4 070 |
| AVB | 465 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 465 |
| N | 3 479 | 118 | 1 280 | 1 | 457 | 869 | 2 869 | 0 | 2 | 9 075 |
| PAE | 362 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 362 |
| PVE | 716 | 182 | 0 | 0 | 273 | 0 | 123 | 0 | 1 | 1 295 |
| SND | 0 | 0 | 0 | 0 | 0 | 698 | 1 | 0 | 0 | 699 |
| SVTA | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 19 |
| VT | 0 | 0 | 9 | 0 | 0 | 1 | 74 | 1 | 0 | 85 |
| WAP/MAT | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| **pred_total** | 8 466 | 300 | 1 924 | 2 | 735 | 1 568 | 3 086 | 1 | 3 | 16 085 |

(PAE = Patterned Atrial Ectopy; PVE = Patterned Ventricular Ectopy.)

Lecturas:

- **AFIB/AFL** es la clase mejor identificada (recall 0.84).
- **N** está fuertemente sub-predicha (recall 0.14): muchos `N` se
  confunden con `AFIB/AFL`, `SVTA` y `SND` porque LinearSVC está
  balanceando agresivamente.
- **SND** es casi perfectamente recuperada (recall 1.00 en 699 muestras).
- **SVTA** tiene precision ínfima (19 reales vs 3 086 predichas): el
  modelo lo usa como clase “catch-all”.
- **AVB, Patterned Atrial Ectopy, WAP/MAT, VT** quedan prácticamente sin
  predecir correctamente; con tan pocos cases en el debug, esto es
  esperable y desaparecerá parcialmente con el full run.

---

## 10. Interpretación objetiva

Antes de leer cualquier `test_f1_macro` global, mirar las tablas de
soporte por clase y de clases ausentes por split
(`tabular_class_support_train_test.csv`,
`tabular_train_test_split_summary.csv`):

- `N` y `AFIB/AFL` dominan el conteo absoluto (61 % + 25 %).
- `AVB`, `WAP/MAT`, `Unclassifiable` y `VT` son clases minoritarias con
  alto riesgo de bajo desempeño individual aun con `class_weight="balanced"`.
- `f1_macro` da igual peso a todas las clases independientemente del
  soporte; un modelo que predice solo `N` tendría `f1_macro` muy bajo
  aunque su `accuracy` sea ~61 %.

Cualquier comparación entre modelos debe basarse en `f1_macro` y en
la matriz de confusión absoluta, no en `accuracy`.

---

## 11. Limitaciones

1. **Modelado por fila / latido, no por caso.** Cada latido se trata
   como ejemplo independiente. La estructura temporal *dentro* del caso
   se representa solo vía `rr_prev`, `rr_next`, `hr_inst_from_rr_prev`,
   `position_in_case`.
2. **Metadata estática repetida por caso.** Como cada caso tiene
   demográficos y metadatos fijos, todas sus filas comparten esos
   valores. El modelo puede aprender efectivamente patrones a nivel
   caso. El split por `case_id` evita la fuga directa.
3. **Cardinalidad alta omitida.** `dx`, `opname`, `age` (cuando era
   string) y otras de alta cardinalidad quedaron fuera del set de
   features. La información clínica de `dx` y `opname` no se aprovecha
   en este baseline.
4. **Faltantes altos en algunos preoperatorios** (gases arteriales,
   `lmasize`, `cline2`, etc.). El `SimpleImputer(median)` los reemplaza
   pero introduce sesgo cuando el patrón de faltantes está correlacionado
   con la clase.
5. **Sin filtrado de señal / sin features morfológicas** porque no se usa
   el ECG crudo. Si en una iteración futura se reactiva la línea ECG,
   esas features podrían complementar.
6. **No clínico.** Proyecto académico; no validado para uso médico.

---

## 12. Recomendaciones para la siguiente fase

1. Decidir si se reintroduce la línea ECG crudo como complemento (no
   reemplazo) del flujo tabular, ahora que el tabular sirve de baseline
   honesto.
2. Probar codificación de `dx` / `opname` con técnicas que no exploten
   (target encoding por fold dentro de la CV, o agrupamiento por
   palabras clave).
3. Probar manejo de desbalance adicional: `imbalanced-learn` (SMOTE,
   SMOTEENN) o `BalancedRandomForest`.
4. Persistir el mejor modelo a disco con `joblib.dump` en `models/`
   (ignorado por git).
5. Reportar varianza entre folds en lugar de solo el promedio.
6. Explorar features temporales adicionales sin tocar la señal cruda:
   ventanas locales de RR (mean / std / RMSSD de los últimos N latidos
   por caso).
