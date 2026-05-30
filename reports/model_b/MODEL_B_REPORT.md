# Model B Report - Binary Rhythm Classification

## Resumen Ejecutivo

- Corrida debug: `True`
- Modelos solicitados: `dummy_most_frequent, logreg_balanced, sgd_log_loss, hist_gradient_boosting`
- Modelos exitosos: `sgd_log_loss, logreg_balanced, hist_gradient_boosting, dummy_most_frequent`
- Modelo ganador por CV: `sgd_log_loss`
- Metrica de seleccion: `balanced_accuracy`
- Valor CV de seleccion: `0.8457237536667869`
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

Si. `train_model_b.py` ejecuta `RandomizedSearchCV` por cada modelo candidato usando validacion cruzada por `case_id`. El ganador se selecciona por el promedio de `balanced_accuracy` en CV sobre train. El test no se usa para seleccionar modelo, hiperparametros, threshold ni semilla; se evalua una sola vez al final para el pipeline ganador.

## 4. Variables Usadas

Se usaron exactamente estas 25 variables originales:

`rr_prev, rr_next, hr_inst_from_rr_prev, position_in_case, rr_prev_rolling_mean_5, rr_prev_rolling_std_5, rr_prev_rolling_mean_20, rr_prev_rolling_std_20, rr_rmssd_5, rr_rmssd_20, rr_pnn50_5, rr_pnn50_20, local_hr_mean_5, local_hr_mean_20, age, sex, bmi, asa, preop_htn, preop_dm, preop_hb, preop_na, preop_k, preop_gluc, preop_cr`

`case_id` se usa solo para split/CV por grupo, nunca como predictor.

## 5. Variables Excluidas

No se usaron `case_id`, `rhythm_binary`, `rhythm_label`, `beat_type`, identificadores, variables administrativas, desenlaces hospitalarios, texto diagnostico/procedimiento ni variables intraoperatorias.

## 6. Auditoria del Dataset

| metric | value |
| --- | --- |
| source_dataset | data\processed\binary_rhythm_modeling_dataset.parquet |
| n_rows | 639401 |
| n_case_id | 482 |
| n_features_model_b | 25 |
| n_numeric_features | 24 |
| n_categorical_features | 1 |
| target_column | rhythm_binary |
| positive_class | arrhythmia_or_abnormal |
| negative_class | normal_sinus |
| all_features_present | True |
| forbidden_columns_in_features | none |
| additional_excluded_columns_in_features | none |

Distribucion por filas:

| rhythm_binary | n_rows | pct_rows |
| --- | --- | --- |
| normal_sinus | 392623 | 61.4048 |
| arrhythmia_or_abnormal | 246778 | 38.5952 |

Distribucion por casos:

| rhythm_binary | n_cases_with_at_least_one_row | pct_cases |
| --- | --- | --- |
| normal_sinus | 370 | 76.7635 |
| arrhythmia_or_abnormal | 478 | 99.1701 |

Composicion de casos:

| composition | n_cases | pct_cases |
| --- | --- | --- |
| mixed_normal_and_abnormal | 366 | 75.9336 |
| only_arrhythmia_or_abnormal | 112 | 23.2365 |
| only_normal_sinus | 4 | 0.8299 |

## 7. Split por `case_id`

| metric | value |
| --- | --- |
| chosen_seed | 42 |
| attempt | 1 |
| test_size_requested | 0.2 |
| actual_test_fraction_rows | 0.2177172481116071 |
| n_train_rows | 82541 |
| n_test_rows | 22972 |
| n_train_cases | 64 |
| n_test_cases | 16 |
| n_overlap_case_id | 0 |
| no_case_overlap | True |
| classes_train | arrhythmia_or_abnormal,normal_sinus |
| classes_test | arrhythmia_or_abnormal,normal_sinus |

Soporte por clase:

| rhythm_binary | train_rows | test_rows | train_cases_with_class | test_cases_with_class |
| --- | --- | --- | --- | --- |
| normal_sinus | 56003 | 9260 | 54 | 9 |
| arrhythmia_or_abnormal | 26538 | 13712 | 64 | 16 |

## 8. Modelos Comparados y Espacio de Busqueda

| model | description | hyperparameters |
| --- | --- | --- |
| dummy_most_frequent | Baseline DummyClassifier(strategy='most_frequent') | clf__strategy=['most_frequent'] |
| logreg_balanced | LogisticRegression(class_weight='balanced') | clf__C=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]; clf__solver=['lbfgs'] |
| sgd_log_loss | SGDClassifier(loss='log_loss', class_weight='balanced') | clf__alpha=[1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003]; clf__penalty=['l2', 'l1', 'elasticnet']; clf__l1_ratio=[0.15, 0.3, 0.5, 0.7] |
| hist_gradient_boosting | HistGradientBoostingClassifier | clf__learning_rate=[0.03, 0.05, 0.08, 0.1]; clf__max_iter=[100, 200, 300]; clf__max_leaf_nodes=[15, 31, 63]; clf__min_samples_leaf=[20, 50, 100]; clf__l2_regularization=[0.0, 0.01, 0.1, 1.0] |
| random_forest_balanced | RandomForestClassifier(class_weight='balanced_subsample') | clf__n_estimators=[100, 200]; clf__max_depth=[None, 10, 20]; clf__min_samples_leaf=[1, 5, 10]; clf__max_features=['sqrt', 'log2'] |
| xgboost_binary | Optional XGBoost with train-fold scale_pos_weight | clf__n_estimators=[100, 200]; clf__max_depth=[2, 3, 4]; clf__learning_rate=[0.03, 0.05, 0.1]; clf__subsample=[0.8, 1.0]; clf__colsample_bytree=[0.8, 1.0]; clf__reg_lambda=[0.1, 1.0, 3.0] |

`random_forest_balanced` solo corre con `--include-random-forest`. `xgboost_binary` solo corre con `--include-xgboost`.

## 9. Mejores Hiperparametros Encontrados

| model | param | value | is_winner |
| --- | --- | --- | --- |
| dummy_most_frequent | clf__strategy | most_frequent | False |
| logreg_balanced | clf__solver | lbfgs | False |
| logreg_balanced | clf__C | 0.01 | False |
| sgd_log_loss | clf__penalty | elasticnet | True |
| sgd_log_loss | clf__l1_ratio | 0.15 | True |
| sgd_log_loss | clf__alpha | 0.003 | True |
| hist_gradient_boosting | clf__min_samples_leaf | 20 | False |
| hist_gradient_boosting | clf__max_leaf_nodes | 15 | False |
| hist_gradient_boosting | clf__max_iter | 100 | False |
| hist_gradient_boosting | clf__learning_rate | 0.08 | False |
| hist_gradient_boosting | clf__l2_regularization | 0.1 | False |

## 10. Metricas CV

| model | status | best_params | selection_metric | best_cv_balanced_accuracy | elapsed_seconds | cv_mean_balanced_accuracy | cv_std_balanced_accuracy | cv_mean_accuracy | cv_std_accuracy | cv_mean_precision_abnormal | cv_std_precision_abnormal | cv_mean_recall_abnormal_sensitivity | cv_std_recall_abnormal_sensitivity | cv_mean_specificity_normal | cv_std_specificity_normal | cv_mean_f1_abnormal | cv_std_f1_abnormal | cv_mean_roc_auc | cv_std_roc_auc | cv_mean_average_precision | cv_std_average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_log_loss | ok | {'clf__penalty': 'elasticnet', 'clf__l1_ratio': 0.15, 'clf__alpha': 0.003} | balanced_accuracy | 0.8457 | 6.7342 | 0.8457 | 0.0391 | 0.8364 | 0.0513 | 0.7051 | 0.1000 | 0.8692 | 0.0068 | 0.8223 | 0.0714 | 0.7751 | 0.0640 | 0.8645 | 0.0792 | 0.6845 | 0.1915 |
| logreg_balanced | ok | {'clf__solver': 'lbfgs', 'clf__C': 0.01} | balanced_accuracy | 0.8444 | 10.6550 | 0.8444 | 0.0427 | 0.8335 | 0.0551 | 0.6999 | 0.1038 | 0.8721 | 0.0100 | 0.8166 | 0.0753 | 0.7730 | 0.0681 | 0.8648 | 0.0829 | 0.6869 | 0.1974 |
| hist_gradient_boosting | ok | {'clf__min_samples_leaf': 20, 'clf__max_leaf_nodes': 15, 'clf__max_iter': 100, 'clf__learning_rate': 0.08, 'clf__l2_regularization': 0.1} | balanced_accuracy | 0.8437 | 31.8656 | 0.8437 | 0.0441 | 0.8644 | 0.0338 | 0.7838 | 0.0577 | 0.7879 | 0.0686 | 0.8995 | 0.0196 | 0.7858 | 0.0631 | 0.9228 | 0.0389 | 0.8402 | 0.0825 |
| dummy_most_frequent | ok | {'clf__strategy': 'most_frequent'} | balanced_accuracy | 0.5000 | 6.5093 | 0.5000 | 0.0000 | 0.6795 | 0.0136 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.3205 | 0.0136 |

## 11. Metricas Test del Ganador

La tabla de test contiene el ganador seleccionado por CV. No es una tabla para elegir modelo.

| model | prediction_rule | accuracy | balanced_accuracy | precision_abnormal | recall_abnormal_sensitivity | specificity_normal | f1_abnormal | n_tn | n_fp | n_fn | n_tp | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_log_loss | youden_j_train | 0.8931 | 0.8882 | 0.9079 | 0.9137 | 0.8627 | 0.9108 | 7989 | 1271 | 1184 | 12528 | 0.9558 | 0.9623 |

Metricas finales del pipeline persistido:

| metric | value |
|---|---:|
| balanced_accuracy | 0.8882 |
| accuracy | 0.8931 |
| precision_abnormal | 0.9079 |
| recall_abnormal_sensitivity | 0.9137 |
| specificity_normal | 0.8627 |
| f1_abnormal | 0.9108 |
| roc_auc | 0.9558 |
| average_precision | 0.9623 |

## 12. Matriz de Confusion

| true_label | normal_sinus | arrhythmia_or_abnormal |
| --- | --- | --- |
| normal_sinus | 7989 | 1271 |
| arrhythmia_or_abnormal | 1184 | 12528 |

Figuras:

- `reports/model_b/figures/model_b_confusion_matrix_absolute.png`
- `reports/model_b/figures/model_b_confusion_matrix_normalized.png`

## 13. Threshold Usado

Archivo: `models/model_b/model_b_threshold.json`

```json
{
  "method": "youden_j_train",
  "negative_class": "normal_sinus",
  "positive_class": "arrhythmia_or_abnormal",
  "score_kind": "predict_proba_or_decision_function",
  "sensitivity": 0.9094129173260984,
  "specificity": 0.927164616181276,
  "threshold": 0.5712766861989581,
  "youden_j": 0.8365775335073744
}
```

Analisis de thresholds sobre train:

| model | threshold_name | threshold | threshold_selection_data | train_balanced_accuracy | train_f1_abnormal | train_recall_abnormal_sensitivity | train_specificity_normal | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy_most_frequent | default | 0.5000 | train | 0.5000 | 0.0000 | 0.0000 | 1.0000 | threshold selected on train only |
| dummy_most_frequent | youden_j_train | 0.0000 | train | 0.5000 | 0.4866 | 1.0000 | 0.0000 | threshold selected on train only |
| dummy_most_frequent | max_f1_train | 0.0000 | train | 0.5000 | 0.4866 | 1.0000 | 0.0000 | threshold selected on train only |
| logreg_balanced | default | 0.5000 | train | 0.9163 | 0.8756 | 0.9165 | 0.9162 | threshold selected on train only |
| logreg_balanced | youden_j_train | 0.5939 | train | 0.9182 | 0.8836 | 0.9035 | 0.9329 | threshold selected on train only |
| logreg_balanced | max_f1_train | 0.6534 | train | 0.9165 | 0.8855 | 0.8895 | 0.9434 | threshold selected on train only |
| sgd_log_loss | default | 0.5000 | train | 0.9166 | 0.8753 | 0.9188 | 0.9144 | threshold selected on train only |
| sgd_log_loss | youden_j_train | 0.5713 | train | 0.9183 | 0.8816 | 0.9094 | 0.9272 | threshold selected on train only |
| sgd_log_loss | max_f1_train | 0.6654 | train | 0.9160 | 0.8852 | 0.8882 | 0.9438 | threshold selected on train only |
| hist_gradient_boosting | default | 0.5000 | train | 0.9569 | 0.9350 | 0.9558 | 0.9580 | threshold selected on train only |
| hist_gradient_boosting | youden_j_train | 0.5224 | train | 0.9573 | 0.9368 | 0.9533 | 0.9612 | threshold selected on train only |
| hist_gradient_boosting | max_f1_train | 0.6681 | train | 0.9540 | 0.9411 | 0.9300 | 0.9780 | threshold selected on train only |

## 14. Donde Quedo el Modelo para la App?

- Pipeline completo: `models/model_b/model_b_best_pipeline.joblib`
- Columnas requeridas: `models/model_b/model_b_feature_columns.json`
- Metadata: `models/model_b/model_b_metadata.json`
- Threshold operativo: `models/model_b/model_b_threshold.json`

La app debe cargar el `.joblib`; ese pipeline ya incluye imputacion, escalado, one-hot encoding y clasificador. La app no debe reconstruir preprocesamiento por separado.

Ejemplo:

```python
from model_b_pipeline.predict_model_b import predict_model_b_dataframe

predicciones = predict_model_b_dataframe(df_entrada)
```

Ver tambien `reports/model_b/MODEL_B_APP_USAGE.md`.

## 15. Modelos Fallidos

_No disponible en esta corrida._

## 16. Comparacion contra Modelo Amplio Anterior

Existe una corrida previa del modelo binario amplio con 85 features candidatas registradas. Mejor fila disponible:

| model | status |
| --- | --- |
| dummy_most_frequent | error |

La comparacion debe leerse con cuidado si aquella corrida fue `--debug` o uso un split distinto. Para una comparacion estricta, repetir ambos flujos con el mismo split.

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
