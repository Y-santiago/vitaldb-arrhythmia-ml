# Model B Report - Binary Rhythm Classification

## 1. Objetivo

Crear un flujo independiente para clasificar `normal_sinus` vs `arrhythmia_or_abnormal` usando `rhythm_binary` como target, con split 80/20 por `case_id` y sin reactivar ECG crudo ni multiclase.

## 2. Justificacion del Modelo B

Modelo B reduce el pipeline binario amplio a 25 variables originales: dinamica RR/local, contexto clinico basal y cinco laboratorios preoperatorios. La meta es un modelo mas limpio, defendible e interpretable, no maximizar metricas con variables administrativas o intraoperatorias.

## 3. Variables Usadas

Se usaron exactamente estas 25 variables:

`rr_prev, rr_next, hr_inst_from_rr_prev, position_in_case, rr_prev_rolling_mean_5, rr_prev_rolling_std_5, rr_prev_rolling_mean_20, rr_prev_rolling_std_20, rr_rmssd_5, rr_rmssd_20, rr_pnn50_5, rr_pnn50_20, local_hr_mean_5, local_hr_mean_20, age, sex, bmi, asa, preop_htn, preop_dm, preop_hb, preop_na, preop_k, preop_gluc, preop_cr`

Agrupos generales: RR/temporales, clinicas basicas y laboratorios preoperatorios seleccionados.

## 4. Variables Excluidas

No se usaron `case_id`, `rhythm_binary`, `rhythm_label`, `beat_type`, identificadores, variables administrativas, desenlaces hospitalarios, texto diagnostico/procedimiento ni variables intraoperatorias. `case_id` solo se uso para el split por grupo.

## 5. Auditoria del Dataset

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

## 6. Split por `case_id`

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
| classes_train | arrhythmia_or_abnormal,normal_sinus |
| classes_test | arrhythmia_or_abnormal,normal_sinus |

Soporte por clase:

| rhythm_binary | train_rows | test_rows | train_cases_with_class | test_cases_with_class |
| --- | --- | --- | --- | --- |
| normal_sinus | 56003 | 9260 | 54 | 9 |
| arrhythmia_or_abnormal | 26538 | 13712 | 64 | 16 |

El archivo `model_b_case_overlap_check.csv` confirma ausencia de overlap entre train y test.

## 7. Modelos Evaluados

Modelos principales: `dummy_most_frequent`, `logreg_balanced`, `sgd_log_loss`, `hist_gradient_boosting`. `random_forest_balanced` queda disponible con `--include-random-forest`.

## 8. Hiperparametros

Busqueda con `RandomizedSearchCV`, CV interna por grupo (`StratifiedGroupKFold` con fallback a `GroupKFold`), `groups = case_id`, y `refit = balanced_accuracy`.

| model | param | value |
| --- | --- | --- |
| dummy_most_frequent | clf__strategy | most_frequent |
| logreg_balanced | clf__solver | lbfgs |
| logreg_balanced | clf__C | 0.01 |
| sgd_log_loss | clf__penalty | elasticnet |
| sgd_log_loss | clf__l1_ratio | 0.5 |
| sgd_log_loss | clf__alpha | 0.003 |
| hist_gradient_boosting | clf__min_samples_leaf | 20 |
| hist_gradient_boosting | clf__max_leaf_nodes | 15 |
| hist_gradient_boosting | clf__max_iter | 100 |
| hist_gradient_boosting | clf__learning_rate | 0.08 |
| hist_gradient_boosting | clf__l2_regularization | 0.1 |

## 9. Metricas CV

| model | best_cv_balanced_accuracy | cv_mean_accuracy | cv_mean_precision_abnormal | cv_mean_recall_abnormal_sensitivity | cv_mean_specificity_normal | cv_mean_f1_abnormal | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_log_loss | 0.8460 | 0.8374 | 0.7080 | 0.8674 | 0.8247 | 0.7759 | 6.0328 |
| logreg_balanced | 0.8444 | 0.8335 | 0.6999 | 0.8721 | 0.8166 | 0.7730 | 10.5669 |
| hist_gradient_boosting | 0.8437 | 0.8644 | 0.7838 | 0.7879 | 0.8995 | 0.7858 | 17.3237 |
| dummy_most_frequent | 0.5000 | 0.6795 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 6.6628 |

Modelo ganador por CV: `sgd_log_loss`.

## 10. Metricas Test

La seleccion del modelo ganador se hizo por CV en train. El test se evaluo despues.

| model | prediction_rule | balanced_accuracy | accuracy | precision_abnormal | recall_abnormal_sensitivity | specificity_normal | f1_abnormal | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_log_loss | default_predict | 0.8837 | 0.8913 | 0.8976 | 0.9234 | 0.8440 | 0.9103 | 0.9550 | 0.9605 |
| logreg_balanced | default_predict | 0.8827 | 0.8906 | 0.8963 | 0.9237 | 0.8417 | 0.9098 | 0.9558 | 0.9630 |
| hist_gradient_boosting | default_predict | 0.8706 | 0.8797 | 0.8849 | 0.9179 | 0.8232 | 0.9011 | 0.9315 | 0.9596 |
| dummy_most_frequent | default_predict | 0.5000 | 0.4031 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.5969 |

Metricas finales del pipeline persistido con el umbral elegido:

| metric | value |
|---|---:|
| balanced_accuracy | 0.8872 |
| accuracy | 0.8922 |
| precision_abnormal | 0.9070 |
| recall_abnormal_sensitivity | 0.9130 |
| specificity_normal | 0.8613 |
| f1_abnormal | 0.9100 |
| roc_auc | 0.9550 |
| average_precision | 0.9605 |

## 11. Matriz de Confusion

| true_label | normal_sinus | arrhythmia_or_abnormal |
| --- | --- | --- |
| normal_sinus | 7976 | 1284 |
| arrhythmia_or_abnormal | 1193 | 12519 |

Figuras:

- `reports/model_b/figures/model_b_confusion_matrix_absolute.png`
- `reports/model_b/figures/model_b_confusion_matrix_normalized.png`

## 12. Threshold Usado

Umbral final:

```json
{
  "name": "youden_j_train",
  "sensitivity": 0.9099781445474414,
  "specificity": 0.9257361212792172,
  "threshold": 0.5675159443727875,
  "youden_j": 0.8357142658266586
}
```

Analisis de umbrales guardado:

| model | threshold_name | threshold | threshold_selection_data | train_balanced_accuracy | train_f1_abnormal | test_balanced_accuracy | test_f1_abnormal | test_recall_abnormal_sensitivity | test_specificity_normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy_most_frequent | default_score_threshold | 0.5000 | train | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 |
| dummy_most_frequent | youden_j_train | 0.0000 | train | 0.5000 | 0.4866 | 0.5000 | 0.7476 | 1.0000 | 0.0000 |
| dummy_most_frequent | max_f1_train | 0.0000 | train | 0.5000 | 0.4866 | 0.5000 | 0.7476 | 1.0000 | 0.0000 |
| logreg_balanced | default_score_threshold | 0.5000 | train | 0.9163 | 0.8756 | 0.8827 | 0.9098 | 0.9237 | 0.8417 |
| logreg_balanced | youden_j_train | 0.5939 | train | 0.9182 | 0.8836 | 0.8840 | 0.9062 | 0.9052 | 0.8629 |
| logreg_balanced | max_f1_train | 0.6534 | train | 0.9165 | 0.8855 | 0.8875 | 0.9066 | 0.8980 | 0.8770 |
| sgd_log_loss | default_score_threshold | 0.5000 | train | 0.9164 | 0.8749 | 0.8837 | 0.9103 | 0.9234 | 0.8440 |
| sgd_log_loss | youden_j_train | 0.5675 | train | 0.9179 | 0.8806 | 0.8872 | 0.9100 | 0.9130 | 0.8613 |
| sgd_log_loss | max_f1_train | 0.6506 | train | 0.9158 | 0.8838 | 0.8882 | 0.9065 | 0.8959 | 0.8806 |
| hist_gradient_boosting | default_score_threshold | 0.5000 | train | 0.9569 | 0.9350 | 0.8706 | 0.9011 | 0.9179 | 0.8232 |
| hist_gradient_boosting | youden_j_train | 0.5224 | train | 0.9573 | 0.9368 | 0.8717 | 0.9014 | 0.9161 | 0.8273 |
| hist_gradient_boosting | max_f1_train | 0.6681 | train | 0.9540 | 0.9411 | 0.8738 | 0.8973 | 0.8945 | 0.8530 |

## 13. Comparacion contra Modelo Amplio Anterior

Existe una corrida previa del modelo binario amplio con 85 features candidatas registradas. Mejor fila disponible:

| model |
| --- |
| dummy_most_frequent |

La comparacion debe leerse con cuidado si aquella corrida fue `--debug` o uso un split distinto.

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
