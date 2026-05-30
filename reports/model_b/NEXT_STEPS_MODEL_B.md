# Next Steps - Model B

## Que se creo o actualizo

- Registro explicito de modelos en `build_model_b_registry()`.
- Busqueda de hiperparametros por modelo con `RandomizedSearchCV`.
- Seleccion del ganador por CV en train, usando `balanced_accuracy`.
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

- Debug: `True`
- Modelos exitosos: `sgd_log_loss, logreg_balanced, hist_gradient_boosting, dummy_most_frequent`
- Modelo ganador por CV: `sgd_log_loss`
- Balanced accuracy test final: `0.8882`
- Sensitivity anormal: `0.9137`
- Specificity normal: `0.8627`
- F1 anormal: `0.9108`

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
