# Model B App Usage

## Archivos que usa la app

Despues de entrenar, la app debe leer:

- `models/model_b/model_b_best_pipeline.joblib`
- `models/model_b/model_b_feature_columns.json`
- `models/model_b/model_b_metadata.json`
- `models/model_b/model_b_threshold.json`

El `.joblib` ya contiene preprocesamiento completo y clasificador. No reconstruyas imputadores, escaladores ni one-hot encoders en la app.

## Ejemplo minimo

```python
import pandas as pd
from model_b_pipeline.predict_model_b import predict_model_b_dataframe

df = pd.read_parquet("data/processed/model_b_dataset.parquet")
pred = predict_model_b_dataframe(df)
print(pred.head())
```

## Columnas de entrada requeridas

El dataframe de entrada debe contener estas 25 columnas:

```python
[
    "rr_prev", "rr_next", "hr_inst_from_rr_prev", "position_in_case",
    "rr_prev_rolling_mean_5", "rr_prev_rolling_std_5",
    "rr_prev_rolling_mean_20", "rr_prev_rolling_std_20",
    "rr_rmssd_5", "rr_rmssd_20", "rr_pnn50_5", "rr_pnn50_20",
    "local_hr_mean_5", "local_hr_mean_20", "age", "sex", "bmi",
    "asa", "preop_htn", "preop_dm", "preop_hb", "preop_na",
    "preop_k", "preop_gluc", "preop_cr",
]
```

Puede tener columnas extra; la utilidad las ignora.

## Salida

`predict_model_b_dataframe(df)` devuelve:

- `model_b_score_arrhythmia_or_abnormal`
- `model_b_prediction`
- `model_b_prediction_default_threshold`
- `model_b_threshold_used`
- `model_b_threshold_method`

Si faltan columnas, lanza un `ValueError` con la lista exacta de columnas faltantes.
