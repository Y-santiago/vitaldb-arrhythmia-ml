# Model B Independent Binary Rhythm Pipeline

Modelo B es un flujo independiente para clasificar:

- Clase negativa: `normal_sinus`
- Clase positiva: `arrhythmia_or_abnormal`
- Target: `rhythm_binary`
- Split externo: 80/20 por `case_id`, sin overlap entre train y test

No usa ECG crudo, archivos `.npy`, `vitaldb.load_case`, multiclase, variables administrativas ni variables intraoperatorias como predictores.

## Que Archivo Correr Primero

Desde la raiz del repositorio:

```bash
python model_b_pipeline/audit_model_b_dataset.py
python model_b_pipeline/build_model_b_dataset.py
python model_b_pipeline/train_model_b.py --debug
python model_b_pipeline/evaluate_model_b.py
```

Que hace cada archivo:

- `audit_model_b_dataset.py`: revisa el parquet base y genera auditoria premodelo.
- `build_model_b_dataset.py`: crea `data/processed/model_b_dataset.parquet` con solo `case_id`, `rhythm_binary` y las 25 variables.
- `train_model_b.py`: compara modelos, busca hiperparametros, selecciona ganador por CV, evalua test una vez y guarda artefactos para app.
- `evaluate_model_b.py`: refresca los reportes Markdown a partir de tablas y JSON ya generados.
- `predict_model_b.py`: carga el modelo guardado y predice sobre un dataframe para usarlo en una app.
- `tests_model_b.py`: valida seguridad de features, split, busqueda, artefactos e inferencia.

## Dataset Base Requerido

El flujo espera:

```bash
data/processed/binary_rhythm_modeling_dataset.parquet
```

Si no existe, falla con un mensaje claro. Modelo B no reconstruye automaticamente flujos previos.

## Variables del Modelo

Se usan exactamente estas 25 variables originales:

```python
[
    "rr_prev",
    "rr_next",
    "hr_inst_from_rr_prev",
    "position_in_case",
    "rr_prev_rolling_mean_5",
    "rr_prev_rolling_std_5",
    "rr_prev_rolling_mean_20",
    "rr_prev_rolling_std_20",
    "rr_rmssd_5",
    "rr_rmssd_20",
    "rr_pnn50_5",
    "rr_pnn50_20",
    "local_hr_mean_5",
    "local_hr_mean_20",
    "age",
    "sex",
    "bmi",
    "asa",
    "preop_htn",
    "preop_dm",
    "preop_hb",
    "preop_na",
    "preop_k",
    "preop_gluc",
    "preop_cr",
]
```

`case_id` se usa solo para split y CV por grupo. Nunca entra como predictor.

## Este Flujo Hace Busqueda de Hiperparametros?

Si. `train_model_b.py` ejecuta `RandomizedSearchCV` por cada modelo candidato usando validacion cruzada por `case_id`. El ganador se selecciona por promedio de `balanced_accuracy` en CV sobre train, salvo que se pase otro `--selection-metric`. El test se evalua una sola vez al final para el ganador.

## Modelos que se Comparan

Por defecto:

- `dummy_most_frequent`
- `logreg_balanced`
- `sgd_log_loss`
- `hist_gradient_boosting`

Opcionales:

- `random_forest_balanced` con `--include-random-forest`
- `xgboost_binary` con `--include-xgboost`

Puedes filtrar modelos asi:

```bash
python model_b_pipeline/train_model_b.py --debug --models logreg_balanced sgd_log_loss
```

## Debug vs Full Run

Debug:

```bash
python model_b_pipeline/train_model_b.py --debug
```

Usa:

- `max_cases = 80`
- `n_iter = 3`
- `n_splits = 2`

Sirve para comprobar que todo corre. Sus metricas no son resultados finales.

Full light recomendado:

```bash
python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5 --models logreg_balanced sgd_log_loss hist_gradient_boosting
```

No incluye Random Forest por defecto para evitar tiempos largos.

## Donde Quedo el Modelo para la App?

Despues de entrenar con `--save-model` activo por defecto:

```bash
models/model_b/model_b_best_pipeline.joblib
models/model_b/model_b_feature_columns.json
models/model_b/model_b_metadata.json
models/model_b/model_b_threshold.json
```

El `.joblib` contiene preprocesamiento completo y modelo. La app no debe reconstruir imputadores, escaladores ni one-hot encoders.

Uso minimo:

```python
from model_b_pipeline.predict_model_b import predict_model_b_dataframe

predicciones = predict_model_b_dataframe(df_entrada)
```

Ver `reports/model_b/MODEL_B_APP_USAGE.md`.

## Correr desde VS Code

El repo incluye `.vscode/launch.json` con:

- `Model B - Audit`
- `Model B - Build dataset`
- `Model B - Train DEBUG`
- `Model B - Train FULL light`
- `Model B - Evaluate reports`
- `Model B - Tests`

Abre Run and Debug en VS Code y elige la configuracion.

## Outputs

Codigo fuente:

```bash
model_b_pipeline/
.vscode/launch.json
```

Reportes versionables:

```bash
reports/model_b/MODEL_B_REPORT.md
reports/model_b/NEXT_STEPS_MODEL_B.md
reports/model_b/MODEL_B_APP_USAGE.md
reports/model_b/figures/
```

Outputs generados e ignorados por `.gitignore`:

```bash
data/processed/model_b_dataset.parquet
reports/model_b/tables/*.csv
models/model_b/*
```

## Tests

```bash
pytest model_b_pipeline/tests_model_b.py
```

## Advertencia de Uso Clinico

Este flujo es de investigacion y auditoria. No esta validado para decisiones clinicas ni debe usarse como dispositivo diagnostico.
