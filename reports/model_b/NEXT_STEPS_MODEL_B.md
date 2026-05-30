# Next Steps - Model B

## Que se creo

- Carpeta independiente `model_b_pipeline/` con configuracion, auditoria, builder, entrenamiento, evaluacion, utilidades y tests.
- Dataset reducido local `data/processed/model_b_dataset.parquet` con `case_id`, `rhythm_binary` y las 25 features de Modelo B.
- Reportes locales en `reports/model_b/`.
- Artefactos locales en `models/model_b/`: pipeline `.joblib`, columnas y metadata.

## Scripts para correr

```bash
python model_b_pipeline/audit_model_b_dataset.py
python model_b_pipeline/build_model_b_dataset.py
python model_b_pipeline/train_model_b.py --debug
python model_b_pipeline/evaluate_model_b.py
```

Corrida full inicial:

```bash
python model_b_pipeline/train_model_b.py --n-iter 20 --n-splits 5
python model_b_pipeline/evaluate_model_b.py
```

## Resultados de esta corrida

- Debug: `True`
- Modelo ganador por CV: `sgd_log_loss`
- Balanced accuracy test final: `0.8872`
- Sensitivity anormal: `0.9130`
- Specificity normal: `0.8613`
- F1 anormal: `0.9100`

## Que fallo

Si este archivo existe, `evaluate_model_b.py` pudo leer la metadata de entrenamiento. Revisar la salida de consola o los tests si alguna tabla/figura esperada falta.

## Que falta

- Correr full run si esta corrida fue debug.
- Revisar si el umbral Youden J es operativo o si conviene fijar sensibilidad minima.
- Comparar Modelo B contra el modelo amplio anterior con el mismo split si se necesita una conclusion metodologica fuerte.

## Recomendacion para la siguiente iteracion

Mantener Modelo B como baseline interpretable de 25 variables. La siguiente mejora deberia enfocarse en validacion y calibracion, no en agregar variables de leakage o intraoperatorias al modelo principal.
