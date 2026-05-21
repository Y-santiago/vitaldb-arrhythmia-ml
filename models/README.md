# Modelos entrenados — VitalDB Arrhythmia ML

Artefactos generados por `notebooks/06_full_modeling_hyperparameter_search.ipynb`
sobre `data/processed/features_baseline.parquet`.

---

## Modelo ganador

**`linear_svc`** — `test_f1_macro = 0.3439`

Entrenado el 2026-05-21. Corrida FAST_MODE (N_ITER=5, N_SPLITS=3).
Para resultados finales ejecutar con `FAST_MODE=False` (N_ITER=30).

---

## Archivos incluidos

| Archivo | Tamaño | Descripción |
|---|---|---|
| `best_model_pipeline.joblib` | ~5 KB | Pipeline del **ganador global** (LinearSVC). Apunta al mismo objeto que `linear_svc_best_pipeline.joblib`. |
| `linear_svc_best_pipeline.joblib` | ~5 KB | LinearSVC: mejor por `test_f1_macro`. |
| `decision_tree_best_pipeline.joblib` | ~2.6 MB | Árbol de decisión. |
| `random_forest_best_pipeline.joblib` | ~10 MB | Random Forest (100 estimadores). |
| `xgboost_best_pipeline.joblib` | ~8 MB | XGBoost (400 estimadores). |
| `mlp_best_pipeline.joblib` | ~140 KB | Red neuronal MLP (64-32). |
| `feature_columns.json` | ~1 KB | Lista exacta de las 26 features de entrada. Necesario para reproducibilidad. |
| `model_artifacts_metadata.json` | ~4 KB | Metadata completa: hiperparámetros, versiones de librerías, fechas, rutas. |

---

## Resultados del entrenamiento (FAST_MODE)

| Modelo | test_f1_macro | test_accuracy | fit_time |
|---|---|---|---|
| **linear_svc** | **0.3439** | 0.8061 | 185 s |
| random_forest | 0.3225 | 0.7879 | 1467 s |
| mlp | 0.3065 | 0.8204 | 561 s |
| xgboost | 0.2739 | 0.8203 | 640 s |
| decision_tree | 0.2332 | 0.7220 | 39 s |

---

## Reglas metodológicas

- **Target**: `rhythm_label` (multiclase, 10 clases de arritmias).
- **Split**: 80/20 por `case_id` — sin leakage entre pacientes.
- **CV interna**: `StratifiedGroupKFold` por `case_id`.
- **beat_type**: prohibido como predictor.
- **case_id**: solo usado como grupo, nunca como feature.
- **Test**: evaluado una sola vez al final del tuning.
- **Features**: 26 columnas numéricas de señal ECG + intervalos RR.

---

## Cómo cargar el modelo ganador

```python
import joblib
import json
import pandas as pd

# Cargar el modelo ganador
model = joblib.load("models/best_model_pipeline.joblib")

# Cargar la lista exacta de features
with open("models/feature_columns.json") as f:
    feature_cols = json.load(f)

# Construir X desde el dataframe de features
df = pd.read_parquet("data/processed/features_baseline.parquet")
X = df[feature_cols].to_numpy(dtype=float)

# Predecir
y_pred = model.predict(X[:10])
print(y_pred)
```

El pipeline incluye internamente `SimpleImputer(strategy="median")` + `StandardScaler`,
por lo que **no es necesario pre-procesar** los datos antes de llamar a `predict`.

---

## Cómo usar en la app Streamlit

```python
import joblib, json
from pathlib import Path

MODEL_PATH  = Path("models/best_model_pipeline.joblib")
FCOLS_PATH  = Path("models/feature_columns.json")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FCOLS_PATH) as f:
        feature_cols = json.load(f)
    return model, feature_cols

model, feature_cols = load_model()

# Construir X con exactamente feature_cols (en ese orden)
X_input = df[feature_cols].to_numpy(dtype=float)
prediction = model.predict(X_input)
```

**Precauciones para producción:**
- Usar **exactamente** las columnas de `feature_columns.json`, en el mismo orden.
- No incluir `beat_type`, `case_id`, `rhythm_label` en `X_input`.
- Los modelos fueron entrenados con `sklearn=1.8.0`, `xgboost=3.2.0`, `numpy=1.26.4`.
  Usar versiones compatibles para evitar errores de deserialización.
- La corrida actual es FAST_MODE (5 iteraciones). Para un modelo de mayor calidad,
  re-ejecutar el notebook con `FAST_MODE=False` (30 iteraciones).

---

## Versiones de librerías al entrenar

```
sklearn  = 1.8.0
numpy    = 1.26.4
pandas   = 2.2.3
xgboost  = 3.2.0
joblib   = 1.5.3
```
