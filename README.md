# VitalDB Arrhythmia ML

Proyecto académico y exploratorio de **clasificación multiclase de
`rhythm_label`** usando datos **tabulares filtrados** de anotaciones y
metadatos de la *VitalDB Arrhythmia Database 1.0.0* (PhysioNet).

> **Pivot metodológico (fase actual).** Esta iteración trabaja sobre datos
> tabulares (anotaciones + metadata). El enfoque previo basado en señal
> ECG cruda (descarga desde VitalDB + ventaneo de señal + features
> temporales) queda como **línea exploratoria histórica** marcada como
> `legacy` y no es el flujo principal de modelado.

> **Advertencia académica.** Este proyecto tiene fines exclusivamente
> educativos y de investigación exploratoria. **No constituye un
> dispositivo médico ni debe usarse para diagnóstico, monitoreo o
> decisión clínica de ningún tipo.**

---

## 1. Descripción del problema

El objetivo es predecir la etiqueta de ritmo (`rhythm_label`) asociada a
cada latido anotado en la *VitalDB Arrhythmia Database*. Se plantea como
**clasificación supervisada multiclase** sobre un dataset tabular
construido a partir de:

- las anotaciones de PhysioNet (`time_second`, `bad_signal_quality`,
  etiquetas de ritmo y latido por caso) y
- los metadatos por caso (`metadata.csv`: edad, sexo, antropometría,
  tipo de cirugía, anestesia, valores preoperatorios, etc.).

A cada fila (un latido) se le adjuntan features temporales locales
dentro del caso (`rr_prev`, `rr_next`, `hr_inst_from_rr_prev`,
`position_in_case`). **No se usa la señal ECG cruda en esta fase.**

Aspectos relevantes:

- Las etiquetas de ritmo provienen de anotaciones validadas presentes en el
  paquete de PhysioNet.
- El dataset presenta **desbalance fuerte** entre clases minoritarias y
  mayoritarias.
- La variable `beat_type` describe el tipo morfológico del latido y **no se
  utilizará como variable predictora** en ningún experimento. Solo se permite
  su uso para análisis descriptivos complementarios.
- Los registros marcados como `bad_signal_quality` se excluyen.
- La clase `Noise` se excluye.
- `case_id` se usa **únicamente** como grupo para validación; nunca como
  predictor.
- Columnas con fuga de información se excluyen del set de features (ver
  `src/config.py::TABULAR_LEAKAGE_COLUMNS`).

---

## 2. Fuente del dataset

- **PhysioNet**: *VitalDB Arrhythmia Database: An anesthesiologist-validated
  large-scale intraoperative arrhythmia dataset with beat and rhythm labels
  1.0.0*. El paquete contiene `metadata.csv` (1 fila por caso) y un archivo
  `Annotation_file_<case_id>.csv` por caso con `time_second`, `beat_type`,
  `rhythm_label`, `bad_signal_quality` y `bad_signal_quality_label`.
- **VitalDB (no usado en esta fase).** La señal ECG cruda *no* está
  incluida en el paquete y *no* se utiliza en la fase tabular actual.
  Existe código `legacy` que la descargaba desde la librería oficial
  `vitaldb`; ese flujo queda pausado.

> El dataset de PhysioNet se distribuye bajo sus propios términos de uso.
> Consulta la licencia original antes de redistribuir cualquier subconjunto.

---

## 3. Objetivo de la fase actual (tabular)

1. Auditar las anotaciones + metadata tras filtros (`Noise`,
   `bad_signal_quality`, etiquetas nulas).
2. Construir un dataset modelable (1 fila por latido) que combine metadata
   por caso con features temporales locales (`rr_prev`, `rr_next`,
   `hr_inst_from_rr_prev`, `position_in_case`).
3. Split **80/20 por `case_id`** con cobertura de clases (sin métricas
   de desempeño en la elección de semilla).
4. Pipeline `ColumnTransformer(Imputer+Scaler / Imputer+OneHotEncoder) →
   Clasificador` con `class_weight="balanced"` cuando aplica.
5. `RandomizedSearchCV` sobre 6 modelos:
   `logreg`, `decision_tree`, `random_forest`, `xgboost`, `linear_svc`,
   `mlp`. Métrica primaria `f1_macro`.
6. CV interna por grupo (`StratifiedGroupKFold` con fallback a
   `GroupKFold`).
7. Test congelado: se evalúa una sola vez al final.

**Estado de la línea ECG (legacy).** Notebooks `04` y `05`, módulos
`src/search.py`, `src/download.py`, `src/windowing.py`, y los scripts
`scripts/01_download_all_available_ecg.py` /
`scripts/02_build_features_all_windows.py` /
`scripts/03_run_hyperparameter_search.py` (ECG) están marcados con un
banner `[LEGACY — ...]` en su docstring. No son el flujo activo.

---

## 4. Estructura del repositorio

```
vitaldb-arrhythmia-ml/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── data/
│   ├── raw/
│   │   ├── physionet_annotations/   # paquete PhysioNet (no se versiona)
│   │   └── vitaldb_waveforms/       # ECG descargado de VitalDB (no se versiona)
│   ├── interim/                     # transformaciones intermedias (no se versiona)
│   └── processed/                   # datasets listos para modelado (no se versiona)
├── scripts/                                            # flujo activo (tabular)
│   ├── 01_audit_filtered_tabular_dataset.py
│   ├── 02_build_filtered_tabular_modeling_dataset.py
│   ├── 03_run_tabular_hyperparameter_search.py
│   ├── 01_download_all_available_ecg.py                # legacy (ECG)
│   ├── 02_build_features_all_windows.py                # legacy (ECG)
│   └── 03_run_hyperparameter_search.py                 # legacy (ECG)
├── notebooks/
│   ├── 01_download_and_structure.ipynb                 # exploratorio
│   ├── 02_eda_annotations.ipynb                        # exploratorio
│   ├── 03_ecg_loading_and_visualization.ipynb          # legacy (ECG)
│   ├── 04_windowing_and_feature_engineering.ipynb      # legacy (ECG)
│   ├── 05_baseline_modeling.ipynb                      # legacy (ECG)
│   ├── 06_full_modeling_hyperparameter_search.ipynb    # legacy (ECG)
│   └── 06_tabular_modeling_hyperparameter_search.ipynb # flujo activo
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── download.py            # legacy (vitaldb.load_case)
│   ├── preprocessing.py       # filtros + build_tabular_preprocessor
│   ├── windowing.py           # legacy (ventaneo de señal)
│   ├── features.py            # estadísticas + per_beat_rr (legacy en parte)
│   ├── modeling.py            # split por grupo + utilidades comunes
│   ├── search.py              # legacy (búsqueda sobre features ECG)
│   ├── tabular_search.py      # flujo activo (búsqueda tabular)
│   ├── evaluation.py
│   └── utils.py
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── PROJECT_REPORT.md
│   ├── TABULAR_MODELING_REPORT.md
│   ├── MODELING_REPORT.md     # legacy (corrida ECG)
│   └── NEXT_STEPS_FOR_CHATGPT.md
├── models/
└── tests/
    ├── test_data_loading.py
    ├── test_windowing.py
    ├── test_features.py
    ├── test_modeling.py
    ├── test_evaluation.py
    ├── test_search.py
    └── test_tabular_search.py
```

---

## 5. Instrucciones de instalación

Recomendado **Python 3.11**. Se asume ejecución en **Visual Studio Code** con
la extensión de Python/Jupyter.

### Opción A — `venv` + `pip`

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name vitaldb-arrhythmia-ml
```

### Opción B — Conda

```bash
conda env create -f environment.yml
conda activate vitaldb-arrhythmia-ml
python -m ipykernel install --user --name vitaldb-arrhythmia-ml
```

En VS Code: abrir la carpeta raíz del proyecto y seleccionar el intérprete
correspondiente al entorno creado, así como el kernel `vitaldb-arrhythmia-ml`
en los notebooks.

---

## 6. Descarga de datos

> **Importante.** Los datos **nunca** se versionan. La carpeta `data/` está
> excluida por `.gitignore`.

### 6.1 Paquete de anotaciones (PhysioNet)

1. Acceder a la página del dataset *VitalDB Arrhythmia Database 1.0.0* en
   PhysioNet y descargar el paquete completo.
2. Colocar el contenido del paquete dentro de:

   ```
   data/raw/physionet_annotations/
   ```

   Debe quedar visible al menos `metadata.csv` y la carpeta de archivos de
   anotación.

### 6.2 Señal ECG cruda (VitalDB) — *no usada en la fase tabular*

Esta fase **no descarga** señales ECG. El flujo activo consume únicamente
las anotaciones y la metadata de PhysioNet. El código que descargaba
desde VitalDB queda como `legacy` (ver banner en
`scripts/01_download_all_available_ecg.py` y `src/download.py`).

---

## 7. Advertencias

- **Proyecto académico, no clínico.** No usar para diagnóstico, monitoreo,
  decisión médica ni dispositivos regulados.
- **No subir datos al repositorio.** Las carpetas `data/raw/`, `data/interim/`,
  `data/processed/` están bloqueadas en `.gitignore`. Antes de cada `git add`
  verifica que no se incluyan archivos `*.csv`, `*.parquet`, `*.pkl`, `*.npy`,
  `*.h5` ni binarios pesados.
- **No subir modelos entrenados.** La carpeta `models/` está bloqueada por
  `.gitignore`.
- **No incluir información personal** ni identificadores reales en commits,
  notebooks o reportes.

---

## 8. Flujo de trabajo recomendado (fase tabular)

### 8.1 Pipeline activo (sin ECG crudo)

```bash
# 1. Auditar anotaciones + metadata
python scripts/01_audit_filtered_tabular_dataset.py

# 2. Construir el dataset modelable
python scripts/02_build_filtered_tabular_modeling_dataset.py

# 3. Búsqueda de hiperparámetros multi-modelo
#    (debug rápido)
python scripts/03_run_tabular_hyperparameter_search.py --debug
#    (full run)
python scripts/03_run_tabular_hyperparameter_search.py --n-iter 30 --n-splits 5
```

El notebook equivalente, con celdas inspeccionables, es
`notebooks/06_tabular_modeling_hyperparameter_search.ipynb`.

### 8.2 Línea exploratoria (legacy, NO usar en esta fase)

Los notebooks `01–05` y `06_full_modeling_hyperparameter_search.ipynb`,
junto con `scripts/01_download_all_available_ecg.py`,
`scripts/02_build_features_all_windows.py` y
`scripts/03_run_hyperparameter_search.py`, son la línea exploratoria
basada en ECG crudo. Quedan disponibles como referencia histórica.

Toda transformación importante debe poder ejecutarse también vía los
módulos de `src/` para mantener reproducibilidad fuera de los notebooks.

---

## 9. Criterios de evaluación

- **Separación train/test por grupos**: nunca dividir aleatoriamente latidos o
  ventanas. La unidad de agrupación es `case_id`. Se usa `GroupKFold` o
  `GroupShuffleSplit`.
- **Métricas principales**:
  - `f1_score` macro
  - `recall` macro
  - `balanced_accuracy_score`
  - `classification_report` por clase
  - Matriz de confusión normalizada
- **Reporte de desbalance**: distribución de clases en train y test, conteo
  por caso, y conteo de ventanas por clase.
- **Trazabilidad**: cualquier resultado reportado debe ser reproducible
  desde los notebooks y los módulos de `src/`.

---

## 10. Limitaciones

- **Desbalance fuerte**: algunas clases de ritmo aparecen con muy baja
  frecuencia, lo que limita el desempeño esperable de modelos clásicos sin
  estrategias específicas de remuestreo o ponderación.
- **Variabilidad inter-paciente**: la morfología y la frecuencia de los ritmos
  varían entre pacientes; un split aleatorio sobreestimaría el desempeño.
- **Calidad heterogénea de señal**: incluso tras filtrar `bad_signal_quality`,
  pueden persistir artefactos.
- **Etiquetas no perfectas**: las anotaciones, aunque validadas, no son una
  verdad absoluta libre de ruido.
- **Tamaño relativo**: el número de casos disponibles es limitado; resultados
  con conjuntos pequeños deben interpretarse con cautela.
- **Sesgo de dominio**: ECG **intraoperatorio**; los modelos no son
  trasladables directamente a ambulatorio, Holter ni unidades de cuidado
  intensivo.
- **No se persiguen métricas clínicamente válidas** en esta fase.

---

## 11. Crear el repositorio remoto manualmente (opcional)

Si no se dispone de `gh` (GitHub CLI) autenticado, el repositorio remoto puede
crearse manualmente:

```bash
# Desde la raíz del proyecto, una vez ejecutado git init y hecho el primer commit
git remote add origin https://github.com/<usuario>/vitaldb-arrhythmia-ml.git
git branch -M main
git push -u origin main
```

Antes de hacer `push`, verifica nuevamente con `git status` que **no** se
estén incluyendo archivos de `data/`, `models/`, ni binarios pesados.
streamlit run frontend\app\app.py