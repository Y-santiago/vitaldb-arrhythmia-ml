# VitalDB Arrhythmia ML

Proyecto académico y exploratorio de machine learning para la **clasificación de
ritmos cardíacos** (`rhythm_label`) a partir de segmentos temporales de ECG
intraoperatorio, utilizando la *VitalDB Arrhythmia Database 1.0.0* publicada en
PhysioNet.

> **Advertencia académica.** Este proyecto tiene fines exclusivamente educativos
> y de investigación exploratoria. **No constituye un dispositivo médico ni
> debe usarse para diagnóstico, monitoreo o decisión clínica de ningún tipo.**

---

## 1. Descripción del problema

El objetivo es predecir la etiqueta de ritmo (`rhythm_label`) asociada a
ventanas temporales de ECG extraídas alrededor de cada latido anotado en la
*VitalDB Arrhythmia Database*. El problema se plantea como una tarea de
**clasificación supervisada multiclase** sobre señales fisiológicas
intraoperatorias.

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

---

## 2. Fuente del dataset

- **PhysioNet**: *VitalDB Arrhythmia Database: An anesthesiologist-validated
  large-scale intraoperative arrhythmia dataset with beat and rhythm labels
  1.0.0*. El paquete contiene metadata por caso y archivos de anotaciones
  (latidos, ritmos, calidad de señal).
- **VitalDB**: la señal ECG cruda **no** está incluida en el paquete de
  PhysioNet. Se descarga desde VitalDB usando la librería oficial `vitaldb`,
  indexando por `case_id`.

> El dataset de PhysioNet y las señales de VitalDB se distribuyen bajo sus
> propios términos de uso. Consulta las licencias originales antes de
> redistribuir cualquier subconjunto.

---

## 3. Objetivo de la primera fase

La primera fase es **exploratoria** y cubre:

1. Estructuración del repositorio y dependencias.
2. Carga y validación de metadata y anotaciones.
3. EDA de anotaciones: distribución de `rhythm_label`, conteos por caso,
   patrones de calidad de señal.
4. Análisis de desbalance entre clases y solapamiento con calidad de señal.
5. Carga de la señal ECG cruda desde VitalDB para un subconjunto reducido de
   casos.
6. Definición de ventanas temporales alrededor de cada latido.
7. Extracción de features iniciales (estadísticas temporales y derivadas RR).
8. Entrenamiento de un baseline pequeño con **validación por grupos
   (`case_id`)**.

No se busca aún optimizar un modelo final ni reportar resultados definitivos.

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
├── notebooks/
│   ├── 01_download_and_structure.ipynb
│   ├── 02_eda_annotations.ipynb
│   ├── 03_ecg_loading_and_visualization.ipynb
│   ├── 04_windowing_and_feature_engineering.ipynb
│   └── 05_baseline_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── download.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── features.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── reports/
│   ├── figures/
│   └── tables/
├── models/
└── tests/
    ├── test_data_loading.py
    ├── test_windowing.py
    └── test_features.py
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

### 6.2 Señal ECG cruda (VitalDB)

La descarga se realiza desde código usando la librería `vitaldb`. El notebook
`03_ecg_loading_and_visualization.ipynb` muestra el flujo. Los waveforms
descargados se almacenan localmente en `data/raw/vitaldb_waveforms/` (también
excluida por `.gitignore`).

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

## 8. Flujo de trabajo recomendado

1. **`notebooks/01_download_and_structure.ipynb`** — verificación del paquete
   de PhysioNet en disco, lectura inicial de `metadata.csv` y exploración
   estructural de las anotaciones.
2. **`notebooks/02_eda_annotations.ipynb`** — EDA sobre las anotaciones:
   distribución de `rhythm_label`, conteos por caso, distribución de
   `bad_signal_quality`, solapamientos de clases, análisis descriptivo
   complementario sobre `beat_type` *(no se usa como predictor)*.
3. **`notebooks/03_ecg_loading_and_visualization.ipynb`** — carga de la señal
   ECG cruda desde VitalDB para un subconjunto reducido de casos y
   visualización de tramos.
4. **`notebooks/04_windowing_and_feature_engineering.ipynb`** — definición de
   ventanas temporales alrededor de cada latido (con sobrelapamiento opcional)
   y extracción de features temporales y RR.
5. **`notebooks/05_baseline_modeling.ipynb`** — *split por `case_id`* con
   `GroupKFold` o similar, entrenamiento de un baseline pequeño y reporte de
   métricas macro.

Toda transformación importante debe poder ejecutarse también vía los módulos
de `src/` para mantener reproducibilidad fuera de los notebooks.

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
