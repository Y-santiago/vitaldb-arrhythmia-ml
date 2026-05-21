# ECG Arrhythmia ML — App Streamlit

Documentación de la app Streamlit generada en la fase 1 de conversión desde el diseño de referencia (`frontend/streamlit_reference/`).

---

## Ubicación y cómo correr la app

```bash
# Desde la raíz del proyecto:
streamlit run frontend/app/app.py
```

La app abre en `http://localhost:8501` por defecto.

---

## Estructura creada (`frontend/app/`)

```
frontend/app/
  app.py                        ← Portada / entry point
  .streamlit/
    config.toml                 ← Tema oscuro (bg #07090f, primary #4a8cff)
  pages/
    01_inicio.py                ← FUNCIONAL — datos reales cargados
    02_pipeline.py              ← Placeholder
    03_dataset_limpieza.py      ← Placeholder
    04_rendimiento_modelo.py    ← Placeholder (datos disponibles)
    05_evaluacion_clase.py      ← Placeholder (datos disponibles)
    06_matriz_confusion.py      ← Placeholder (solo PNG disponible)
    07_predicciones.py          ← Placeholder (faltan archivos clave)
    08_interpretabilidad.py     ← Placeholder (datos disponibles)
    09_conclusiones.py          ← Placeholder (todo estático)
  components/
    layout.py    ← inject_css(), sidebar_branding(), page_header(), placeholder_page()
    cards.py     ← card_header(), metric_card(), callout(), kv_table(), section_title()
    badges.py    ← badge(), badge_row(), pill(), status_badge()
    charts.py    ← mini_ecg_placeholder(), bar_chart_h(), bar_chart_v_grouped()
    tables.py    ← safe_dataframe(), format_model_table(), style_model_table()
  utils/
    paths.py     ← PROJECT_ROOT, REPORTS_DIR, MODELS_DIR, etc. (basado en __file__)
    loaders.py   ← loaders cacheados para todos los archivos de datos
  assets/
    styles.css   ← Variables CSS, cards, badges, callouts, hero, sidebar
```

---

## Archivos reales disponibles y cargados

| Archivo | Loader en loaders.py | Notas |
|---|---|---|
| `reports/tables/model_comparison.csv` | `load_model_comparison()` | 5 modelos, datos reales |
| `reports/tables/best_model_classification_report.csv` | `load_classification_report()` | 10 clases reales |
| `reports/tables/best_model_feature_importance.csv` | `load_feature_importance()` | 26 features |
| `reports/figures/best_model_confusion_matrix.png` | `confusion_matrix_figure_path()` | Solo imagen PNG |
| `reports/figures/feature_correlation_heatmap.png` | `correlation_figure_path()` | Solo imagen PNG |
| `models/best_model_pipeline.joblib` | `load_model()` | Pipeline LinearSVC completo |
| `models/model_artifacts_metadata.json` | `load_model_metadata()` | Metadata de entrenamiento |
| `models/feature_columns.json` | `load_feature_columns()` | Lista de 26 features |

---

## Archivos faltantes (placeholders activos)

| Archivo | Página afectada | Impacto |
|---|---|---|
| `reports/tables/confusion_matrix.csv` | 06 Matriz de confusión | Sin él, solo se puede mostrar la imagen PNG |
| `reports/tables/test_predictions.csv` | 07 Predicciones | Sin él, se usan 6 casos demo hardcoded |
| `data/demo/demo_windows.parquet` | 07 Predicciones | Necesario para visor de ECG real |
| `data/demo/waveforms/` | 07 Predicciones | Señales brutas para ECG interactivo |

---

## Datos reales del modelo (diferencia importante vs. diseño de referencia)

El diseño de referencia (`frontend/streamlit_reference/data.jsx`) usaba datos **simulados**.
Los datos reales del pipeline son:

| Dato | Valor diseño (simulado) | Valor real |
|---|---|---|
| Modelo ganador | LinearSVC | LinearSVC |
| F1-macro | 0.742 | **0.344** |
| Accuracy | 0.918 | **0.806** |
| Clases | NSR, SB, ST, PAC, PVC, AFib, AFlut, JR, VT, Asys | AFIB/AFL, AVB, N, Patterned Atrial Ectopy, Patterned Ventricular Ectopy, SND, SVTA, Unclassifiable, VT, WAP/MAT |
| Features | 12 | **26** |
| Train grupos | — | 384 |
| Test grupos | — | 97 |

**Nota:** La app `01_inicio.py` muestra los valores reales, no los simulados.

Mejores hiperparámetros LinearSVC: `clf__C = 0.0745`

---

## Componentes reutilizables — referencia rápida

### `components/layout.py`
```python
inject_css()                                     # inyectar estilos en cada página
sidebar_branding(winner_model, winner_f1, ok)    # branding + estado del pipeline
page_header(title, subtitle, badge_html)         # encabezado de página estándar
placeholder_page(title, desc, files, note)       # página placeholder completa
```

### `components/cards.py`
```python
card_header(title, subtitle, right_html)         # encabezado de card con st.container(border=True)
metric_card(label, value, helper, accent)        # metric card HTML con accent border
callout(kind, title, body)                       # info/warn/err/ok callout
kv_table(rows)                                   # tabla key-value
section_title(text)                              # separador de sección
```

### `components/badges.py`
```python
badge(text, kind)      # retorna HTML string — usar con st.markdown(unsafe_allow_html=True)
badge_row(*badges)     # envuelve badges en flex row
pill(text, kind)       # igual que badge pero totalmente redondo
status_badge(text, status)  # mapea ok/warning/error/info a badge kind
```

### `components/charts.py`
```python
mini_ecg_placeholder(height, n_beats)           # ECG sintético decorativo (Plotly)
bar_chart_h(labels, values, ...)                # barra horizontal Plotly tema oscuro
bar_chart_v_grouped(df, x_col, y_cols, ...)     # barras agrupadas verticales
empty_chart_placeholder(title, description)     # placeholder HTML para gráfico pendiente
```

### `utils/paths.py`
```python
PROJECT_ROOT        # raíz del proyecto (detectada desde __file__)
APP_DIR             # frontend/app/
REPORTS_DIR         # reports/
REPORT_TABLES_DIR   # reports/tables/
REPORT_FIGURES_DIR  # reports/figures/
MODELS_DIR          # models/
ASSETS_DIR          # frontend/app/assets/
```

---

## Patrón estándar para cada página

```python
import streamlit as st

st.set_page_config(page_title="ECG · NombrePágina", page_icon="🫀", layout="wide")

from components.layout import inject_css, sidebar_branding, page_header
from utils.loaders import load_model_metadata

inject_css()
meta = load_model_metadata()
winner = meta.get("winner_model","—").replace("_"," ").title() if meta else "—"
f1_str = f"{meta.get('winner_test_f1_macro',0):.3f}" if meta else "—"
sidebar_branding(winner_model=winner, winner_f1=f1_str, pipeline_ok=meta is not None)

page_header("Título", "Descripción breve de la sección.")

# ... contenido de la página
```

---

## Estado de implementación

| Página | Estado | Datos disponibles |
|---|---|---|
| `app.py` (portada) | Funcional | metadata |
| `01_inicio.py` | **Funcional** | model_comparison, classification_report, metadata |
| `02_pipeline.py` | Placeholder | metadata |
| `03_dataset_limpieza.py` | Placeholder | classification_report (clases) |
| `04_rendimiento_modelo.py` | Placeholder | **model_comparison disponible — implementar 2°** |
| `05_evaluacion_clase.py` | Placeholder | **classification_report disponible — implementar 3°** |
| `06_matriz_confusion.py` | Placeholder | solo PNG — falta CSV numérico |
| `07_predicciones.py` | Placeholder | falta test_predictions.csv y demo_windows.parquet |
| `08_interpretabilidad.py` | Placeholder | **feature_importance disponible — implementar 4°** |
| `09_conclusiones.py` | Placeholder | todo estático |

---

## Próxima fase recomendada

**Implementar `04_rendimiento_modelo.py`** — todos los datos ya existen:
- `model_comparison.csv` → tabla comparativa, grouped bar chart
- `model_artifacts_metadata.json` → hiperparámetros, tiempos, estrategia CV

Después: `05_evaluacion_clase.py` (classification_report disponible).

---

## Referencia visual

La carpeta `frontend/streamlit_reference/` contiene el diseño original en HTML/CSS/JSX.
**No modificar** — es la referencia visual intocable.

Archivos relevantes:
- `styles.css` — paleta de colores y variables CSS
- `data.jsx` — datos simulados (ver diferencias con datos reales en la tabla arriba)
- `performance.jsx`, `classes.jsx`, `confusion.jsx` — referencia visual para las páginas pendientes
