"""Configuración global del proyecto.

Centraliza rutas relativas, nombres de columnas y parámetros compartidos
entre los módulos del paquete y los notebooks.

Reglas metodológicas codificadas aquí:
    * `TARGET_COLUMN = "rhythm_label"` es la variable objetivo.
    * `BEAT_TYPE_COLUMN = "beat_type"` queda registrada solo para análisis
      descriptivo. Está prohibido usarla como variable predictora.
    * Se excluyen registros con `bad_signal_quality`.
    * Se excluye la clase `Noise` de la variable objetivo.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas relativas al proyecto
# ---------------------------------------------------------------------------
# `PROJECT_ROOT` apunta a la raíz del repositorio independientemente del
# directorio de trabajo desde el que se importe el paquete.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

PHYSIONET_DIR: Path = RAW_DIR / "physionet_annotations"
VITALDB_WAVEFORMS_DIR: Path = RAW_DIR / "vitaldb_waveforms"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Nombres de columnas (esperados en metadata.csv y anotaciones)
# ---------------------------------------------------------------------------
CASE_ID_COLUMN: str = "case_id"
TARGET_COLUMN: str = "rhythm_label"
BEAT_TYPE_COLUMN: str = "beat_type"          # uso descriptivo únicamente
SIGNAL_QUALITY_COLUMN: str = "bad_signal_quality"
BEAT_TIME_COLUMN: str = "time_second"        # nombre real en los CSV de PhysioNet

# Patrón del nombre de los archivos de anotación. El paquete oficial usa
# `Annotation_file_<case_id>.csv` (singular). Se admite también la variante
# plural por robustez ante posibles renombrados.
ANNOTATION_FILENAME_REGEX: str = r"^Annotations?_file_(\d+)\.csv$"

# Etiquetas y categorías a excluir
EXCLUDED_RHYTHM_LABELS: tuple[str, ...] = ("Noise",)

# Columnas prohibidas como predictoras (deben filtrarse antes de entrenar)
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    BEAT_TYPE_COLUMN,
    TARGET_COLUMN,
    CASE_ID_COLUMN,
    SIGNAL_QUALITY_COLUMN,
)

# ---------------------------------------------------------------------------
# Parámetros por defecto del ventanado
# ---------------------------------------------------------------------------
# Frecuencia de muestreo nominal del ECG en VitalDB (Hz).
# Verificar contra la señal real antes de procesar.
DEFAULT_ECG_FS_HZ: int = 500

# Duración total de la ventana centrada en el latido (segundos).
DEFAULT_WINDOW_SECONDS: float = 2.0

# Proporción de sobrelapamiento entre ventanas consecutivas (0.0 - 1.0).
DEFAULT_WINDOW_OVERLAP: float = 0.0

# ---------------------------------------------------------------------------
# Modelado y validación
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
DEFAULT_N_SPLITS: int = 5

# Nombre del canal ECG por defecto a solicitar a VitalDB.
# El identificador exacto debe confirmarse al cargar la primera señal.
DEFAULT_ECG_TRACK_NAME: str = "SNUADC/ECG_II"

# ---------------------------------------------------------------------------
# Modelado tabular (flujo activo desde la iteración tabular)
# ---------------------------------------------------------------------------
# Columnas que NUNCA pueden entrar al set de features predictoras:
#   * el target o su codificación;
#   * `beat_type` (regla metodológica del proyecto);
#   * el identificador del caso (se usa solo como grupo);
#   * filtros (`bad_signal_quality*`);
#   * outcomes post-operatorios (fuga temporal);
#   * timestamps administrativos.
TABULAR_LEAKAGE_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,            # rhythm_label
    BEAT_TYPE_COLUMN,         # beat_type — prohibido como predictor
    "rhythm_classes",         # contiene la lista de ritmos del caso → leakage directo
    SIGNAL_QUALITY_COLUMN,    # bad_signal_quality (filtro)
    "bad_signal_quality_label",  # texto descriptivo del filtro
    CASE_ID_COLUMN,           # case_id — solo se usa como grupo
    "caseid",                 # variante con typo en un archivo (Annotation_file_2453) — alias del id
    "subjectid",              # identificador alternativo del paciente
    "source_file",            # identificador del archivo origen
    "icu_days",               # estancia en UCI post-op (outcome posterior)
    "death_inhosp",           # mortalidad intra-hospitalaria (outcome posterior)
    "adm",                    # timestamp administrativo
    "dis",                    # timestamp administrativo (egreso)
)

# Umbral de cardinalidad para considerar una columna como categórica
# elegible. Variables categóricas por encima de este límite quedan fuera
# del set inicial para evitar explosión dimensional de OneHotEncoder.
TABULAR_MAX_CATEGORY_CARDINALITY: int = 30

# Mínima frecuencia (en filas) que debe tener una categoría para mantener
# su propia columna en OneHotEncoder. El resto va a la categoría
# `infrequent_sklearn` automáticamente.
TABULAR_OHE_MIN_FREQUENCY: int = 50

# Nombre del parquet de salida del flujo tabular.
TABULAR_DATASET_FILENAME: str = "filtered_tabular_modeling_dataset.parquet"
