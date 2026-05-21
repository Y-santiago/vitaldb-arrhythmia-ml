from pathlib import Path

# Anchored to THIS file: frontend/app/utils/paths.py
_UTILS_DIR   = Path(__file__).resolve().parent   # frontend/app/utils/
APP_DIR      = _UTILS_DIR.parent                  # frontend/app/
FRONTEND_DIR = APP_DIR.parent                     # frontend/
PROJECT_ROOT = FRONTEND_DIR.parent                # project root

REPORTS_DIR       = PROJECT_ROOT / "reports"
REPORT_TABLES_DIR = REPORTS_DIR  / "tables"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR        = PROJECT_ROOT / "models"
DATA_DIR          = PROJECT_ROOT / "data"
DEMO_DATA_DIR     = DATA_DIR     / "demo"
ASSETS_DIR        = APP_DIR      / "assets"
