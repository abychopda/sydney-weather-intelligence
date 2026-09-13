from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

# Raw data covers the full period available to the project. Individual
# experiments subset this according to the needs of their target.
RAW_DATA_START = "2000-01-01"
RAW_DATA_END = "2025-12-31"

RAW_WEATHER_FILE = RAW_DATA_DIR / "sydney_hourly_2000_2025.csv"

# Modelling windows differ by target, chosen from the evidence:
#
# * Mean CCI has drifted downwards by roughly four points since 2000, so the
#   regression model is trained on a recent window that resembles the
#   conditions it will be deployed into.
# * The hazard classes are defined by absolute thresholds rather than
#   relative ones, so drift matters less, and the rare classes need every
#   example available: High Risk grows from 72 days to 274 across the full
#   period.
CCI_TRAIN_START = "2015-01-01"
WHC_TRAIN_START = "2000-01-01"

# Forecast horizons, in days.
CCI_HORIZONS = [1, 2, 3]
WHC_HORIZON = 7

# Target-specific output directories.
CCI_MODELS_DIR = MODELS_DIR / "comfort_climate"
WHC_MODELS_DIR = MODELS_DIR / "weather_hazard"
