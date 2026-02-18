"""Configuration partagée pour le pipeline ML Vigicrue."""

from datetime import date
from pathlib import Path

# --- Chemins ---
ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ML_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_DIR = MODELS_DIR / "onnx"

# --- Stations du bassin Vilaine Amont ---
STATIONS = [
    {
        "code": "J700061001",
        "label": "Bourgon - Pont D 106",
        "lat": 48.187,
        "lon": -1.058,
        "river": "Vilaine",
        "position": "amont",
    },
    {
        "code": "J701064001",
        "label": "La Chapelle-Erbrée (barrage)",
        "lat": 48.146,
        "lon": -1.128,
        "river": "Vilaine",
        "position": "amont",
        "barrage": True,
    },
    {
        "code": "J702401001",
        "label": "Erbrée - Les Ravenières",
        "lat": 48.109,
        "lon": -1.127,
        "river": "Valière",
        "position": "amont",
    },
    {
        "code": "J702403001",
        "label": "Erbrée - retenue Valière",
        "lat": 48.083,
        "lon": -1.164,
        "river": "Valière",
        "position": "amont",
        "barrage": True,
    },
    {
        "code": "J702402001",
        "label": "Vitré - Château des Rochers",
        "lat": 48.077,
        "lon": -1.166,
        "river": "Valière",
        "position": "central",
    },
    {
        "code": "J701061001",
        "label": "Vitré - Bas Pont",
        "lat": 48.123,
        "lon": -1.223,
        "river": "Vilaine",
        "position": "central",
    },
    {
        "code": "J704301001",
        "label": "Taillis",
        "lat": 48.163,
        "lon": -1.225,
        "river": "Cantache",
        "position": "amont",
        "no_q": True,
    },
    {
        "code": "J705302001",
        "label": "Pocé-les-Bois (barrage)",
        "lat": 48.128,
        "lon": -1.293,
        "river": "Cantache",
        "position": "amont",
        "barrage": True,
    },
    {
        "code": "J706062001",
        "label": "Châteaubourg - Bel Air",
        "lat": 48.108,
        "lon": -1.404,
        "river": "Vilaine",
        "position": "central",
    },
    {
        "code": "J708311001",
        "label": "La Bouëxière - Le Drugeon",
        "lat": 48.183,
        "lon": -1.499,
        "river": "Veuvre/Chevré",
        "position": "amont",
    },
    {
        "code": "J709063002",
        "label": "Cesson-Sévigné - Pont Briand",
        "lat": 48.124,
        "lon": -1.564,
        "river": "Vilaine",
        "position": "aval",
    },
]

STATION_CODES = [s["code"] for s in STATIONS]

# Stations sans données de débit (barrages + stations sans Q temps réel)
STATIONS_NO_Q = {s["code"] for s in STATIONS if s.get("barrage") or s.get("no_q")}

# Stations barrages
BARRAGE_CODES = {s["code"] for s in STATIONS if s.get("barrage")}

# --- Paramètres de collecte ---
COLLECT_START_DATE = "2000-01-01"
COLLECT_END_DATE = date.today().strftime("%Y-%m-%d")

# --- Paramètres du dataset ---
INPUT_WINDOW_HOURS = 72       # fenêtre d'entrée par défaut (heures)
FORECAST_HORIZONS = [1, 3, 6, 12, 24]  # horizons de prédiction (heures)
FUTURE_PRECIP_HOURS = 6                # heures de précipitations futures

# --- Paramètres d'entraînement ---
TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
# Test = tout ce qui reste (inclut la crue de janvier 2025)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 20  # early stopping

# --- Station cible par défaut ---
TARGET_STATION = "J706062001"  # Châteaubourg - Bel Air

# --- API URLs ---
HYDRO_BASE_URL = "https://www.hydro.eaufrance.fr/stationhydro/ajax/{station_id}/series"
METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# --- Clipping des outliers sur les features dérivées ---
DH_CLIP = 100    # ±100 mm/h (variation max réaliste pour dH/dt)
DQ_CLIP = 2000   # ±2000 L/s/h (variation max réaliste pour dQ/dt)
