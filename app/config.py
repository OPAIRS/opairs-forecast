from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

MODELS_FILE = Path(os.getenv("FORECAST_MODELS_FILE", Path(__file__).resolve().parent.parent / "models.yaml"))
API_KEY = os.getenv("FORECAST_API_KEY", "")
MAX_SERIES = int(os.getenv("FORECAST_MAX_SERIES", "500"))
MAX_POINTS = int(os.getenv("FORECAST_MAX_POINTS", "200000"))
MAX_HORIZON = int(os.getenv("FORECAST_MAX_HORIZON", "366"))
MAX_CONCURRENCY = int(os.getenv("FORECAST_MAX_CONCURRENCY", "1"))
DEVICE = os.getenv("FORECAST_DEVICE", "auto")  # auto | cpu | cuda | cuda:1
# Parallelität der Statistikmodelle (Prozesse je Anfrage; 1 = aus, -1 = alle Kerne). Greift erst ab 4 Serien.
N_JOBS = int(os.getenv("FORECAST_N_JOBS", "1"))
# Torch-Threads für Chronos/TiRex auf der CPU (0 = Torch-Standard, also meist alle Kerne). Alternativ OMP_NUM_THREADS setzen.
TORCH_THREADS = int(os.getenv("FORECAST_TORCH_THREADS", "0"))
# Beim Start Modelle laden und Statistikmodelle kompilieren (1 = an). Die erste echte Anfrage wartet dann nicht auf den Kaltstart.
WARMUP = os.getenv("FORECAST_WARMUP", "0").lower() in ("1", "true", "yes", "an")
# Ensemble (Mittel der zwei besten Backtest-Modelle): wird statt des besten Einzelmodells gewählt, wenn seine
# relative MAE höchstens um diesen Faktor schlechter ist (Backtest-Rauschen: Ensembles verallgemeinern meist besser).
ENSEMBLE_TOLERANCE = float(os.getenv("FORECAST_ENSEMBLE_TOLERANCE", "1.05"))

# Optionaler Erklär-Endpunkt (OpenAI-kompatibel, z. B. LiteLLM). Ohne LLM_MODEL aus.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(MODELS_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("baseline", "seasonal_naive")
    cfg.setdefault("models", {})
    cfg.setdefault("default_candidates", [cfg["baseline"]])
    if cfg["baseline"] not in cfg["models"]:
        raise ValueError(f"baseline '{cfg['baseline']}' fehlt in models")
    return cfg
