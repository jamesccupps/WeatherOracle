"""WeatherOracle core — constants, config, and shared types."""

from .config import (
    LOCATIONS, WEATHER_MODELS, FORECAST_VARS, HOURLY_PARAMS,
    WMO_CODES, DEFAULT_CONFIG, APP_DIR, DB_PATH, MODEL_DIR,
    CONFIG_PATH, load_config, save_config,
)

__version__ = "2.0.0"
__all__ = [
    "LOCATIONS", "WEATHER_MODELS", "FORECAST_VARS", "HOURLY_PARAMS",
    "WMO_CODES", "DEFAULT_CONFIG", "APP_DIR", "DB_PATH", "MODEL_DIR",
    "CONFIG_PATH", "load_config", "save_config",
]
