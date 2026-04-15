"""Configuration, constants, and location definitions.

Locations, API keys, and sensor entities are loaded from
data/weather_oracle_config.json. On first run, a default config
is created that you can edit in the Settings tab or directly.
"""

import json
import os
from pathlib import Path

# Load .env file if present (for API keys)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

APP_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = APP_DIR / "data" / "weather_oracle.db"
CONFIG_PATH = APP_DIR / "data" / "weather_oracle_config.json"
MODEL_DIR = APP_DIR / "models"

# Ensure dirs exist
(APP_DIR / "data").mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─── Default Locations (override in config JSON — see README) ─────────────────

_DEFAULT_LOCATIONS = {
    "location_1": {
        "name": "My Weather Station",
        "short": "HOME",
        "tempest_station": 0,
        "lat": 0.0,
        "lon": 0.0,
        "metar_station": "",
        "nws_office": "",
        "nws_gridpoint": None,
        "color": "#00d4aa",
        "color_dark": "#008f72",
    },
}

# ─── Open-Meteo Weather Models ───────────────────────────────────────────────

WEATHER_MODELS = {
    "gfs": {
        "name": "GFS (NOAA)",
        "url": "https://api.open-meteo.com/v1/gfs",
        "desc": "Global 0.25deg, strong 3-7 day",
        "priority": 1.5,
        "api_model": "gfs_global",
    },
    "hrrr": {
        "name": "HRRR (NOAA)",
        "url": "https://api.open-meteo.com/v1/gfs",
        "desc": "High-res rapid refresh, best <18hr (US only)",
        "priority": 3.0,
        "api_model": "ncep_hrrr_conus",
        "max_hours": 48,
    },
    "ecmwf": {
        "name": "ECMWF IFS",
        "url": "https://api.open-meteo.com/v1/ecmwf",
        "desc": "European model, best overall skill",
        "priority": 2.5,
    },
    "icon": {
        "name": "ICON (DWD)",
        "url": "https://api.open-meteo.com/v1/dwd-icon",
        "desc": "German model, good mesoscale",
        "priority": 1.5,
    },
    "gem": {
        "name": "GEM (Canada)",
        "url": "https://api.open-meteo.com/v1/gem",
        "desc": "Canadian model, relevant for northern US/Canada",
        "priority": 1.2,
    },
    "jma": {
        "name": "JMA (Japan)",
        "url": "https://api.open-meteo.com/v1/jma",
        "desc": "Japanese model, independent global view",
        "priority": 1.0,
    },
    "tempest_bf": {
        "name": "Tempest BetterForecast",
        "url": None,
        "desc": "Tempest ML-enhanced local forecast",
        "priority": 2.5,
    },
}

# ─── Forecast Variables ───────────────────────────────────────────────────────

FORECAST_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation_probability", "precipitation",
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
    "cloud_cover", "pressure_msl", "weather_code",
]

HOURLY_PARAMS = ",".join(FORECAST_VARS)

# ─── WMO Weather Codes ───────────────────────────────────────────────────────

WMO_CODES = {
    0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime Fog",
    51: "Lt Drizzle", 53: "Drizzle", 55: "Hvy Drizzle",
    56: "Frzg Drizzle", 57: "Hvy Frzg Drizzle",
    61: "Lt Rain", 63: "Rain", 65: "Heavy Rain",
    66: "Frzg Rain", 67: "Hvy Frzg Rain",
    71: "Lt Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
    80: "Lt Showers", 81: "Showers", 82: "Heavy Showers",
    85: "Lt Snow Shwrs", 86: "Snow Showers",
    95: "Thunderstorm", 96: "T-Storm + Hail", 99: "Severe T-Storm",
}

WMO_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌧️", 55: "🌧️", 56: "🌧️", 57: "🌧️",
    61: "🌦️", 63: "🌧️", 65: "🌧️", 66: "🌧️", 67: "🌧️",
    71: "🌨️", 73: "❄️", 75: "❄️", 77: "❄️",
    80: "🌦️", 81: "🌧️", 82: "⛈️",
    85: "🌨️", 86: "🌨️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}

# ─── Default Config ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "tempest_api_token": "",
    "ha_url": "",
    "ha_token": "",
    "collection_interval_min": 15,
    "retrain_interval_hours": 6,
    "advisor_interval_hours": 4,
    "min_training_samples": 48,
    "forecast_hours": 72,
    "claude_api_key": "",
    "yolink_temp_entity": "",
    "yolink_humidity_entity": "",
    "outdoor_temp_entity": "",
    "outdoor_humidity_entity": "",
    "avg_outdoor_temp_entity": "",
    "avg_outdoor_humidity_entity": "",
}

_SECRET_KEYS = {
    "WEATHERORACLE_TEMPEST_TOKEN": "tempest_api_token",
    "WEATHERORACLE_HA_TOKEN": "ha_token",
    "WEATHERORACLE_CLAUDE_KEY": "claude_api_key",
}


def load_config() -> dict:
    """Load config from file, then overlay environment variables for secrets."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            saved = json.load(f)
        cfg = {**DEFAULT_CONFIG, **saved}
    else:
        cfg = dict(DEFAULT_CONFIG)

    for env_key, cfg_key in _SECRET_KEYS.items():
        env_val = os.environ.get(env_key)
        if env_val:
            cfg[cfg_key] = env_val

    return cfg


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def load_locations() -> dict:
    """Load locations from config JSON. Falls back to defaults if unconfigured."""
    cfg = load_config()
    locs = cfg.get("locations")
    if locs and isinstance(locs, dict):
        _colors = [
            ("#00d4aa", "#008f72"), ("#4a90d9", "#2d5f9e"),
            ("#e0a030", "#b07820"), ("#d94a7a", "#9e2d55"),
        ]
        for i, (key, loc) in enumerate(locs.items()):
            c = _colors[i % len(_colors)]
            loc.setdefault("color", c[0])
            loc.setdefault("color_dark", c[1])
            loc.setdefault("nws_gridpoint", None)
            loc.setdefault("metar_station", "")
            loc.setdefault("nws_office", "")
        return locs
    return _DEFAULT_LOCATIONS


# Module-level — used everywhere
LOCATIONS = load_locations()
