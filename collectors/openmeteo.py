"""Open-Meteo multi-model forecast collector."""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from core.config import WEATHER_MODELS, HOURLY_PARAMS

log = logging.getLogger("WeatherOracle.openmeteo")


class OpenMeteoCollector:
    """Pull hourly forecasts from 6 weather models via Open-Meteo."""

    def get_forecast(self, model_key: str, lat: float, lon: float,
                     hours: int = 72) -> Optional[list]:
        """Fetch hourly forecast from one model."""
        info = WEATHER_MODELS.get(model_key)
        if not info:
            return None
        try:
            # Respect per-model max forecast hours
            effective_hours = min(hours, info.get("max_hours", hours))
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": HOURLY_PARAMS,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "forecast_hours": effective_hours,
                "timezone": "America/New_York",
            }
            # Some models require an explicit model selector (e.g. HRRR)
            if info.get("api_model"):
                params["models"] = info["api_model"]
            r = requests.get(info["url"], params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            n = len(times)

            issued_at = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0).isoformat()
            now = datetime.now()

            def _get(key, i):
                arr = hourly.get(key, [None] * n)
                return arr[i] if i < len(arr) else None

            results = []
            for i, t in enumerate(times):
                try:
                    valid_dt = datetime.fromisoformat(t)
                except:
                    continue
                lead = max(0, int((valid_dt - now).total_seconds() / 3600))
                results.append({
                    "issued_at": issued_at,
                    "valid_at": valid_dt.isoformat(),
                    "lead_hours": lead,
                    "temp_f": _get("temperature_2m", i),
                    "humidity": _get("relative_humidity_2m", i),
                    "dewpoint_f": _get("dew_point_2m", i),
                    "wind_mph": _get("wind_speed_10m", i),
                    "wind_gust_mph": _get("wind_gusts_10m", i),
                    "wind_dir": _get("wind_direction_10m", i),
                    "pressure_mb": _get("pressure_msl", i),
                    "precip_prob": _get("precipitation_probability", i),
                    "precip_in": _get("precipitation", i),
                    "cloud_cover": _get("cloud_cover", i),
                    "weather_code": _get("weather_code", i),
                })
            return results
        except Exception as e:
            log.error("Open-Meteo %s error: %s", model_key, e)
            return None

    def get_all_forecasts(self, lat: float, lon: float,
                          hours: int = 72) -> dict:
        """Fetch forecasts from all models. Returns {model_key: [forecasts]}."""
        results = {}
        for mk in WEATHER_MODELS:
            fcs = self.get_forecast(mk, lat, lon, hours)
            if fcs:
                results[mk] = fcs
            time.sleep(0.4)  # Rate limit
        return results

    def get_historical(self, lat: float, lon: float,
                       start_date: str, end_date: str) -> Optional[list]:
        """Pull archived weather observations from Open-Meteo."""
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,"
                          "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                          "pressure_msl,precipitation,cloud_cover,weather_code",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "America/New_York",
            }
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params=params, timeout=30)
            if r.status_code != 200:
                return None
            data = r.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            n = len(times)

            def _get(key, i):
                arr = hourly.get(key, [None] * n)
                return arr[i] if i < len(arr) else None

            return [{
                "valid_at": times[i],
                "temp_f": _get("temperature_2m", i),
                "humidity": _get("relative_humidity_2m", i),
                "dewpoint_f": _get("dew_point_2m", i),
                "wind_mph": _get("wind_speed_10m", i),
                "wind_gust_mph": _get("wind_gusts_10m", i),
                "wind_dir": _get("wind_direction_10m", i),
                "pressure_mb": _get("pressure_msl", i),
                "precip_in": _get("precipitation", i),
                "cloud_cover": _get("cloud_cover", i),
                "weather_code": _get("weather_code", i),
            } for i in range(n)]
        except Exception as e:
            log.error("Open-Meteo historical error: %s", e)
            return None
