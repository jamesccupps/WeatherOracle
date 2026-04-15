"""WeatherFlow Tempest REST API collector."""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("WeatherOracle.tempest")


def _c_to_f(c):
    return round(c * 9 / 5 + 32, 1) if c is not None else None


def _mps_to_mph(mps):
    return round(mps * 2.237, 1) if mps is not None else None


def _mm_to_in(mm):
    return round(mm * 0.03937, 3) if mm is not None else None


class TempestCollector:
    """Pull observations and forecasts from WeatherFlow Tempest."""

    BASE = "https://swd.weatherflow.com/swd/rest"

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def _parse_obs(self, o: dict) -> dict:
        """Convert a single Tempest observation dict to our format."""
        return {
            "temp_f": _c_to_f(o.get("air_temperature")),
            "humidity": o.get("relative_humidity"),
            "dewpoint_f": _c_to_f(o.get("dew_point")),
            "wind_mph": _mps_to_mph(o.get("wind_avg")),
            "wind_gust_mph": _mps_to_mph(o.get("wind_gust")),
            "wind_dir": o.get("wind_direction"),
            "pressure_mb": o.get("sea_level_pressure") or o.get("station_pressure"),
            "precip_in": _mm_to_in(o.get("precip_accum_local_day")),
            "solar_radiation": o.get("solar_radiation"),
            "uv_index": o.get("uv"),
            "feels_like_f": _c_to_f(o.get("feels_like")),
            "wet_bulb_f": _c_to_f(o.get("wet_bulb_temperature")),
            "timestamp": datetime.fromtimestamp(
                o.get("timestamp", 0), tz=timezone.utc
            ).isoformat(),
        }

    def get_current(self, station_id: int) -> Optional[dict]:
        """Latest observation from a Tempest station."""
        try:
            r = self.session.get(
                f"{self.BASE}/observations/station/{station_id}",
                params={"token": self.token}, timeout=15)
            r.raise_for_status()
            obs = r.json().get("obs", [])
            return self._parse_obs(obs[0]) if obs else None
        except Exception as e:
            log.error("Tempest current (station %s): %s", station_id, e)
            return None

    def get_history(self, station_id: int, days_back: int = 30) -> list:
        """Pull historical observations for backfilling."""
        results = []
        try:
            end = int(time.time())
            for d in range(days_back):
                day_start = end - (d + 1) * 86400
                day_end = day_start + 86400
                r = self.session.get(
                    f"{self.BASE}/observations/station/{station_id}",
                    params={"token": self.token,
                            "time_start": day_start, "time_end": day_end},
                    timeout=20)
                if r.status_code != 200:
                    continue
                for o in r.json().get("obs", []):
                    parsed = self._parse_obs(o)
                    # Override precip to per-observation not daily accum
                    parsed["precip_in"] = _mm_to_in(o.get("precip"))
                    results.append(parsed)
                time.sleep(0.3)
            log.info("Tempest history: %d obs from station %s (%d days)",
                     len(results), station_id, days_back)
        except Exception as e:
            log.error("Tempest history error: %s", e)
        return results

    def get_better_forecast(self, station_id: int) -> Optional[dict]:
        """Get Tempest's BetterForecast (their own blended forecast)."""
        try:
            r = self.session.get(
                f"{self.BASE}/better_forecast",
                params={
                    "station_id": station_id, "token": self.token,
                    "units_temp": "f", "units_wind": "mph",
                    "units_pressure": "mb", "units_precip": "in",
                    "units_distance": "mi",
                },
                timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error("Tempest BetterForecast error: %s", e)
            return None

    def get_better_forecast_hourly(self, station_id: int) -> Optional[list]:
        """Get Tempest BetterForecast parsed into standard hourly format.

        Returns list of dicts matching the Open-Meteo forecast format so it
        can be stored in the forecasts table alongside the other models.
        """
        raw = self.get_better_forecast(station_id)
        if not raw:
            return None

        forecast = raw.get("forecast", {})
        hourly = forecast.get("hourly", [])
        if not hourly:
            return None

        issued_at = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0).isoformat()
        now = datetime.now()

        results = []
        for h in hourly:
            ts = h.get("time")
            if ts is None:
                continue
            try:
                valid_dt = datetime.fromtimestamp(ts)
                lead = max(0, int((valid_dt - now).total_seconds() / 3600))
                valid_at = valid_dt.isoformat()
            except:
                continue

            # Tempest uses precip_probability (0-100) and precip_type
            # 0=none, 1=rain, 2=snow, 3=sleet
            precip_prob = h.get("precip_probability", 0)

            # Map conditions to WMO codes approximately
            conditions = h.get("conditions", "").lower()
            wmo = self._conditions_to_wmo(conditions, precip_prob)

            results.append({
                "issued_at": issued_at,
                "valid_at": valid_at,
                "lead_hours": lead,
                "temp_f": h.get("air_temperature"),
                "humidity": h.get("relative_humidity"),
                "dewpoint_f": h.get("dew_point"),  # already in °F with units_temp=f
                "wind_mph": h.get("wind_avg"),
                "wind_gust_mph": h.get("wind_gust"),
                "wind_dir": h.get("wind_direction"),
                "pressure_mb": h.get("sea_level_pressure"),
                "precip_prob": precip_prob,
                "precip_in": h.get("precip"),
                "cloud_cover": h.get("cloud_cover"),
                "weather_code": wmo,
            })

        log.info("Tempest BetterForecast station %s: %d hours", station_id, len(results))
        return results

    @staticmethod
    def _conditions_to_wmo(conditions: str, precip_prob: int = 0) -> int:
        """Approximate Tempest condition strings to WMO weather codes."""
        c = conditions.lower()
        if "thunder" in c:
            return 95
        if "heavy rain" in c or "heavy showers" in c:
            return 65
        if "rain" in c or "showers" in c:
            return 63
        if "drizzle" in c:
            return 53
        if "heavy snow" in c or "blizzard" in c:
            return 75
        if "snow" in c or "flurr" in c:
            return 73
        if "sleet" in c or "freezing" in c or "ice" in c:
            return 66
        if "fog" in c or "mist" in c:
            return 45
        if "overcast" in c or "cloudy" in c:
            return 3
        if "mostly cloudy" in c:
            return 3
        if "partly" in c:
            return 2
        if "mostly clear" in c:
            return 1
        if "clear" in c or "sunny" in c:
            return 0
        # Fallback: use precip probability
        if precip_prob > 50:
            return 63
        if precip_prob > 20:
            return 3
        return 2

    def test_connection(self, station_id: int) -> tuple:
        """Test API token + station. Returns (success, message)."""
        obs = self.get_current(station_id)
        if obs:
            return True, f"Connected: {obs['temp_f']}°F, {obs['humidity']}% RH"
        return False, "Could not reach Tempest API — check token"
