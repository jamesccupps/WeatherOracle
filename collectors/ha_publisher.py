"""Push WeatherOracle data into Home Assistant as sensors.

Creates/updates sensor entities dynamically based on configured locations:
  sensor.weatheroracle_{short}_current   - current conditions
  sensor.weatheroracle_{short}_forecast  - 72hr hourly forecast
  sensor.weatheroracle_{short}_daily     - 3-day daily outlook
  sensor.weatheroracle_status            - system status + ML info

All forecast data is stored in sensor attributes as JSON so the
Lovelace card can render it.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import requests

from core.config import LOCATIONS

log = logging.getLogger("WeatherOracle.ha_publish")

WMO_CODES = {
    0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime Fog", 51: "Lt Drizzle", 53: "Drizzle",
    55: "Hvy Drizzle", 56: "Frzg Drizzle", 57: "Hvy Frzg Drizzle",
    61: "Lt Rain", 63: "Rain", 65: "Heavy Rain",
    66: "Frzg Rain", 67: "Hvy Frzg Rain",
    71: "Lt Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
    80: "Lt Showers", 81: "Showers", 82: "Heavy Showers",
    85: "Lt Snow Shwrs", 86: "Snow Showers",
    95: "Thunderstorm", 96: "T-Storm+Hail", 99: "Severe T-Storm",
}

WMO_ICONS = {
    0: "☀️", 1: "🌤", 2: "⛅", 3: "☁️", 45: "🌫", 48: "🌫",
    51: "🌦", 53: "🌧", 55: "🌧", 56: "🌧", 57: "🌧",
    61: "🌦", 63: "🌧", 65: "🌧", 66: "🌧", 67: "🌧",
    71: "🌨", 73: "❄️", 75: "❄️", 77: "❄️",
    80: "🌦", 81: "🌧", 82: "⛈", 85: "🌨", 86: "🌨",
    95: "⛈", 96: "⛈", 99: "⛈",
}


class HAPublisher:
    """Push WeatherOracle data to Home Assistant as sensors."""

    def __init__(self, ha_url: str, ha_token: str):
        self.url = ha_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        })

    def _set_state(self, entity_id: str, state: str, attributes: dict):
        """Create or update a sensor in HA."""
        try:
            r = self.session.post(
                f"{self.url}/api/states/{entity_id}",
                json={"state": state, "attributes": attributes},
                timeout=10,
            )
            if r.status_code in (200, 201):
                return True
            log.warning("HA publish %s: HTTP %d", entity_id, r.status_code)
            return False
        except Exception as e:
            log.error("HA publish %s error: %s", entity_id, e)
            return False

    def publish_current(self, location: str, obs: dict, crosscheck: dict = None):
        """Publish current conditions for a location."""
        loc_short = LOCATIONS.get(location, {}).get("short", location[:3]).lower()
        loc_name = LOCATIONS.get(location, {}).get("name", location)
        entity = f"sensor.weatheroracle_{loc_short}_current"

        if not obs:
            return

        attrs = {
            "friendly_name": f"WeatherOracle {loc_name}",
            "icon": "mdi:weather-partly-cloudy",
            "device_class": "temperature",
            "unit_of_measurement": "°F",
            "location": location,
            "location_name": loc_name,
            "humidity": obs.get("humidity"),
            "dewpoint_f": obs.get("dewpoint_f"),
            "wind_mph": obs.get("wind_mph"),
            "wind_gust_mph": obs.get("wind_gust_mph"),
            "wind_dir": obs.get("wind_dir"),
            "pressure_mb": obs.get("pressure_mb"),
            "precip_in": obs.get("precip_in"),
            "solar_radiation": obs.get("solar_radiation"),
            "uv_index": obs.get("uv_index"),
            "feels_like_f": obs.get("feels_like_f"),
            "timestamp": obs.get("timestamp", ""),
        }

        if crosscheck:
            for src, data in crosscheck.items():
                attrs[f"xcheck_{src}_temp"] = data.get("temp_f")
                attrs[f"xcheck_{src}_humidity"] = data.get("humidity")

        state = str(obs.get("temp_f", "unknown"))
        self._set_state(entity, state, attrs)

    def publish_forecast(self, location: str, forecasts: list):
        """Publish hourly ensemble forecast as sensor attributes."""
        loc_short = LOCATIONS.get(location, {}).get("short", location[:3]).lower()
        loc_name = LOCATIONS.get(location, {}).get("short", location)
        entity = f"sensor.weatheroracle_{loc_short}_forecast"

        if not forecasts:
            return

        # Trim to essential fields to stay under HA attribute size limits
        hours = []
        for fc in forecasts[:72]:
            hours.append({
                "t": fc.get("valid_at", "")[:16],  # trim seconds
                "temp": fc.get("temp_f"),
                "hum": fc.get("humidity"),
                "wind": fc.get("wind_mph"),
                "gust": fc.get("wind_gust_mph"),
                "pp": fc.get("precip_prob"),
                "wc": fc.get("weather_code"),
                "conf": fc.get("confidence"),
                "pres": fc.get("pressure_mb"),
            })

        # State = next hour's temp
        state = str(hours[0]["temp"]) if hours else "unknown"

        attrs = {
            "friendly_name": f"WeatherOracle {loc_name} Forecast",
            "icon": "mdi:crystal-ball",
            "unit_of_measurement": "°F",
            "location": location,
            "hours": json.dumps(hours),
            "issued_at": forecasts[0].get("issued_at", ""),
            "method": forecasts[0].get("method", ""),
            "count": len(hours),
        }

        self._set_state(entity, state, attrs)

    def publish_daily(self, location: str, forecasts: list, oracle=None):
        """Aggregate hourly forecast into 3-day daily summaries.
        For today, includes actual observations so hi/lo reflects the full day."""
        loc_short = LOCATIONS.get(location, {}).get("short", location[:3]).lower()
        loc_name = LOCATIONS.get(location, {}).get("short", location)
        entity = f"sensor.weatheroracle_{loc_short}_daily"

        if not forecasts:
            return

        # Get today's actual observed temps from the orchestrator
        today_obs_temps = []
        today_str = datetime.now().strftime("%Y-%m-%d")
        try:
            # Pull today's Tempest observations for actual hi/lo
            obs = oracle.db.get_observations(
                location, source="tempest",
                since=f"{today_str}T00:00:00")
            for ob in obs:
                t = ob.get("temp_f")
                if t is not None:
                    today_obs_temps.append(t)
        except:
            pass

        # Group by date
        by_date = defaultdict(list)
        for fc in forecasts:
            try:
                dt = datetime.fromisoformat(fc["valid_at"][:19])
                key = dt.strftime("%Y-%m-%d")
                by_date[key].append({**fc, "_hour": dt.hour})
            except:
                pass

        days = []
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for date_key in sorted(by_date.keys())[:3]:
            hours = by_date[date_key]
            dt = datetime.strptime(date_key, "%Y-%m-%d")

            temps = [h.get("temp_f") for h in hours if h.get("temp_f") is not None]

            # For today: merge actual observed temps with forecast temps
            if date_key == today_str and today_obs_temps:
                temps = today_obs_temps + temps

            humids = [h.get("humidity") for h in hours if h.get("humidity") is not None]
            winds = [h.get("wind_mph") for h in hours if h.get("wind_mph") is not None]
            precips = [h.get("precip_prob") for h in hours if h.get("precip_prob") is not None]

            # Dominant weather code (midday preference)
            midday = [h for h in hours if 8 <= h["_hour"] <= 18]
            codes = [h.get("weather_code") for h in (midday or hours) if h.get("weather_code") is not None]
            code_counts = defaultdict(int)
            for c in codes:
                code_counts[c] += 1
            dom_code = max(code_counts, key=code_counts.get) if code_counts else None

            days.append({
                "date": date_key,
                "day": day_names[dt.weekday()],
                "label": f"{month_names[dt.month-1]} {dt.day}",
                "hi": round(max(temps)) if temps else None,
                "lo": round(min(temps)) if temps else None,
                "hum": round(sum(humids)/len(humids)) if humids else None,
                "wind": round(max(winds)) if winds else None,
                "precip": round(max(precips)) if precips else None,
                "wc": dom_code,
                "icon": WMO_ICONS.get(dom_code, ""),
                "condition": WMO_CODES.get(dom_code, ""),
            })

        # State = today's high
        state = str(days[0]["hi"]) if days and days[0]["hi"] else "unknown"

        attrs = {
            "friendly_name": f"WeatherOracle {loc_name} 3-Day",
            "icon": "mdi:calendar-week",
            "unit_of_measurement": "°F",
            "location": location,
            "days": json.dumps(days),
            "count": len(days),
        }

        self._set_state(entity, state, attrs)

    def publish_status(self, db_stats: dict, ml_models_apt: int,
                       ml_models_occ: int):
        """Publish system status sensor."""
        entity = "sensor.weatheroracle_status"

        ml_total = ml_models_apt + ml_models_occ
        state = "active" if ml_total > 0 else "collecting"

        attrs = {
            "friendly_name": "WeatherOracle Status",
            "icon": "mdi:brain",
            "observations": db_stats.get("observations", 0),
            "forecasts": db_stats.get("forecasts", 0),
            "ensemble_forecasts": db_stats.get("ensemble_forecasts", 0),
            "ml_models_apartment": ml_models_apt,
            "ml_models_occ": ml_models_occ,
            "ml_models_total": ml_total,
        }

        self._set_state(entity, state, attrs)

    def publish_all(self, oracle):
        """Convenience: publish everything from an Oracle instance."""
        try:
            for loc_key in ("apartment", "occ"):
                # Current conditions
                obs = oracle.get_current(loc_key)
                xcheck = oracle.get_crosscheck(loc_key) if loc_key == "apartment" else {}
                if obs:
                    self.publish_current(loc_key, obs, xcheck)

                # Hourly forecast
                fc = oracle.get_forecast(loc_key)
                if fc:
                    self.publish_forecast(loc_key, fc)
                    self.publish_daily(loc_key, fc, oracle)

            # Status
            stats = oracle.db.get_db_stats()
            self.publish_status(
                stats,
                oracle.ml.get_model_count("apartment"),
                oracle.ml.get_model_count("occ"),
            )

            log.info("[HA] Published all sensors")
        except Exception as e:
            log.error("[HA] Publish error: %s", e)
