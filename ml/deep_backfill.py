"""Deep historical backfill — pull years of observation + forecast data
for ML training. Pairs what each model predicted with what actually happened
at each Tempest station.

Data sources:
  - Tempest API: actual observations (ground truth) — goes back to install date
  - Open-Meteo Archive API: hourly historical observations (ERA5 reanalysis)
    as the primary high-resolution ground truth for training
  - Open-Meteo Historical Forecast API: what each model actually predicted
    historically — this is the forecast side of the training pair
"""

import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

import requests

from core.config import LOCATIONS, WEATHER_MODELS
from core.database import WeatherDB

log = logging.getLogger("WeatherOracle.backfill")

# Chunking and rate limiting
CHUNK_DAYS = 3  # 3-day chunks so lead_hours stay realistic (0-72)
ARCHIVE_CHUNK_DAYS = 90  # 90-day chunks for archive API
RATE_LIMIT_DELAY = 0.6

# Open-Meteo Historical Forecast API — what each model ACTUALLY predicted
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Model names as used in the Historical Forecast API's `models=` parameter
# Response columns will be named like: temperature_2m_gfs_seamless
MODEL_PARAMS = {
    "gfs":   {"param": "gfs_global",        "available_from": "2021-03-01"},
    "hrrr":  {"param": "ncep_hrrr_conus",   "available_from": "2022-06-08"},
    "ecmwf": {"param": "ecmwf_ifs025",      "available_from": "2022-06-08"},
    "icon":  {"param": "icon_seamless",     "available_from": "2022-06-08"},
    "gem":   {"param": "gem_seamless",       "available_from": "2022-06-08"},
    "jma":   {"param": "jma_seamless",       "available_from": "2024-01-01"},
}

# Variables to fetch from Historical Forecast API
HIST_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,"
    "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
    "pressure_msl,precipitation,cloud_cover,weather_code"
)

# Variables for archive API (ground truth)
ARCHIVE_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,"
    "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
    "pressure_msl,precipitation,cloud_cover,weather_code"
)

# Mapping from Open-Meteo field names to our DB columns
FIELD_MAP = {
    "temperature_2m": "temp_f",
    "relative_humidity_2m": "humidity",
    "dew_point_2m": "dewpoint_f",
    "wind_speed_10m": "wind_mph",
    "wind_gusts_10m": "wind_gust_mph",
    "wind_direction_10m": "wind_dir",
    "pressure_msl": "pressure_mb",
    "precipitation": "precip_in",
    "cloud_cover": "cloud_cover",
    "weather_code": "weather_code",
}


class DeepBackfill:
    """Pulls years of historical data and pairs observations with forecasts."""

    def __init__(self, db: WeatherDB, tempest_token: str,
                 ha_url: str = "", ha_token: str = ""):
        self.db = db
        self.tempest_token = tempest_token
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self._cancel = False
        self._progress_cb = None

    def cancel(self):
        self._cancel = True

    def set_progress_callback(self, cb: Callable):
        self._progress_cb = cb

    def _progress(self, phase: str, pct: float, msg: str):
        log.info("[%s %.0f%%] %s", phase, pct, msg)
        if self._progress_cb:
            try:
                self._progress_cb(phase, pct, msg)
            except:
                pass

    # ── Phase 1: Tempest Station History ──────────────────────────────────

    def backfill_tempest(self, location: str, station_id: int,
                         days_back: int = 730) -> int:
        """Pull Tempest station history. Note: the station observations API
        returns ~1 summary per day for day-long windows. This gives us daily
        ground truth from the physical station — supplemented by Open-Meteo
        archive hourly data in Phase 2."""
        self._cancel = False
        total_obs = 0
        end_ts = int(time.time())

        self._progress("tempest", 0,
                       f"Fetching {days_back} days from station #{station_id}")

        for day in range(days_back):
            if self._cancel:
                self._progress("tempest", 100, "Cancelled")
                return total_obs

            day_start = end_ts - (day + 1) * 86400
            day_end = day_start + 86400

            try:
                r = self.session.get(
                    f"https://swd.weatherflow.com/swd/rest/observations/station/{station_id}",
                    params={"token": self.tempest_token,
                            "time_start": day_start, "time_end": day_end},
                    timeout=20)

                if r.status_code == 429:
                    log.warning("Tempest rate limited, sleeping 30s")
                    time.sleep(30)
                    continue
                if r.status_code != 200:
                    continue

                for o in r.json().get("obs", []):
                    ts = datetime.fromtimestamp(
                        o.get("timestamp", 0), tz=timezone.utc).isoformat()

                    def _c2f(c):
                        return round(c * 9/5 + 32, 1) if c is not None else None
                    def _mps(m):
                        return round(m * 2.237, 1) if m is not None else None

                    self.db.insert_observation(
                        location, ts, "tempest",
                        temp_f=_c2f(o.get("air_temperature")),
                        humidity=o.get("relative_humidity"),
                        dewpoint_f=_c2f(o.get("dew_point")),
                        wind_mph=_mps(o.get("wind_avg")),
                        wind_gust_mph=_mps(o.get("wind_gust")),
                        wind_dir=o.get("wind_direction"),
                        pressure_mb=o.get("sea_level_pressure") or o.get("station_pressure"),
                        precip_in=round(o.get("precip", 0) * 0.03937, 3) if o.get("precip") is not None else None,
                        solar_radiation=o.get("solar_radiation"),
                        uv_index=o.get("uv"),
                        feels_like_f=_c2f(o.get("feels_like")),
                        wet_bulb_f=_c2f(o.get("wet_bulb_temperature")))
                    total_obs += 1

            except requests.exceptions.RequestException as e:
                log.warning("Tempest day %d error: %s", day, e)
                time.sleep(2)
                continue

            pct = (day + 1) / days_back * 100
            if day % 30 == 0 or pct >= 100:
                dt = datetime.fromtimestamp(day_start)
                self._progress("tempest", pct,
                               f"Station #{station_id}: {dt.strftime('%Y-%m-%d')} — "
                               f"{total_obs:,} obs")
            time.sleep(RATE_LIMIT_DELAY)

        self._progress("tempest", 100,
                       f"Complete: {total_obs:,} observations from station #{station_id}")
        return total_obs

    # ── Phase 2: Open-Meteo Archive (hourly ground truth) ─────────────────

    def backfill_archive(self, location: str, lat: float, lon: float,
                         start_date: str, end_date: str) -> int:
        """Pull Open-Meteo ERA5 archive — hourly ground truth for training.
        This is the primary source of hourly observation data for pairing
        against model forecasts."""
        self._cancel = False
        total_obs = 0
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        self._progress("archive", 0,
                       f"Fetching ERA5 archive {start_date} → {end_date}")

        chunk_start = start_dt
        while chunk_start < end_dt:
            if self._cancel:
                return total_obs

            chunk_end = min(chunk_start + timedelta(days=ARCHIVE_CHUNK_DAYS), end_dt)

            try:
                r = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat, "longitude": lon,
                        "hourly": ARCHIVE_HOURLY_VARS,
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "precipitation_unit": "inch",
                        "start_date": chunk_start.strftime("%Y-%m-%d"),
                        "end_date": chunk_end.strftime("%Y-%m-%d"),
                        "timezone": "America/New_York",
                    }, timeout=30)

                if r.status_code == 429:
                    time.sleep(60); continue
                if r.status_code != 200:
                    chunk_start = chunk_end; continue

                hourly = r.json().get("hourly", {})
                times = hourly.get("time", [])
                n = len(times)

                def _g(key, i):
                    arr = hourly.get(key, [None] * n)
                    return arr[i] if i < len(arr) else None

                for i in range(n):
                    self.db.insert_observation(
                        location, times[i], "openmeteo_archive",
                        temp_f=_g("temperature_2m", i),
                        humidity=_g("relative_humidity_2m", i),
                        dewpoint_f=_g("dew_point_2m", i),
                        wind_mph=_g("wind_speed_10m", i),
                        wind_gust_mph=_g("wind_gusts_10m", i),
                        wind_dir=_g("wind_direction_10m", i),
                        pressure_mb=_g("pressure_msl", i),
                        precip_in=_g("precipitation", i),
                        cloud_cover=_g("cloud_cover", i))
                    total_obs += 1

            except Exception as e:
                log.warning("Archive chunk %s error: %s", chunk_start, e)
                time.sleep(2)

            elapsed = (chunk_end - start_dt).days
            pct = min(100, elapsed / max(total_days, 1) * 100)
            self._progress("archive", pct,
                           f"{chunk_start.strftime('%Y-%m-%d')}: {total_obs:,} obs")
            chunk_start = chunk_end
            time.sleep(RATE_LIMIT_DELAY)

        self._progress("archive", 100, f"Archive complete: {total_obs:,} hourly observations")
        return total_obs

    # ── Phase 3: Historical Model Forecasts ───────────────────────────────

    def backfill_model_forecasts(self, location: str, lat: float, lon: float,
                                  start_date: str, end_date: str) -> int:
        """Pull what each weather model ACTUALLY predicted historically.

        Uses Open-Meteo Historical Forecast API which stores past model runs.
        Fetches all models in a single request per chunk (multi-model mode),
        then splits the response into per-model forecast rows.

        The API returns columns like temperature_2m_gfs_seamless,
        temperature_2m_ecmwf_ifs025, etc.
        """
        self._cancel = False
        total_forecasts = 0

        # Determine actual start based on model availability
        earliest_available = min(m["available_from"] for m in MODEL_PARAMS.values())
        actual_start = max(start_date, earliest_available)
        if actual_start >= end_date:
            self._progress("forecasts", 100, "No historical forecast data available")
            return 0

        start_dt = datetime.strptime(actual_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        # Build models parameter string
        all_model_params = ",".join(m["param"] for m in MODEL_PARAMS.values())

        self._progress("forecasts", 0,
                       f"Fetching 5 models {actual_start} → {end_date}")

        chunk_start = start_dt
        chunk_num = 0
        total_chunks = total_days // CHUNK_DAYS + 1

        while chunk_start < end_dt:
            if self._cancel:
                return total_forecasts

            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
            chunk_num += 1

            try:
                # Single API call fetches ALL models for this chunk
                r = requests.get(HIST_FORECAST_URL, params={
                    "latitude": lat, "longitude": lon,
                    "hourly": HIST_HOURLY_VARS,
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "precipitation_unit": "inch",
                    "start_date": chunk_start.strftime("%Y-%m-%d"),
                    "end_date": chunk_end.strftime("%Y-%m-%d"),
                    "models": all_model_params,
                    "timezone": "America/New_York",
                }, timeout=45)

                if r.status_code == 429:
                    log.warning("Historical forecast API rate limited, sleeping 60s")
                    time.sleep(60)
                    continue
                if r.status_code != 200:
                    log.warning("Historical forecast chunk %s: HTTP %d — %s",
                                chunk_start.strftime("%Y-%m-%d"), r.status_code,
                                r.text[:200])
                    chunk_start = chunk_end
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

                hourly = r.json().get("hourly", {})
                times = hourly.get("time", [])
                n = len(times)

                if n == 0:
                    chunk_start = chunk_end
                    continue

                issued_at = chunk_start.strftime("%Y-%m-%dT00:00:00")

                # Parse per-model data from multi-model response
                # Columns are named like: temperature_2m_gfs_seamless
                for our_key, model_info in MODEL_PARAMS.items():
                    suffix = model_info["param"]

                    # Check if this model has data in this chunk
                    model_available = datetime.strptime(
                        model_info["available_from"], "%Y-%m-%d")
                    if chunk_start < model_available:
                        continue

                    for i in range(n):
                        valid_at = times[i]
                        try:
                            valid_dt = datetime.fromisoformat(valid_at)
                            lead = max(0, int((valid_dt - chunk_start).total_seconds() / 3600))
                        except:
                            lead = i

                        # Extract this model's values for each variable
                        def _get_model_val(om_var):
                            """Get value from multi-model column."""
                            col = f"{om_var}_{suffix}"
                            arr = hourly.get(col, [])
                            if i < len(arr):
                                return arr[i]
                            # Fallback: single-model column (if only 1 model returned)
                            arr2 = hourly.get(om_var, [])
                            if i < len(arr2):
                                return arr2[i]
                            return None

                        self.db.insert_forecast(
                            location, our_key, issued_at, valid_at, lead,
                            temp_f=_get_model_val("temperature_2m"),
                            humidity=_get_model_val("relative_humidity_2m"),
                            dewpoint_f=_get_model_val("dew_point_2m"),
                            wind_mph=_get_model_val("wind_speed_10m"),
                            wind_gust_mph=_get_model_val("wind_gusts_10m"),
                            wind_dir=_get_model_val("wind_direction_10m"),
                            pressure_mb=_get_model_val("pressure_msl"),
                            precip_in=_get_model_val("precipitation"),
                            cloud_cover=_get_model_val("cloud_cover"),
                            weather_code=_get_model_val("weather_code"))
                        total_forecasts += 1

            except Exception as e:
                log.warning("Historical forecast chunk %s error: %s",
                            chunk_start.strftime("%Y-%m-%d"), e)
                time.sleep(2)

            pct = min(100, chunk_num / max(total_chunks, 1) * 100)
            self._progress("forecasts", pct,
                           f"{chunk_start.strftime('%Y-%m-%d')}: "
                           f"{total_forecasts:,} forecast hours stored")

            chunk_start = chunk_end
            time.sleep(RATE_LIMIT_DELAY)

        self._progress("forecasts", 100,
                       f"Complete: {total_forecasts:,} forecast hours across {len(MODEL_PARAMS)} models")
        return total_forecasts

    # ── Phase 4: Home Assistant Sensor History ────────────────────────────

    def backfill_ha_history(self, location: str, days_back: int = 365) -> int:
        """Pull long-range history from HA sensors as ground truth."""
        if not self.ha_url or not self.ha_token:
            self._progress("ha_history", 100, "No HA credentials — skipping")
            return 0

        from collectors.homeassistant import HACollector
        from core.config import load_config
        ha = HACollector(self.ha_url, self.ha_token)
        cfg = load_config()

        # Build sensor pairs from config (only if entities are configured)
        sensor_pairs = []
        pairs_config = [
            ("outdoor_temp_entity", "outdoor_humidity_entity", "ha_outdoor_sensor"),
            ("yolink_temp_entity", "yolink_humidity_entity", "ha_yolink"),
            ("avg_outdoor_temp_entity", "avg_outdoor_humidity_entity", "ha_average"),
        ]
        for temp_key, hum_key, src_name in pairs_config:
            temp_eid = cfg.get(temp_key, "")
            hum_eid = cfg.get(hum_key, "")
            if temp_eid:
                sensor_pairs.append((temp_eid, hum_eid, src_name))

        if not sensor_pairs:
            self._progress("ha_history", 100, f"No HA sensors for {location}")
            return 0

        self._cancel = False
        total_obs = 0

        for idx, (temp_eid, hum_eid, src_name) in enumerate(sensor_pairs):
            if self._cancel:
                return total_obs

            self._progress("ha_history", (idx / len(sensor_pairs)) * 100,
                           f"Pulling {src_name} ({days_back} days)...")

            temp_recs = ha.get_long_history(temp_eid, days_back, chunk_days=7)
            hum_recs = ha.get_long_history(hum_eid, days_back, chunk_days=7)

            # Build humidity lookup by hour
            hum_map = {}
            for h in hum_recs:
                ts = h.get("last_changed", "")[:13]
                if ts:
                    hum_map[ts] = h["state"]

            seen = set()
            count = 0
            for t in temp_recs:
                ts = t.get("last_changed", "")
                hour_key = ts[:13]
                if not hour_key or hour_key in seen:
                    continue
                seen.add(hour_key)

                self.db.insert_observation(
                    location, ts, f"ha_{src_name}",
                    temp_f=t["state"],
                    humidity=hum_map.get(hour_key))
                count += 1
                total_obs += 1

            self._progress("ha_history", ((idx + 1) / len(sensor_pairs)) * 100,
                           f"{src_name}: {count} hourly records")

        self._progress("ha_history", 100, f"HA history: {total_obs} observations")
        return total_obs

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def run_full_backfill(self, days_back: int = 730,
                          include_archive: bool = True,
                          include_ha: bool = True) -> dict:
        """Run complete deep backfill for both locations (4 phases).
        Uses batch mode for 10-50x faster inserts."""
        self._cancel = False
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        results = {}
        n_phases = 2 + int(include_archive) + int(include_ha)

        for loc_key, loc in LOCATIONS.items():
            if self._cancel: break
            self._progress("overall", 0, f"Starting {loc['name']}...")
            loc_results = {"tempest_obs": 0, "archive_obs": 0,
                          "forecasts": 0, "ha_obs": 0}
            phase = 0

            # Enable batch mode for bulk inserts
            self.db.begin_batch()

            try:
                phase += 1
                self._progress("overall", 0,
                              f"Phase {phase}/{n_phases}: Tempest for {loc['name']}")
                loc_results["tempest_obs"] = self.backfill_tempest(
                    loc_key, loc["tempest_station"], days_back)
                self.db.end_batch()  # commit Tempest phase
                if self._cancel: break

                if include_archive:
                    self.db.begin_batch()
                    phase += 1
                    self._progress("overall", (phase-1)/n_phases*100,
                                  f"Phase {phase}/{n_phases}: ERA5 for {loc['name']}")
                    loc_results["archive_obs"] = self.backfill_archive(
                        loc_key, loc["lat"], loc["lon"], start_date, end_date)
                    self.db.end_batch()
                if self._cancel: break

                self.db.begin_batch()
                phase += 1
                self._progress("overall", (phase-1)/n_phases*100,
                              f"Phase {phase}/{n_phases}: Forecasts for {loc['name']}")
                loc_results["forecasts"] = self.backfill_model_forecasts(
                    loc_key, loc["lat"], loc["lon"], start_date, end_date)
                self.db.end_batch()

                # Clean up bad lead_hours after forecast backfill
                self.db.cleanup_bad_forecasts()

                if self._cancel: break

                if include_ha:
                    self.db.begin_batch()
                    phase += 1
                    self._progress("overall", (phase-1)/n_phases*100,
                                  f"Phase {phase}/{n_phases}: HA history for {loc['name']}")
                    loc_results["ha_obs"] = self.backfill_ha_history(loc_key, days_back)
                    self.db.end_batch()

            except Exception as e:
                # Ensure batch mode is cleared on error
                self.db.end_batch()
                log.error("Backfill error for %s: %s", loc_key, e)

            results[loc_key] = loc_results

        self._progress("overall", 100, "Deep backfill complete!")
        return results

    def estimate_time(self, days_back: int) -> dict:
        """Estimate time and data volume for a backfill."""
        # Tempest: 1 call/day × 2 stations
        tempest_calls = days_back * 2
        tempest_time = tempest_calls * RATE_LIMIT_DELAY

        # Archive: 1 call per 90-day chunk × 2 stations
        archive_chunks = (days_back // ARCHIVE_CHUNK_DAYS + 1) * 2
        archive_time = archive_chunks * RATE_LIMIT_DELAY

        # Historical forecasts: 1 call per 30-day chunk × 2 stations
        # (all models in single call!)
        forecast_chunks = (days_back // CHUNK_DAYS + 1) * 2
        forecast_time = forecast_chunks * RATE_LIMIT_DELAY

        total_minutes = (tempest_time + archive_time + forecast_time) / 60

        est_obs = days_back * 24 * 2  # hourly archive × 2 stations
        est_forecasts = days_back * 24 * len(MODEL_PARAMS) * 2

        return {
            "days": days_back,
            "est_minutes": round(total_minutes, 0),
            "est_observations": est_obs,
            "est_forecasts": est_forecasts,
            "est_total_rows": est_obs + est_forecasts,
            "tempest_api_calls": tempest_calls,
            "openmeteo_api_calls": archive_chunks + forecast_chunks,
        }


class BackfillThread(threading.Thread):
    """Threaded wrapper for deep backfill with progress callbacks."""

    def __init__(self, db: WeatherDB, tempest_token: str,
                 days_back: int = 730, include_archive: bool = True,
                 include_ha: bool = True,
                 ha_url: str = "", ha_token: str = "",
                 on_progress: Callable = None,
                 on_complete: Callable = None):
        super().__init__(daemon=True)
        self.backfiller = DeepBackfill(db, tempest_token, ha_url, ha_token)
        self.days_back = days_back
        self.include_archive = include_archive
        self.include_ha = include_ha
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.results = None

        if on_progress:
            self.backfiller.set_progress_callback(on_progress)

    def run(self):
        try:
            self.results = self.backfiller.run_full_backfill(
                self.days_back, self.include_archive, self.include_ha)
        except Exception as e:
            log.error("Backfill thread error: %s", e, exc_info=True)
            self.results = {"error": str(e)}

        if self.on_complete:
            try:
                self.on_complete(self.results)
            except:
                pass

    def cancel(self):
        self.backfiller.cancel()
