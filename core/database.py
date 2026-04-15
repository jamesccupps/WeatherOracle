"""SQLite database for observations, forecasts, model accuracy, and training logs."""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import DB_PATH, WEATHER_MODELS

log = logging.getLogger("WeatherOracle.db")


class WeatherDB:
    """Thread-safe SQLite database for all weather data."""

    def __init__(self, path: Path = None):
        self.path = path or DB_PATH
        self.path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.Lock()
        self._batch_mode = False
        self._create_tables()

    def begin_batch(self):
        """Start batch mode — commits are deferred until end_batch()."""
        self._batch_mode = True

    def end_batch(self):
        """End batch mode and commit all pending changes."""
        self._batch_mode = False
        with self.lock:
            self.conn.commit()

    def _maybe_commit(self):
        """Commit unless in batch mode."""
        if not self._batch_mode:
            self.conn.commit()

    def _create_tables(self):
        with self.lock:
            c = self.conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    temp_f REAL, humidity REAL, dewpoint_f REAL,
                    wind_mph REAL, wind_gust_mph REAL, wind_dir INTEGER,
                    pressure_mb REAL, precip_in REAL,
                    solar_radiation REAL, uv_index REAL,
                    feels_like_f REAL, cloud_cover REAL,
                    wet_bulb_f REAL,
                    UNIQUE(location, timestamp, source)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    model TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    valid_at TEXT NOT NULL,
                    lead_hours INTEGER NOT NULL,
                    temp_f REAL, humidity REAL, dewpoint_f REAL,
                    wind_mph REAL, wind_gust_mph REAL, wind_dir INTEGER,
                    pressure_mb REAL, precip_prob REAL, precip_in REAL,
                    cloud_cover REAL, weather_code INTEGER,
                    UNIQUE(location, model, issued_at, valid_at)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    valid_at TEXT NOT NULL,
                    lead_hours INTEGER NOT NULL,
                    temp_f REAL, humidity REAL, dewpoint_f REAL,
                    wind_mph REAL, wind_gust_mph REAL,
                    precip_prob REAL, precip_in REAL,
                    cloud_cover REAL, weather_code INTEGER,
                    feels_like_f REAL, pressure_mb REAL,
                    confidence REAL, method TEXT,
                    UNIQUE(location, issued_at, valid_at)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS model_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    model TEXT NOT NULL,
                    variable TEXT NOT NULL,
                    lead_hours_bucket TEXT NOT NULL,
                    mae REAL, rmse REAL, bias REAL,
                    n_samples INTEGER,
                    computed_at TEXT NOT NULL
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    variable TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    n_samples INTEGER, cv_mae REAL,
                    ensemble_mae REAL, best_individual TEXT,
                    best_individual_mae REAL, improvement_pct REAL
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS bias_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    hour_of_day INTEGER,
                    month INTEGER,
                    day_of_year INTEGER,
                    obs_temp_f REAL,
                    obs_humidity REAL,
                    obs_wind_mph REAL,
                    obs_pressure_mb REAL,
                    ensemble_temp_f REAL,
                    ensemble_humidity REAL,
                    ensemble_wind_mph REAL,
                    bias_temp_f REAL,
                    bias_humidity REAL,
                    bias_wind_mph REAL,
                    gfs_temp_f REAL, gfs_error REAL,
                    hrrr_temp_f REAL, hrrr_error REAL,
                    ecmwf_temp_f REAL, ecmwf_error REAL,
                    icon_temp_f REAL, icon_error REAL,
                    gem_temp_f REAL, gem_error REAL,
                    jma_temp_f REAL, jma_error REAL,
                    UNIQUE(location, timestamp)
                )
            """)

            # Indexes for fast queries
            c.execute("CREATE INDEX IF NOT EXISTS idx_obs_loc_ts ON observations(location, timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_obs_loc_src ON observations(location, source)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_fc_loc_model ON forecasts(location, model, valid_at)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ens_loc ON ensemble_forecasts(location, issued_at)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_bias_loc_ts ON bias_log(location, timestamp)")

            self._maybe_commit()

    # ── Insert methods ────────────────────────────────────────────────────

    # Physically plausible ranges for southern Maine
    _OBS_RANGES = {
        "temp_f": (-60, 130),
        "humidity": (0, 100),
        "dewpoint_f": (-80, 100),
        "wind_mph": (0, 200),
        "wind_gust_mph": (0, 250),
        "wind_dir": (0, 360),
        "pressure_mb": (870, 1084),
        "precip_in": (0, 20),
        "solar_radiation": (0, 2000),
        "uv_index": (0, 16),
        "feels_like_f": (-100, 160),
        "cloud_cover": (0, 100),
        "wet_bulb_f": (-60, 110),
    }

    def _validate_obs(self, kw: dict) -> dict:
        """Reject (set to None) observation values outside physical bounds."""
        for key, (lo, hi) in self._OBS_RANGES.items():
            val = kw.get(key)
            if val is not None:
                try:
                    val = float(val)
                    if val < lo or val > hi:
                        log.warning("Obs validation: %s=%.1f outside [%s,%s] — rejected",
                                    key, val, lo, hi)
                        kw[key] = None
                except (ValueError, TypeError):
                    kw[key] = None
        return kw

    def insert_observation(self, location: str, ts: str, source: str, **kw):
        kw = self._validate_obs(kw)
        with self.lock:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO observations
                    (location, timestamp, source, temp_f, humidity, dewpoint_f,
                     wind_mph, wind_gust_mph, wind_dir, pressure_mb, precip_in,
                     solar_radiation, uv_index, feels_like_f, cloud_cover, wet_bulb_f)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (location, ts, source,
                      kw.get("temp_f"), kw.get("humidity"), kw.get("dewpoint_f"),
                      kw.get("wind_mph"), kw.get("wind_gust_mph"), kw.get("wind_dir"),
                      kw.get("pressure_mb"), kw.get("precip_in"),
                      kw.get("solar_radiation"), kw.get("uv_index"),
                      kw.get("feels_like_f"), kw.get("cloud_cover"),
                      kw.get("wet_bulb_f")))
                self._maybe_commit()
            except Exception as e:
                log.error("Insert obs error: %s", e)

    def insert_forecast(self, location: str, model: str, issued_at: str,
                        valid_at: str, lead_hours: int, **kw):
        with self.lock:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO forecasts
                    (location, model, issued_at, valid_at, lead_hours,
                     temp_f, humidity, dewpoint_f, wind_mph, wind_gust_mph,
                     wind_dir, pressure_mb, precip_prob, precip_in,
                     cloud_cover, weather_code)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (location, model, issued_at, valid_at, lead_hours,
                      kw.get("temp_f"), kw.get("humidity"), kw.get("dewpoint_f"),
                      kw.get("wind_mph"), kw.get("wind_gust_mph"), kw.get("wind_dir"),
                      kw.get("pressure_mb"), kw.get("precip_prob"), kw.get("precip_in"),
                      kw.get("cloud_cover"), kw.get("weather_code")))
                self._maybe_commit()
            except Exception as e:
                log.error("Insert forecast error: %s", e)

    def insert_ensemble(self, location: str, issued_at: str, valid_at: str,
                        lead_hours: int, method: str, **kw):
        with self.lock:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO ensemble_forecasts
                    (location, issued_at, valid_at, lead_hours,
                     temp_f, humidity, dewpoint_f, wind_mph, wind_gust_mph,
                     precip_prob, precip_in, cloud_cover, weather_code,
                     feels_like_f, pressure_mb, confidence, method)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (location, issued_at, valid_at, lead_hours,
                      kw.get("temp_f"), kw.get("humidity"), kw.get("dewpoint_f"),
                      kw.get("wind_mph"), kw.get("wind_gust_mph"),
                      kw.get("precip_prob"), kw.get("precip_in"),
                      kw.get("cloud_cover"), kw.get("weather_code"),
                      kw.get("feels_like_f"), kw.get("pressure_mb"),
                      kw.get("confidence"), method))
                self._maybe_commit()
            except Exception as e:
                log.error("Insert ensemble error: %s", e)

    def insert_bias(self, location: str, ts: str, obs: dict,
                    ensemble_preds: dict, per_model: dict):
        """Log bias correction data for ML feedback.

        obs: {temp_f, humidity, wind_mph, pressure_mb}
        ensemble_preds: {temp_f, humidity, wind_mph}
        per_model: {model_key: {temp_f: predicted, error: obs-pred}, ...}
        """
        with self.lock:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                self.conn.execute("""
                    INSERT OR REPLACE INTO bias_log
                    (location, timestamp, hour_of_day, month, day_of_year,
                     obs_temp_f, obs_humidity, obs_wind_mph, obs_pressure_mb,
                     ensemble_temp_f, ensemble_humidity, ensemble_wind_mph,
                     bias_temp_f, bias_humidity, bias_wind_mph,
                     gfs_temp_f, gfs_error, hrrr_temp_f, hrrr_error,
                     ecmwf_temp_f, ecmwf_error, icon_temp_f, icon_error,
                     gem_temp_f, gem_error, jma_temp_f, jma_error)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    location, ts, dt.hour, dt.month, dt.timetuple().tm_yday,
                    obs.get("temp_f"), obs.get("humidity"),
                    obs.get("wind_mph"), obs.get("pressure_mb"),
                    ensemble_preds.get("temp_f"), ensemble_preds.get("humidity"),
                    ensemble_preds.get("wind_mph"),
                    (obs.get("temp_f") or 0) - (ensemble_preds.get("temp_f") or 0) if obs.get("temp_f") and ensemble_preds.get("temp_f") else None,
                    (obs.get("humidity") or 0) - (ensemble_preds.get("humidity") or 0) if obs.get("humidity") and ensemble_preds.get("humidity") else None,
                    (obs.get("wind_mph") or 0) - (ensemble_preds.get("wind_mph") or 0) if obs.get("wind_mph") and ensemble_preds.get("wind_mph") else None,
                    per_model.get("gfs", {}).get("temp_f"),
                    per_model.get("gfs", {}).get("error"),
                    per_model.get("hrrr", {}).get("temp_f"),
                    per_model.get("hrrr", {}).get("error"),
                    per_model.get("ecmwf", {}).get("temp_f"),
                    per_model.get("ecmwf", {}).get("error"),
                    per_model.get("icon", {}).get("temp_f"),
                    per_model.get("icon", {}).get("error"),
                    per_model.get("gem", {}).get("temp_f"),
                    per_model.get("gem", {}).get("error"),
                    per_model.get("jma", {}).get("temp_f"),
                    per_model.get("jma", {}).get("error"),
                ))
                self._maybe_commit()
            except Exception as e:
                log.error("Insert bias error: %s", e)

    def get_recent_bias(self, location: str, hours: int = 6) -> list:
        """Get recent bias log entries for trend analysis."""
        with self.lock:
            since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            return [dict(r) for r in self.conn.execute("""
                SELECT * FROM bias_log
                WHERE location=? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (location, since)).fetchall()]

    def get_bias_features(self, location: str, variable: str = "temp_f",
                          lookback_hours: int = 72) -> Optional[dict]:
        """Compute bias statistics for ML feature augmentation.

        Returns rolling averages of per-model errors over the lookback window,
        so the ML can learn patterns like 'GFS has been running 5°F cold
        for the last 3 days at this location.'
        """
        with self.lock:
            since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
            rows = self.conn.execute("""
                SELECT hour_of_day, month,
                       gfs_error, hrrr_error, ecmwf_error,
                       icon_error, gem_error, jma_error,
                       bias_temp_f, bias_humidity, bias_wind_mph
                FROM bias_log
                WHERE location=? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (location, since)).fetchall()

            if not rows:
                return None

            features = {}
            for model in ("gfs", "hrrr", "ecmwf", "icon", "gem", "jma"):
                errors = [r[f"{model}_error"] for r in rows if r[f"{model}_error"] is not None]
                if errors:
                    features[f"{model}_recent_bias"] = sum(errors) / len(errors)
                    features[f"{model}_recent_abs_error"] = sum(abs(e) for e in errors) / len(errors)
                    # Positive = bias trending more positive over time
                    # Normalized per-entry since entries are ~hourly
                    if len(errors) > 1:
                        features[f"{model}_bias_trend"] = (errors[0] - errors[-1]) / len(errors)
                    else:
                        features[f"{model}_bias_trend"] = 0

            # Overall ensemble bias
            for var in ("temp_f", "humidity", "wind_mph"):
                biases = [r[f"bias_{var}"] for r in rows if r[f"bias_{var}"] is not None]
                if biases:
                    features[f"ensemble_recent_bias_{var}"] = sum(biases) / len(biases)

            features["bias_sample_count"] = len(rows)
            return features

    # ── Query methods ─────────────────────────────────────────────────────

    def get_observations(self, location: str, since: str = None,
                         source: str = None, limit: int = 1000) -> list:
        with self.lock:
            q = "SELECT * FROM observations WHERE location=?"
            p = [location]
            if since:
                q += " AND timestamp >= ?"; p.append(since)
            if source:
                q += " AND source=?"; p.append(source)
            q += " ORDER BY timestamp DESC LIMIT ?"
            p.append(limit)
            return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def get_latest_obs(self, location: str, source: str = "tempest") -> Optional[dict]:
        rows = self.get_observations(location, source=source, limit=1)
        return rows[0] if rows else None

    def get_latest_forecasts(self, location: str) -> list:
        """Most recent forecast set from each model."""
        with self.lock:
            rows = []
            for model in WEATHER_MODELS:
                latest = self.conn.execute("""
                    SELECT * FROM forecasts
                    WHERE location=? AND model=?
                    ORDER BY issued_at DESC, valid_at ASC LIMIT 120
                """, (location, model)).fetchall()
                if latest:
                    issued = latest[0]["issued_at"]
                    rows.extend([dict(r) for r in latest if r["issued_at"] == issued])
            return rows

    def get_ensemble_forecast(self, location: str) -> list:
        with self.lock:
            rows = self.conn.execute("""
                SELECT * FROM ensemble_forecasts
                WHERE location=?
                ORDER BY issued_at DESC, valid_at ASC LIMIT 120
            """, (location,)).fetchall()
            if not rows:
                return []
            issued = rows[0]["issued_at"]
            return [dict(r) for r in rows if r["issued_at"] == issued]

    def get_training_data(self, location: str, variable: str,
                          lead_range: tuple = (0, 72)) -> tuple:
        """Build feature matrix for ML: pair model forecasts with actual obs.

        Uses ALL observation sources (tempest preferred, openmeteo_archive as
        fallback) to maximize training data. Deduplicates to one obs per hour.

        Returns (X, y, feature_names) or (None, None, None).
        """
        with self.lock:
            # Get all observations, preferring tempest over archive
            # GROUP BY rounds timestamp to the hour; MIN(source) picks
            # sources alphabetically so 'openmeteo_archive' < 'tempest' —
            # we use CASE to prefer tempest explicitly
            obs = self.conn.execute("""
                SELECT
                    strftime('%Y-%m-%dT%H:00:00', timestamp) as hour_ts,
                    AVG(temp_f) as temp_f,
                    AVG(humidity) as humidity,
                    AVG(wind_mph) as wind_mph,
                    AVG(pressure_mb) as pressure_mb,
                    AVG(dewpoint_f) as dewpoint_f,
                    AVG(precip_in) as precip_in
                FROM observations
                WHERE location=?
                  AND source IN ('tempest', 'openmeteo_archive', 'ha_outdoor_sensor', 'outdoor_sensor', 'ha_yolink', 'yolink', 'ha_average', 'ha_ha_average', 'metar_kpwm')
                  AND temp_f IS NOT NULL
                GROUP BY strftime('%Y-%m-%dT%H', timestamp)
                ORDER BY hour_ts
            """, (location,)).fetchall()

            if not obs:
                return None, None, None

            X_rows, y_rows = [], []
            feature_names = []
            first = True

            for ob in obs:
                ts = ob["hour_ts"]
                y_val = ob[variable]
                if y_val is None:
                    continue

                try:
                    ob_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    continue

                ts_lo = (ob_dt - timedelta(minutes=30)).isoformat()
                ts_hi = (ob_dt + timedelta(minutes=30)).isoformat()

                features = {}
                model_vals = []

                for mk in WEATHER_MODELS:
                    rows = self.conn.execute("""
                        SELECT temp_f, humidity, wind_mph, pressure_mb,
                               dewpoint_f, precip_prob, precip_in,
                               cloud_cover, wind_gust_mph, lead_hours
                        FROM forecasts
                        WHERE location=? AND model=?
                          AND valid_at >= ? AND valid_at <= ?
                          AND lead_hours >= ? AND lead_hours <= ?
                        ORDER BY lead_hours ASC LIMIT 1
                    """, (location, mk, ts_lo, ts_hi,
                          lead_range[0], lead_range[1])).fetchall()

                    if rows:
                        r = rows[0]
                        val = r[variable] if variable in r.keys() else None
                        features[f"{mk}_{variable}"] = val
                        features[f"{mk}_lead_hours"] = r["lead_hours"]
                        if val is not None:
                            model_vals.append(val)
                    else:
                        features[f"{mk}_{variable}"] = None
                        features[f"{mk}_lead_hours"] = None

                features["hour_of_day"] = ob_dt.hour
                features["day_of_year"] = ob_dt.timetuple().tm_yday
                features["month"] = ob_dt.month

                if len(model_vals) < 2:
                    continue

                if first:
                    feature_names = sorted(features.keys())
                    first = False

                median = float(np.median(model_vals))
                row = [features.get(fn, median) for fn in feature_names]
                row = [median if v is None else v for v in row]
                X_rows.append(row)
                y_rows.append(y_val)

            if not X_rows:
                return None, None, None
            return np.array(X_rows), np.array(y_rows), feature_names

    # ── Cleanup ─────────────────────────────────────────────────────────

    def cleanup_bad_forecasts(self) -> int:
        """Remove forecast data with unrealistic lead_hours (>72).
        These come from backfills that used 30-day chunks with wrong lead times."""
        with self.lock:
            r = self.conn.execute(
                "SELECT COUNT(*) FROM forecasts WHERE lead_hours > 72"
            ).fetchone()
            bad_count = r[0]
            if bad_count > 0:
                self.conn.execute("DELETE FROM forecasts WHERE lead_hours > 72")
                self.conn.commit()
                log.info("Cleaned up %d forecast rows with lead_hours > 72", bad_count)
            return bad_count

    def cleanup_all_forecasts(self) -> int:
        """Remove ALL backfilled forecast data to start clean.
        Keeps observations intact."""
        with self.lock:
            r = self.conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()
            count = r[0]
            self.conn.execute("DELETE FROM forecasts")
            self.conn.execute("DELETE FROM ensemble_forecasts")
            self.conn.commit()
            log.info("Cleared all %d forecasts and ensemble data", count)
            return count

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_db_stats(self) -> dict:
        with self.lock:
            stats = {}
            for tbl in ("observations", "forecasts", "ensemble_forecasts",
                        "model_accuracy", "training_log", "bias_log"):
                r = self.conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
                stats[tbl] = r[0]
            for loc in ("apartment", "occ"):
                r = self.conn.execute(
                    "SELECT COUNT(*) FROM observations WHERE location=?", (loc,)
                ).fetchone()
                stats[f"obs_{loc}"] = r[0]
            r = self.conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM observations"
            ).fetchone()
            stats["obs_start"] = r[0]
            stats["obs_end"] = r[1]
            return stats

    def get_location_stats(self, location: str) -> dict:
        with self.lock:
            s = {}
            r = self.conn.execute(
                "SELECT COUNT(*) FROM observations WHERE location=?", (location,)
            ).fetchone()
            s["observations"] = r[0]
            r = self.conn.execute(
                "SELECT COUNT(*) FROM forecasts WHERE location=?", (location,)
            ).fetchone()
            s["forecasts"] = r[0]
            r = self.conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM observations WHERE location=?",
                (location,)
            ).fetchone()
            s["first_obs"] = r[0]
            s["last_obs"] = r[1]
            return s
