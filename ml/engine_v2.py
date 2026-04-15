"""WeatherOracle ML Engine v2 — Verification-driven ensemble prediction.

Key differences from v1:
  1. Verification loop: Scores past predictions against reality every cycle
  2. Persistence nowcasting: 0-6hr uses Tempest trend extrapolation
  3. Real lead-time training: Only learns from actual forecast lead times
  4. Dynamic model scoring: Per-model, per-lead-time accuracy tracking
  5. Claude advisor integration: Weather regime awareness
"""

import logging
import math
import pickle
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.config import WEATHER_MODELS, MODEL_DIR

log = logging.getLogger("WeatherOracle.ml")

TRAINABLE_VARS = ["temp_f", "humidity", "wind_mph", "dewpoint_f", "pressure_mb"]

LEAD_BUCKETS = [
    ("0-6h", 0, 6),
    ("6-12h", 6, 12),
    ("12-24h", 12, 24),
    ("24-48h", 24, 48),
    ("48-72h", 48, 72),
]


class ModelScoreboard:
    """Tracks per-model, per-lead-bucket accuracy with exponential decay.

    Recent errors are weighted more heavily than old ones. A 48-hour
    half-life means errors from 2 days ago count half as much as new ones.
    This lets the scoreboard respond quickly to weather regime changes.

    Entries are stored as (timestamp, error) tuples. Legacy entries
    (plain floats from old pickles) are auto-migrated on load.
    """

    def __init__(self, half_life_hours: float = 48.0):
        # {(model, bucket, variable): [(timestamp, error), ...]}
        self.errors = defaultdict(list)
        self.max_window = 300  # keep last 300 entries per slot
        self.half_life_seconds = half_life_hours * 3600
        self._decay_ln2 = 0.693147 / self.half_life_seconds

    def record(self, model: str, bucket: str, variable: str, error: float):
        key = (model, bucket, variable)
        self.errors[key].append((time.time(), error))
        if len(self.errors[key]) > self.max_window:
            self.errors[key] = self.errors[key][-self.max_window:]

    def _weighted_stats(self, key: tuple, min_entries: int = 3):
        """Compute decay-weighted mean absolute error and bias.
        Returns (weighted_mae, weighted_bias, count) or (None, None, 0).
        """
        entries = self.errors.get(key, [])
        if len(entries) < min_entries:
            return None, None, 0

        now = time.time()
        w_abs_sum, w_sum, w_bias_sum = 0.0, 0.0, 0.0
        for ts, err in entries:
            age = now - ts
            weight = math.exp(-self._decay_ln2 * age)
            w_abs_sum += abs(err) * weight
            w_bias_sum += err * weight
            w_sum += weight

        if w_sum < 1e-10:
            return None, None, 0
        return w_abs_sum / w_sum, w_bias_sum / w_sum, len(entries)

    def get_mae(self, model: str, bucket: str, variable: str) -> Optional[float]:
        mae, _, _ = self._weighted_stats((model, bucket, variable))
        return round(mae, 3) if mae is not None else None

    def get_bias(self, model: str, bucket: str, variable: str) -> Optional[float]:
        _, bias, _ = self._weighted_stats((model, bucket, variable))
        return round(bias, 3) if bias is not None else None

    def get_weight(self, model: str, bucket: str, variable: str = "temp_f") -> float:
        """Dynamic weight: inverse of recent MAE. Better models get higher weight."""
        mae = self.get_mae(model, bucket, variable)
        if mae is None:
            # No data yet — use default priority from config
            return WEATHER_MODELS.get(model, {}).get("priority", 1.0)
        # Inverse MAE with floor. MAE of 1°F → weight 3.0, MAE of 5°F → weight 0.6
        return max(0.2, min(4.0, 3.0 / max(mae, 0.5)))

    def get_all_weights(self, bucket: str, variable: str = "temp_f") -> dict:
        """Get weights for all models for a given bucket."""
        weights = {}
        for mk in WEATHER_MODELS:
            weights[mk] = self.get_weight(mk, bucket, variable)
        return weights

    def summary(self) -> dict:
        """Return summary for logging/display."""
        result = {}
        for (model, bucket, variable), entries in self.errors.items():
            if variable != "temp_f":
                continue
            mae, bias, n = self._weighted_stats((model, bucket, variable))
            if mae is not None:
                key = f"{model}/{bucket}"
                result[key] = {
                    "mae": round(mae, 2),
                    "bias": round(bias, 2),
                    "n": n,
                }
        return result

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.errors), f)

    def load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Auto-migrate legacy entries (plain floats → timestamped tuples)
            migrated = defaultdict(list)
            now = time.time()
            for key, entries in data.items():
                for entry in entries:
                    if isinstance(entry, (int, float)):
                        # Legacy: assign synthetic timestamps spread over last 48h
                        idx = entries.index(entry)
                        fake_ts = now - (len(entries) - idx) * 900  # ~15min apart
                        migrated[key].append((fake_ts, float(entry)))
                    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                        migrated[key].append((entry[0], entry[1]))
                    else:
                        migrated[key].append((now, float(entry)))
            self.errors = migrated
            log.info("Loaded scoreboard: %d slots", len(self.errors))
        except Exception:
            pass


class PersistenceForecaster:
    """Nowcasting using recent observation trend extrapolation.

    For 0-6 hours, the best predictor is usually: current value + recent trend.
    This beats ALL weather models for very short lead times.
    """

    @staticmethod
    def forecast(recent_obs: list, hours_ahead: int,
                 variable: str = "temp_f") -> Optional[float]:
        """Extrapolate from recent observations.

        recent_obs: list of dicts with 'timestamp' and variable, sorted ascending
        hours_ahead: how many hours to project forward
        """
        vals = []
        times = []
        for obs in recent_obs[-12:]:  # last 12 readings
            v = obs.get(variable)
            ts = obs.get("timestamp", "")
            if v is not None and ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    vals.append(v)
                    times.append(dt.timestamp())
                except:
                    pass

        if len(vals) < 3:
            return None

        # Linear regression on recent trend
        x = np.array(times) - times[0]
        y = np.array(vals)
        # Fit line
        n = len(x)
        sx = np.sum(x)
        sy = np.sum(y)
        sxx = np.sum(x * x)
        sxy = np.sum(x * y)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-10:
            # Flat trend — return current value
            return float(vals[-1])

        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        # Project forward
        future_x = x[-1] + hours_ahead * 3600
        projected = intercept + slope * future_x

        # Dampen extreme extrapolations — the further out, the more we
        # blend back toward the current value.
        # Gentler curve: diurnal temp trends are predictable 3-6h out
        current = float(vals[-1])
        damping = 1.0 / (1.0 + hours_ahead * 0.15)  # 0h: 1.0, 3h: 0.69, 6h: 0.53
        result = projected * damping + current * (1 - damping)

        # Variable-specific sanity clamps
        if variable == "temp_f":
            result = max(current - 15, min(current + 15, result))
        elif variable == "humidity":
            result = max(0, min(100, result))
        elif variable == "wind_mph":
            result = max(0, min(current * 3 + 5, result))  # wind can spike
        else:
            result = max(current - 15, min(current + 15, result))

        return round(result, 1)


class MLEnsembleV2:
    """Gradient-boosted ensemble trained on verified real-lead-time errors."""

    def __init__(self, db):
        self.db = db
        self.models = {}       # {(location, variable, bucket): GBR}
        self.feature_names = {}
        self.scoreboard = ModelScoreboard()
        self._load_saved()

    def _model_path(self, location, variable, bucket):
        return MODEL_DIR / f"v2_{location}_{variable}_{bucket}.pkl"

    def _scoreboard_path(self):
        return MODEL_DIR / "scoreboard.pkl"

    def _load_saved(self):
        for p in MODEL_DIR.glob("v2_*.pkl"):
            try:
                with open(p, "rb") as f:
                    saved = pickle.load(f)
                key = tuple(saved["key"])
                self.models[key] = saved["model"]
                self.feature_names[key] = saved["features"]
                log.info("Loaded ML model: %s", key)
            except Exception as e:
                log.warning("Failed to load %s: %s", p.name, e)
        self.scoreboard.load(self._scoreboard_path())

    def has_trained_models(self, location: str) -> bool:
        return any(k[0] == location for k in self.models)

    def get_model_count(self, location: str) -> int:
        return sum(1 for k in self.models if k[0] == location)

    # ── Verification ──────────────────────────────────────────────────────

    def verify_past_predictions(self, location: str, db) -> dict:
        """Core verification loop: compare what we predicted to what happened.

        For each recent observation, find what each model predicted for that
        hour and score the error. This is the REAL training signal.
        """
        # Get observations from last 6 hours
        since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        obs_list = db.get_observations(location, since=since, source="tempest")

        if not obs_list:
            return {}

        verified = 0
        for obs in obs_list:
            ts = obs["timestamp"]
            try:
                ob_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                continue

            ts_lo = (ob_dt - timedelta(minutes=30)).isoformat()
            ts_hi = (ob_dt + timedelta(minutes=30)).isoformat()

            for mk in WEATHER_MODELS:
                for variable in TRAINABLE_VARS:
                    ob_val = obs.get(variable)
                    if ob_val is None:
                        continue

                    # Find what this model predicted for this hour
                    with db.lock:
                        rows = db.conn.execute("""
                            SELECT temp_f, humidity, wind_mph, dewpoint_f,
                                   pressure_mb, lead_hours
                            FROM forecasts
                            WHERE location=? AND model=?
                              AND valid_at >= ? AND valid_at <= ?
                            ORDER BY lead_hours DESC LIMIT 1
                        """, (location, mk, ts_lo, ts_hi)).fetchall()

                    if not rows:
                        continue

                    pred_val = rows[0][variable]
                    lead = rows[0]["lead_hours"]
                    if pred_val is None or lead is None:
                        continue

                    # Determine bucket
                    bucket = self._get_bucket(lead)
                    error = pred_val - ob_val  # positive = model predicted too high

                    self.scoreboard.record(mk, bucket, variable, error)
                    verified += 1

            # Also verify ensemble predictions
            for variable in TRAINABLE_VARS:
                ob_val = obs.get(variable)
                if ob_val is None:
                    continue

                with db.lock:
                    rows = db.conn.execute("""
                        SELECT temp_f, humidity, wind_mph, dewpoint_f,
                               pressure_mb, lead_hours
                        FROM ensemble_forecasts
                        WHERE location=? AND valid_at >= ? AND valid_at <= ?
                        LIMIT 1
                    """, (location, ts_lo, ts_hi)).fetchall()

                if rows and rows[0][variable] is not None:
                    lead = rows[0]["lead_hours"] or 0
                    bucket = self._get_bucket(lead)
                    error = rows[0][variable] - ob_val
                    self.scoreboard.record("ensemble", bucket, variable, error)

        # Save scoreboard
        self.scoreboard.save(self._scoreboard_path())

        return {"verified": verified, "summary": self.scoreboard.summary()}

    # ── Prediction ────────────────────────────────────────────────────────

    def _get_bucket(self, lead_hours: int) -> str:
        if lead_hours <= 6: return "0-6h"
        if lead_hours <= 12: return "6-12h"
        if lead_hours <= 24: return "12-24h"
        if lead_hours <= 48: return "24-48h"
        return "48-72h"

    def predict(self, location: str, variable: str,
                model_forecasts: dict, lead_hours: int,
                claude_adjustments: dict = None) -> Optional[dict]:
        """Make prediction using dynamic model weighting.

        model_forecasts: {model_key: {field: value, ...}, ...}
        claude_adjustments: optional {model_key: weight_multiplier} from advisor
        """
        bucket = self._get_bucket(lead_hours)
        key = (location, variable, bucket)

        # Collect model values
        model_vals = []
        for mk in WEATHER_MODELS:
            v = model_forecasts.get(mk, {}).get(variable)
            if v is not None:
                model_vals.append((mk, v))

        if len(model_vals) < 2:
            return None

        vals_only = [v for _, v in model_vals]

        # Get dynamic weights from scoreboard (based on recent verification)
        weights = self.scoreboard.get_all_weights(bucket, variable)

        # Apply Claude advisor adjustments if available
        if claude_adjustments:
            for mk, multiplier in claude_adjustments.items():
                if mk in weights:
                    weights[mk] *= multiplier

        # ── ML path (if trained) ──
        if key in self.models and key in self.feature_names:
            fnames = self.feature_names[key]
            features = {}
            for mk in WEATHER_MODELS:
                mf = model_forecasts.get(mk, {})
                features[f"{mk}_{variable}"] = mf.get(variable)
                features[f"{mk}_lead_hours"] = lead_hours
                # Add model's recent bias as feature
                bias = self.scoreboard.get_bias(mk, bucket, variable)
                features[f"{mk}_recent_bias"] = bias if bias is not None else 0

            now = datetime.now()
            features["hour_of_day"] = now.hour
            features["day_of_year"] = now.timetuple().tm_yday
            features["month"] = now.month
            features["lead_hours"] = lead_hours

            # ── Enhanced features (must match training) ──
            features["model_spread"] = max(vals_only) - min(vals_only)
            features["model_std"] = float(np.std(vals_only))
            doy = now.timetuple().tm_yday
            features["season_sin"] = round(math.sin(2 * math.pi * doy / 365.25), 4)
            features["season_cos"] = round(math.cos(2 * math.pi * doy / 365.25), 4)
            features["hour_sin"] = round(math.sin(2 * math.pi * now.hour / 24), 4)
            features["hour_cos"] = round(math.cos(2 * math.pi * now.hour / 24), 4)
            features["is_daytime"] = 1 if 6 <= now.hour <= 20 else 0

            median = float(np.median(vals_only))
            row = [features.get(fn, median) for fn in fnames]
            row = [median if v is None else v for v in row]

            try:
                ml_pred = float(self.models[key].predict([row])[0])

                # Apply scoreboard-informed bias correction
                # DECAY with lead time: today's bias may not apply to day 2-3
                ens_bias = self.scoreboard.get_bias("ensemble", bucket, variable)
                if ens_bias is not None and abs(ens_bias) > 0.5:
                    bias_decay = max(0, 1.0 - lead_hours / 24.0)  # Full at 0h, zero at 24h+
                    ml_pred -= ens_bias * 0.5 * bias_decay

                # ── ML confidence gate ──
                # Compute the weighted average first to compare
                _tw, _ws = 0, 0
                for mk, val in model_vals:
                    w = weights.get(mk, 1.0)
                    _ws += val * w
                    _tw += w
                weighted_avg = _ws / _tw if _tw else median

                divergence = abs(ml_pred - weighted_avg)

                # ML must earn the right to override raw models:
                # - If ML diverges > 5°F from consensus AND we have < 200
                #   verified samples for this bucket, defer to weighted avg.
                # - This prevents a barely-trained ML from overriding correct
                #   model predictions for weather patterns it hasn't seen yet.
                ens_mae = self.scoreboard.get_mae("ensemble", bucket, variable)
                n_samples = len(self.scoreboard.errors.get(
                    ("ensemble", bucket, variable), []))

                ml_trustworthy = (
                    n_samples >= 200  # Enough verification data
                    or divergence <= 5.0  # ML agrees with models (small correction)
                    or (ens_mae is not None and ens_mae < 3.0)  # ML has proven accurate
                )

                if ml_trustworthy:
                    pred = ml_pred
                    method = "ml_v2"
                else:
                    # Blend: mostly weighted avg, small ML influence
                    pred = weighted_avg * 0.8 + ml_pred * 0.2
                    method = "ml_v2+gated"
                    log.debug("ML gated for %s/%s: ML=%.1f, models=%.1f, "
                              "diverge=%.1f, n=%d",
                              variable, bucket, ml_pred, weighted_avg,
                              divergence, n_samples)

                q25, q75 = float(np.percentile(vals_only, 25)), float(np.percentile(vals_only, 75))
                iqr = q75 - q25
                lead_penalty = min(20, lead_hours * 0.3)
                spread_penalty = iqr * 5
                model_bonus = min(5, (len(model_vals) - 4) * 2)
                confidence = max(10, min(98, 95 - spread_penalty - lead_penalty + model_bonus))

                return {
                    "value": round(pred, 1),
                    "confidence": round(confidence, 1),
                    "method": method,
                    "bucket": bucket,
                    "n_models": len(model_vals),
                }
            except Exception as e:
                log.error("ML predict error: %s", e)

        # ── Weighted average with dynamic scores ──
        total_w, weighted_sum = 0, 0
        for mk, val in model_vals:
            w = weights.get(mk, 1.0)
            weighted_sum += val * w
            total_w += w

        pred = weighted_sum / total_w if total_w else float(np.mean(vals_only))

        # Scoreboard-informed correction — decay with lead time
        # Today's bias shouldn't suppress a pattern change on day 2-3
        ens_bias = self.scoreboard.get_bias("ensemble", bucket, variable)
        if ens_bias is not None and abs(ens_bias) > 0.5:
            bias_decay = max(0, 1.0 - lead_hours / 24.0)  # Zero beyond 24h
            pred -= ens_bias * 0.4 * bias_decay

        q25, q75 = float(np.percentile(vals_only, 25)), float(np.percentile(vals_only, 75))
        iqr = q75 - q25
        lead_penalty = min(20, lead_hours * 0.3)
        spread_penalty = iqr * 5
        model_bonus = min(5, (len(model_vals) - 4) * 2)
        confidence = max(10, min(98, 95 - spread_penalty - lead_penalty + model_bonus))

        return {
            "value": round(pred, 1),
            "confidence": round(confidence, 1),
            "method": "weighted_v2",
            "bucket": bucket,
            "n_models": len(model_vals),
        }

    def predict_hour(self, location: str, model_forecasts: dict,
                     lead_hours: int,
                     claude_adjustments: dict = None) -> dict:
        """Predict all variables for one forecast hour."""
        result = {}
        for var in TRAINABLE_VARS:
            p = self.predict(location, var, model_forecasts, lead_hours,
                             claude_adjustments)
            if p:
                result[var] = p

        # Precip prob: weighted average using scoreboard
        bucket = self._get_bucket(lead_hours)
        weights = self.scoreboard.get_all_weights(bucket)
        total_w, weighted_sum = 0, 0
        for mk in WEATHER_MODELS:
            pp = model_forecasts.get(mk, {}).get("precip_prob")
            if pp is not None:
                w = weights.get(mk, 1.0)
                weighted_sum += pp * w
                total_w += w
        if total_w > 0:
            result["precip_prob"] = {"value": round(weighted_sum / total_w, 0),
                                     "method": "weighted_v2"}
        return result

    # ── Training ──────────────────────────────────────────────────────────

    def train_all(self, location: str, min_samples: int = 48) -> dict:
        """Train ML models using ONLY verified real-lead-time data.

        Adapts model complexity to dataset size:
          <48 samples: skip (insufficient)
          48-100: Ridge regression (no overfitting risk)
          100-500: Light GBR (50 trees, depth 3)
          500+: Full GBR (150 trees, depth 4)
        """
        from sklearn.linear_model import Ridge

        results = {}
        for variable in TRAINABLE_VARS:
            for bucket_name, lead_min, lead_max in LEAD_BUCKETS:
                X, y, fnames = self._build_training_set(
                    location, variable, lead_min, lead_max)

                n = 0 if X is None else len(X)
                if n < min_samples:
                    results[f"{variable}/{bucket_name}"] = {
                        "status": "insufficient_data",
                        "n_samples": n,
                        "need": min_samples,
                    }
                    continue

                # ── Select model complexity based on data size ──
                if n < 100:
                    model = Ridge(alpha=1.0)
                    model_type = "ridge"
                elif n < 500:
                    model = GradientBoostingRegressor(
                        n_estimators=50, max_depth=3, learning_rate=0.1,
                        subsample=0.8, min_samples_leaf=max(5, n // 20),
                        random_state=42)
                    model_type = "gbr_light"
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=150, max_depth=4, learning_rate=0.08,
                        subsample=0.8, min_samples_leaf=max(3, n // 50),
                        random_state=42)
                    model_type = "gbr_full"

                # ── Cross-validate ──
                n_splits = min(5, max(2, n // 10))
                try:
                    cv = cross_val_score(model, X, y, cv=n_splits,
                                         scoring="neg_mean_absolute_error")
                    cv_mae = round(-cv.mean(), 3)
                except Exception:
                    cv_mae = None

                # ── Fit ──
                model.fit(X, y)
                preds = model.predict(X)
                mae = mean_absolute_error(y, preds)

                # Best individual model comparison
                best_mae, best_name = float("inf"), "none"
                for fn in fnames:
                    if fn.endswith(f"_{variable}") and not fn.endswith("_recent_bias"):
                        col = fnames.index(fn)
                        m = mean_absolute_error(y, X[:, col])
                        if m < best_mae:
                            best_mae, best_name = m, fn.replace(f"_{variable}", "")

                improvement = ((best_mae - mae) / best_mae * 100
                               if best_mae > 0 else 0)

                # ── Overfitting guard: reject if CV is much worse than train ──
                if cv_mae is not None and mae > 0:
                    overfit_ratio = cv_mae / max(mae, 0.01)
                    if overfit_ratio > 10.0:
                        log.warning("OVERFIT REJECTED %s/%s/%s: train_mae=%.3f, "
                                    "cv_mae=%.3f (ratio=%.1fx, n=%d, type=%s)",
                                    location, variable, bucket_name,
                                    mae, cv_mae, overfit_ratio, n, model_type)
                        results[f"{variable}/{bucket_name}"] = {
                            "status": "overfit_rejected",
                            "n_samples": n, "mae": round(mae, 3),
                            "cv_mae": cv_mae, "overfit_ratio": round(overfit_ratio, 1),
                            "model_type": model_type,
                        }
                        continue

                key = (location, variable, bucket_name)
                self.models[key] = model
                self.feature_names[key] = fnames
                with open(self._model_path(location, variable, bucket_name), "wb") as f:
                    pickle.dump({"key": key, "model": model, "features": fnames}, f)

                results[f"{variable}/{bucket_name}"] = {
                    "status": "trained", "n_samples": n,
                    "mae": round(mae, 3), "cv_mae": cv_mae,
                    "best_individual": best_name,
                    "best_individual_mae": round(best_mae, 3),
                    "improvement_pct": round(improvement, 1),
                    "model_type": model_type,
                }

                # Log
                try:
                    self.db.conn.execute("""
                        INSERT INTO training_log
                        (location, variable, bucket, timestamp, n_samples,
                         cv_mae, ensemble_mae, best_individual,
                         best_individual_mae, improvement_pct)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (location, variable, bucket_name,
                          datetime.now(timezone.utc).isoformat(),
                          n, cv_mae, round(mae, 3),
                          best_name, round(best_mae, 3),
                          round(improvement, 1)))
                    self.db.conn.commit()
                except Exception:
                    pass

                log.info("Trained %s/%s/%s [%s]: MAE=%.3f, CV=%.3f, "
                         "vs best %.3f (%+.1f%%), n=%d",
                         location, variable, bucket_name, model_type,
                         mae, cv_mae or -1, best_mae, improvement, n)

        return results

    def _build_training_set(self, location, variable, lead_min, lead_max):
        """Build training features from REAL lead-time forecasts + observations.

        Only uses forecasts where lead_hours > 2 to avoid Day-0 pseudo-forecasts.
        """
        with self.db.lock:
            obs = self.db.conn.execute("""
                SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) as hour_ts,
                       AVG(temp_f) as temp_f, AVG(humidity) as humidity,
                       AVG(wind_mph) as wind_mph, AVG(pressure_mb) as pressure_mb,
                       AVG(dewpoint_f) as dewpoint_f
                FROM observations
                WHERE location=? AND source IN ('tempest', 'openmeteo_archive', 'ha_outdoor_sensor', 'outdoor_sensor', 'ha_yolink', 'yolink', 'ha_average', 'ha_ha_average', 'metar_kpwm')
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
                    # Require lead_hours >= 3 (exclude Day-0) and <= 72 (hard cap)
                    actual_max = min(lead_max, 72)
                    rows = self.db.conn.execute("""
                        SELECT temp_f, humidity, wind_mph, pressure_mb,
                               dewpoint_f, lead_hours
                        FROM forecasts
                        WHERE location=? AND model=?
                          AND valid_at >= ? AND valid_at <= ?
                          AND lead_hours >= 3 AND lead_hours <= ?
                        ORDER BY lead_hours DESC LIMIT 1
                    """, (location, mk, ts_lo, ts_hi,
                          actual_max)).fetchall()

                    if rows:
                        r = rows[0]
                        val = r[variable] if variable in r.keys() else None
                        features[f"{mk}_{variable}"] = val
                        features[f"{mk}_lead_hours"] = r["lead_hours"]
                        # Include model's known bias as feature
                        bias = self.scoreboard.get_bias(mk, self._get_bucket(r["lead_hours"]), variable)
                        features[f"{mk}_recent_bias"] = bias if bias is not None else 0
                        if val is not None:
                            model_vals.append(val)
                    else:
                        features[f"{mk}_{variable}"] = None
                        features[f"{mk}_lead_hours"] = None
                        features[f"{mk}_recent_bias"] = 0

                features["hour_of_day"] = ob_dt.hour
                features["day_of_year"] = ob_dt.timetuple().tm_yday
                features["month"] = ob_dt.month
                features["lead_hours"] = (lead_min + lead_max) // 2

                if len(model_vals) < 2:
                    continue

                # ── Enhanced features for accuracy ──
                # (must be AFTER model_vals check since these depend on it)
                # Model disagreement (high spread = uncertain, good signal)
                features["model_spread"] = max(model_vals) - min(model_vals)
                features["model_std"] = float(np.std(model_vals))

                # Cyclical season encoding (avoids Jan=1/Dec=12 boundary issue)
                doy = ob_dt.timetuple().tm_yday
                features["season_sin"] = round(math.sin(2 * math.pi * doy / 365.25), 4)
                features["season_cos"] = round(math.cos(2 * math.pi * doy / 365.25), 4)

                # Cyclical hour encoding
                features["hour_sin"] = round(math.sin(2 * math.pi * ob_dt.hour / 24), 4)
                features["hour_cos"] = round(math.cos(2 * math.pi * ob_dt.hour / 24), 4)

                # Daytime flag (affects solar heating bias)
                features["is_daytime"] = 1 if 6 <= ob_dt.hour <= 20 else 0

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

    # ── Accuracy for display ──────────────────────────────────────────────

    def compute_accuracy(self, location: str, days: int = 7) -> dict:
        """Compute accuracy from scoreboard data."""
        results = {}
        for variable in TRAINABLE_VARS:
            results[variable] = {}
            for mk in list(WEATHER_MODELS.keys()) + ["ensemble"]:
                combined_errors = []
                for bucket_name, _, _ in LEAD_BUCKETS:
                    key = (mk, bucket_name, variable)
                    if key in self.scoreboard.errors:
                        combined_errors.extend(self.scoreboard.errors[key])

                if combined_errors:
                    e = np.array(combined_errors)
                    display_name = WEATHER_MODELS.get(mk, {}).get("name", mk)
                    if mk == "ensemble":
                        display_name = "★ ML Ensemble"
                    results[variable][mk] = {
                        "mae": round(float(np.mean(np.abs(e))), 2),
                        "rmse": round(float(np.sqrt(np.mean(e**2))), 2),
                        "bias": round(float(np.mean(e)), 2),
                        "n": len(combined_errors),
                    }
        return results

    def get_training_summary(self, location: str) -> list:
        with self.db.lock:
            return [dict(r) for r in self.db.conn.execute("""
                SELECT * FROM training_log
                WHERE location=? ORDER BY timestamp DESC LIMIT 50
            """, (location,)).fetchall()]
