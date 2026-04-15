"""WeatherOracle orchestrator v2 — verification-driven prediction
with persistence nowcasting and Claude advisor integration."""

import logging
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.config import LOCATIONS, WEATHER_MODELS, WMO_CODES, load_config
from core.database import WeatherDB
from collectors import (TempestCollector, OpenMeteoCollector, NWSCollector,
                       HACollector, METARCollector)
from collectors.ha_publisher import HAPublisher
from ml.engine_v2 import MLEnsembleV2, PersistenceForecaster
from ml.claude_advisor import ClaudeAdvisor

log = logging.getLogger("WeatherOracle.engine")


class WeatherOracle:
    """Main engine v2: verification-driven prediction pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.db = WeatherDB()

        # Clean up any bad forecast data from previous backfills
        bad = self.db.cleanup_bad_forecasts()
        if bad > 0:
            log.info("Startup cleanup: removed %d forecasts with bad lead_hours", bad)

        self.ml = MLEnsembleV2(self.db)
        self.persistence = PersistenceForecaster()

        # Claude advisor (optional — works without API key)
        self.advisor = ClaudeAdvisor(config.get("claude_api_key"))

        # METAR for weather condition verification
        self.metar = METARCollector()
        self.last_metar = None

        # Collectors
        self.tempest = TempestCollector(config.get("tempest_api_token", ""))
        self.openmeteo = OpenMeteoCollector()
        self.nws = NWSCollector()
        self.ha = None
        self.ha_publisher = None
        if config.get("ha_url") and config.get("ha_token"):
            self.ha = HACollector(config["ha_url"], config["ha_token"])
            self.ha_publisher = HAPublisher(config["ha_url"], config["ha_token"])

        self._running = False
        self._thread = None
        self.callbacks = []

        # Per-location state
        self.current_obs = {}
        self.recent_obs = {}       # {location: [last ~12 obs sorted ascending]}
        self.crosscheck = {}
        self.alerts = {}
        self.nws_discussion = None

    def notify(self, msg: str):
        log.info(msg)
        for cb in self.callbacks:
            try:
                cb(msg)
            except:
                pass

    # ── Observation Collection ────────────────────────────────────────────

    def collect_observations(self):
        """Pull current observations from Tempest and HA for both locations."""
        for loc_key, loc in LOCATIONS.items():
            obs = self.tempest.get_current(loc["tempest_station"])
            if obs:
                ts = obs.pop("timestamp")
                self.db.insert_observation(loc_key, ts, "tempest", **obs)
                obs["timestamp"] = ts
                self.current_obs[loc_key] = obs

                # Maintain recent obs list for persistence forecasting
                if loc_key not in self.recent_obs:
                    self.recent_obs[loc_key] = []
                self.recent_obs[loc_key].append(obs)
                # Keep last 24 readings (~2 hours at 5-min intervals)
                self.recent_obs[loc_key] = self.recent_obs[loc_key][-24:]

                self.notify(f"[{loc['short']}] Tempest: {obs.get('temp_f')}°F, "
                           f"{obs.get('humidity')}% RH, "
                           f"wind {obs.get('wind_mph')} mph")
            else:
                self.notify(f"[{loc['short']}] Tempest fetch FAILED")

        # HA cross-check sensors (apartment only)
        # Source names must match what training queries expect:
        #   ha_yolink, ha_outdoor_sensor, ha_average
        _HA_SOURCE_MAP = {
            "yolink": "ha_yolink",
            "outdoor_sensor": "ha_outdoor_sensor",
            "ha_average": "ha_average",
        }
        if self.ha:
            xcheck = self.ha.get_outdoor_crosscheck(self.config)
            if xcheck:
                self.crosscheck["apartment"] = xcheck
                for src, data in xcheck.items():
                    db_source = _HA_SOURCE_MAP.get(src, f"ha_{src}")
                    self.db.insert_observation(
                        "apartment",
                        datetime.now(timezone.utc).isoformat(),
                        db_source,
                        temp_f=data.get("temp_f"),
                        humidity=data.get("humidity"))
                    self.notify(f"[APT] {db_source}: {data.get('temp_f')}°F, "
                               f"{data.get('humidity')}% RH")

    # ── Forecast Collection ───────────────────────────────────────────────

    def collect_forecasts(self):
        """Pull forecasts from all models for both locations."""
        hours = self.config.get("forecast_hours", 72)
        for loc_key, loc in LOCATIONS.items():
            collected = 0

            # Open-Meteo models
            for mk in WEATHER_MODELS:
                if mk == "tempest_bf":
                    continue
                fcs = self.openmeteo.get_forecast(mk, loc["lat"], loc["lon"], hours)
                if fcs:
                    for fc in fcs:
                        self.db.insert_forecast(
                            loc_key, mk, fc["issued_at"], fc["valid_at"],
                            fc["lead_hours"],
                            temp_f=fc.get("temp_f"), humidity=fc.get("humidity"),
                            dewpoint_f=fc.get("dewpoint_f"),
                            wind_mph=fc.get("wind_mph"),
                            wind_gust_mph=fc.get("wind_gust_mph"),
                            wind_dir=fc.get("wind_dir"),
                            pressure_mb=fc.get("pressure_mb"),
                            precip_prob=fc.get("precip_prob"),
                            precip_in=fc.get("precip_in"),
                            cloud_cover=fc.get("cloud_cover"),
                            weather_code=fc.get("weather_code"))
                    collected += 1
                    self.notify(f"[{loc['short']}] {WEATHER_MODELS[mk]['name']}: "
                               f"{len(fcs)}h forecast")
                else:
                    self.notify(f"[{loc['short']}] {mk}: FAILED")
                time.sleep(0.5)

            # Tempest BetterForecast
            bf = self.tempest.get_better_forecast_hourly(loc["tempest_station"])
            if bf:
                for fc in bf:
                    self.db.insert_forecast(
                        loc_key, "tempest_bf", fc["issued_at"], fc["valid_at"],
                        fc["lead_hours"],
                        temp_f=fc.get("temp_f"), humidity=fc.get("humidity"),
                        dewpoint_f=fc.get("dewpoint_f"),
                        wind_mph=fc.get("wind_mph"),
                        wind_gust_mph=fc.get("wind_gust_mph"),
                        wind_dir=fc.get("wind_dir"),
                        pressure_mb=fc.get("pressure_mb"),
                        precip_prob=fc.get("precip_prob"),
                        precip_in=fc.get("precip_in"),
                        cloud_cover=fc.get("cloud_cover"),
                        weather_code=fc.get("weather_code"))
                collected += 1
                self.notify(f"[{loc['short']}] Tempest BetterForecast: {len(bf)}h")
            else:
                self.notify(f"[{loc['short']}] tempest_bf: FAILED")

            self.notify(f"[{loc['short']}] {collected}/{len(WEATHER_MODELS)} models collected")

    # ── Verification (the key learning signal) ────────────────────────────

    def verify_predictions(self):
        """Score past predictions against what actually happened.
        This is what drives the model accuracy learning."""
        for loc_key, loc in LOCATIONS.items():
            result = self.ml.verify_past_predictions(loc_key, self.db)
            if result.get("verified", 0) > 0:
                self.notify(f"[{loc['short']}] Verified {result['verified']} predictions")

                # Log best/worst model for this verification batch
                summary = result.get("summary", {})
                temp_scores = {k: v for k, v in summary.items()
                              if k.endswith("/0-6h") and "/ensemble" not in k}
                if temp_scores:
                    best = min(temp_scores.items(), key=lambda x: x[1]["mae"])
                    worst = max(temp_scores.items(), key=lambda x: x[1]["mae"])
                    self.notify(f"[{loc['short']}] Best short-term: {best[0]} "
                               f"(MAE={best[1]['mae']}°F), "
                               f"Worst: {worst[0]} (MAE={worst[1]['mae']}°F)")

                # Score Claude advisor feedback if active
                if self.advisor.is_configured() and self.advisor.last_run:
                    ens_score = summary.get("ensemble/0-6h")
                    if ens_score and temp_scores:
                        best_model_mae = min(v["mae"] for v in temp_scores.values())
                        ens_mae = ens_score.get("mae", 99)
                        # If ensemble beats average of individual models, advice helped
                        avg_model_mae = sum(v["mae"] for v in temp_scores.values()) / len(temp_scores)
                        delta = ens_mae - avg_model_mae  # negative = ensemble is better
                        helped = delta < 0
                        self.advisor.record_feedback(
                            helped, delta,
                            f"ens={ens_mae:.1f} vs avg_model={avg_model_mae:.1f}")
                        self.notify(f"[{loc['short']}] Advisor feedback: "
                                   f"{'✓ helped' if helped else '✗ hurt'} "
                                   f"(ΔMAE={delta:+.1f}°F)")

    # ── METAR Collection & Condition Verification ───────────────────────

    def collect_metar(self):
        """Pull METAR for each location's configured station."""
        seen_stations = set()
        for loc_key, loc in LOCATIONS.items():
            station = loc.get("metar_station", "")
            if not station:
                continue
            # Avoid fetching the same station twice if shared between locations
            parsed = None
            if station not in seen_stations:
                parsed = self.metar.get_current(station)
                seen_stations.add(station)
                if parsed:
                    self.last_metar = parsed
                    cond = parsed.get("primary_condition", "?")
                    temp = parsed.get("temp_f", "?")
                    vis = parsed.get("visibility_miles", "?")
                    self.notify(f"[METAR] {station}: {temp}°F, {cond}, vis={vis}mi")
            else:
                parsed = self.last_metar  # reuse if same station

            if parsed:
                ts = parsed.get("timestamp") or datetime.now(timezone.utc).isoformat()
                self.db.insert_observation(
                    loc_key, ts, f"metar_{station.lower()}",
                    temp_f=parsed.get("temp_f"),
                    humidity=parsed.get("humidity"),
                    wind_mph=parsed.get("wind_mph"),
                    wind_gust_mph=parsed.get("wind_gust_mph"),
                    wind_dir=parsed.get("wind_dir"),
                    pressure_mb=parsed.get("pressure_mb"),
                    dewpoint_f=parsed.get("dewpoint_f"),
                )

    def verify_conditions(self):
        """Verify model weather conditions against METAR observations.

        Checks if models predicted the right weather type (rain, snow, etc.)
        and records condition accuracy in the scoreboard.
        """
        if not self.last_metar:
            return

        metar = self.last_metar
        metar_wmo = metar.get("wmo_code", 2)
        metar_precip = metar.get("has_precip", False)
        metar_type = metar.get("precip_type")
        metar_fog = metar.get("is_fog", False)

        # Also check Tempest precipitation
        for loc_key in ("apartment", "occ"):
            obs = self.current_obs.get(loc_key, {})
            tempest_precip_in = obs.get("precip_in", 0) or 0

            # Compare each model's weather_code to METAR
            latest = self.db.get_latest_forecasts(loc_key)
            now = datetime.now()

            # Find forecasts closest to current hour
            closest = {}
            for fc in latest:
                try:
                    vdt = datetime.fromisoformat(fc["valid_at"])
                    delta = abs((vdt - now).total_seconds())
                    model = fc["model"]
                    if model not in closest or delta < closest[model][0]:
                        closest[model] = (delta, fc)
                except:
                    pass

            for model, (_, fc) in closest.items():
                model_wmo = fc.get("weather_code")
                model_precip_prob = fc.get("precip_prob", 0) or 0

                if model_wmo is None:
                    continue

                # Score: did model predict precipitation correctly?
                model_predicts_precip = (model_wmo >= 51 or model_precip_prob > 50)
                actual_precip = metar_precip or tempest_precip_in > 0.01

                if model_predicts_precip == actual_precip:
                    # Correct precipitation call
                    self.ml.scoreboard.record(model, "conditions", "precip_correct", 0)
                else:
                    # Wrong precipitation call (1 = error)
                    self.ml.scoreboard.record(model, "conditions", "precip_correct", 1)
                    log.debug("%s/%s: precip mismatch (predicted=%s, actual=%s)",
                              loc_key, model, model_predicts_precip, actual_precip)

                # Score: how close is the WMO code?
                # Group codes: 0-3 = clear/cloudy, 45-48 = fog,
                # 51-57 = drizzle, 61-67 = rain, 71-77 = snow, 80-82 = showers, 95+ = thunder
                model_group = self._wmo_group(model_wmo)
                metar_group = self._wmo_group(metar_wmo)
                if model_group == metar_group:
                    self.ml.scoreboard.record(model, "conditions", "wx_type_correct", 0)
                else:
                    self.ml.scoreboard.record(model, "conditions", "wx_type_correct", 1)

            # Per-location summary
            summary_parts = []
            if metar_precip:
                summary_parts.append(f"precip={metar_type or 'yes'}")
            if metar_fog:
                summary_parts.append("fog")
            if tempest_precip_in > 0:
                summary_parts.append(f"Tempest precip={tempest_precip_in:.3f}in")

            if summary_parts:
                self.notify(f"[VERIFY] {loc_key}: {', '.join(summary_parts)}")

    @staticmethod
    def _wmo_group(code: int) -> str:
        if code <= 3: return "clear_cloudy"
        if code <= 48: return "fog"
        if code <= 57: return "drizzle"
        if code <= 67: return "rain"
        if code <= 77: return "snow"
        if code <= 82: return "showers"
        return "thunderstorm"

    # ── Ensemble Generation (tiered prediction) ──────────────────────────

    def generate_ensemble(self):
        """Generate predictions using tiered approach:
        - 0-6h: Persistence (trend extrapolation from Tempest) blended with models
        - 6-12h: Mostly models with persistence influence
        - 12-72h: Pure model ensemble with verification-based weights
        """
        issued_at = datetime.now(timezone.utc).isoformat()
        now = datetime.now()

        # Get Claude advisor adjustments if available
        claude_adj = self.advisor.get_adjustments()

        for loc_key, loc in LOCATIONS.items():
            latest = self.db.get_latest_forecasts(loc_key)
            if not latest:
                self.notify(f"[{loc['short']}] No forecast data for ensemble")
                continue

            # Group by valid_at
            by_valid = defaultdict(dict)
            for fc in latest:
                by_valid[fc["valid_at"]][fc["model"]] = fc

            sorted_hours = sorted(by_valid.items())
            obs = self.current_obs.get(loc_key, {})
            recent = self.recent_obs.get(loc_key, [])

            count = 0
            for valid_at, model_fcs in sorted_hours:
                lead = list(model_fcs.values())[0].get("lead_hours", 0)

                # Get model ensemble prediction
                model_pred = self.ml.predict_hour(
                    loc_key, model_fcs, lead, claude_adj)

                if "temp_f" not in model_pred:
                    continue

                kw = {}

                # Variables that benefit from persistence/obs anchoring
                _PERSIST_VARS = ("temp_f", "humidity", "wind_mph")
                # Variables that are model-only (no persistence needed)
                _MODEL_ONLY_VARS = ("dewpoint_f", "pressure_mb")

                # ── TIER 1: Persistence + Anchor (0-6h) ──────────────
                if lead <= 6 and obs and recent:
                    for variable in _PERSIST_VARS:
                        obs_val = obs.get(variable)
                        model_val = model_pred.get(variable, {}).get("value")

                        # Persistence forecast from trend
                        persist_val = self.persistence.forecast(
                            recent, lead, variable)

                        if obs_val is not None and model_val is not None:
                            # Blend: persistence + current obs weighted heavily,
                            # models less so for near-term
                            obs_w = max(0.1, 0.85 - lead * 0.12)  # 0.85 → 0.13
                            model_w = 1.0 - obs_w

                            # Use persistence if available, otherwise current obs
                            anchor = persist_val if persist_val is not None else obs_val
                            blended = round(anchor * obs_w + model_val * model_w, 1)

                            if variable == "humidity":
                                blended = max(0, min(100, blended))
                            elif variable == "wind_mph":
                                blended = max(0, blended)

                            kw[variable] = blended
                        elif model_val is not None:
                            kw[variable] = model_val

                    # Model-only vars: use ML prediction directly
                    for variable in _MODEL_ONLY_VARS:
                        if variable in model_pred:
                            kw[variable] = model_pred[variable]["value"]

                    # High confidence when anchored to reality
                    base_conf = model_pred["temp_f"]["confidence"]
                    obs_w = max(0.1, 0.85 - lead * 0.12)
                    kw["confidence"] = round(min(98, base_conf + obs_w * 15), 1)
                    method = f"persist+model({1-obs_w:.0%}model)"

                # ── TIER 2: Model-heavy blend (6-12h) ─────────────────
                elif lead <= 12 and obs:
                    for variable in _PERSIST_VARS:
                        obs_val = obs.get(variable)
                        model_val = model_pred.get(variable, {}).get("value")

                        if obs_val is not None and model_val is not None:
                            # Models dominate but obs provides sanity check
                            obs_w = max(0.02, 0.15 - (lead - 6) * 0.02)
                            blended = round(obs_val * obs_w + model_val * (1 - obs_w), 1)
                            if variable == "humidity":
                                blended = max(0, min(100, blended))
                            elif variable == "wind_mph":
                                blended = max(0, blended)
                            kw[variable] = blended
                        elif model_val is not None:
                            kw[variable] = model_val

                    for variable in _MODEL_ONLY_VARS:
                        if variable in model_pred:
                            kw[variable] = model_pred[variable]["value"]

                    kw["confidence"] = round(model_pred["temp_f"]["confidence"], 1)
                    method = model_pred["temp_f"].get("method", "weighted_v2")

                # ── TIER 3: Pure model ensemble (12-72h) ──────────────
                else:
                    for variable in _PERSIST_VARS + _MODEL_ONLY_VARS:
                        if variable in model_pred:
                            kw[variable] = model_pred[variable]["value"]
                    kw["confidence"] = round(model_pred["temp_f"]["confidence"], 1)
                    method = model_pred["temp_f"].get("method", "weighted_v2")

                # Pass through non-predicted fields
                for src in ("ecmwf", "gfs", "hrrr", "icon", "tempest_bf"):
                    if src in model_fcs:
                        for field in ("cloud_cover", "weather_code", "pressure_mb",
                                      "dewpoint_f", "wind_gust_mph"):
                            if field not in kw and model_fcs[src].get(field) is not None:
                                kw[field] = model_fcs[src][field]
                        break

                if "precip_prob" in model_pred:
                    kw["precip_prob"] = model_pred["precip_prob"]["value"]

                self.db.insert_ensemble(loc_key, issued_at, valid_at, lead,
                                        method, **kw)
                count += 1

            # Log what method was used for the first hour
            if sorted_hours:
                first_lead = list(sorted_hours[0][1].values())[0].get("lead_hours", 0)
                first_temp = obs.get("temp_f", "?") if obs else "?"
                self.notify(
                    f"[{loc['short']}] Ensemble: {count}h "
                    f"(obs={first_temp}°F, lead0_method=persist+model, "
                    f"ML={self.ml.has_trained_models(loc_key)}"
                    f"{', Claude' if claude_adj else ''})")

    # ── Claude Advisor ────────────────────────────────────────────────────

    def run_advisor(self):
        """Run Claude advisor to get model weight adjustments."""
        if not self.advisor.is_configured():
            return

        # Build raw forecasts for advisor
        raw_fcs = {}
        for loc_key in LOCATIONS:
            raw_fcs[loc_key] = {}
            latest = self.db.get_latest_forecasts(loc_key)
            by_model = defaultdict(list)
            for fc in latest:
                by_model[fc["model"]].append(fc)
            raw_fcs[loc_key] = dict(by_model)

        # Run advisor for each location
        for loc_key, loc in LOCATIONS.items():
            advice = self.advisor.analyze(
                nws_discussion=self.nws_discussion,
                observations=self.current_obs,
                model_forecasts=raw_fcs.get(loc_key, {}),
                scoreboard_summary=self.ml.scoreboard.summary(),
                location_name=loc["name"],
            )
            if advice:
                self.notify(f"[{loc['short']}] Claude: {advice.get('regime', '?')}")
            else:
                self.notify(f"[{loc['short']}] Claude advisor: skipped (no key or error)")

    # ── NWS ───────────────────────────────────────────────────────────────

    def collect_alerts(self):
        for loc_key, loc in LOCATIONS.items():
            alerts = self.nws.get_alerts(loc["lat"], loc["lon"])
            self.alerts[loc_key] = alerts
            if alerts:
                self.notify(f"[{loc['short']}] ⚠ {len(alerts)} NWS alert(s): "
                           f"{alerts[0]['event']}")
            else:
                self.notify(f"[{loc['short']}] No active NWS alerts")

    def collect_discussion(self):
        # Use first location's NWS office, if configured
        office = "GYX"
        for loc in LOCATIONS.values():
            if loc.get("nws_office"):
                office = loc["nws_office"]
                break
        self.nws_discussion = self.nws.get_forecast_discussion(office)
        if self.nws_discussion:
            self.notify("[NWS] Forecast discussion updated")

    # ── Backfill ──────────────────────────────────────────────────────────

    def backfill_observations(self, days: int = 14):
        for loc_key, loc in LOCATIONS.items():
            self.notify(f"[{loc['short']}] Backfilling {days} days of Tempest obs...")
            history = self.tempest.get_history(loc["tempest_station"], days)
            count = 0
            for obs in history:
                ts = obs.pop("timestamp")
                self.db.insert_observation(loc_key, ts, "tempest", **obs)
                count += 1
            self.notify(f"[{loc['short']}] Backfilled {count} observations")

    def backfill_forecasts(self, days: int = 14):
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        model_params = {
            "gfs": "gfs_global", "hrrr": "ncep_hrrr_conus",
            "ecmwf": "ecmwf_ifs025", "icon": "icon_seamless",
            "gem": "gem_seamless", "jma": "jma_seamless",
        }
        all_models = ",".join(model_params.values())

        for loc_key, loc in LOCATIONS.items():
            self.notify(f"[{loc['short']}] Backfilling {days}-day model forecasts...")
            try:
                import requests as req
                r = req.get(
                    "https://historical-forecast-api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": loc["lat"], "longitude": loc["lon"],
                        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,"
                                  "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                                  "pressure_msl,precipitation,cloud_cover,weather_code",
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "precipitation_unit": "inch",
                        "start_date": start, "end_date": end,
                        "models": all_models,
                        "timezone": "America/New_York",
                    }, timeout=30)
                if r.status_code != 200:
                    self.notify(f"[{loc['short']}] Historical forecast API: HTTP {r.status_code}")
                    continue

                hourly = r.json().get("hourly", {})
                times = hourly.get("time", [])
                n = len(times)
                issued = f"{start}T00:00:00"

                for our_key, suffix in model_params.items():
                    count = 0
                    for i in range(n):
                        valid = times[i]
                        try:
                            vdt = datetime.fromisoformat(valid)
                            sdt = datetime.fromisoformat(start)
                            lead = max(0, int((vdt - sdt).total_seconds() / 3600))
                        except:
                            lead = i

                        def _g(var):
                            col = f"{var}_{suffix}"
                            arr = hourly.get(col, hourly.get(var, []))
                            return arr[i] if i < len(arr) else None

                        self.db.insert_forecast(
                            loc_key, our_key, issued, valid, lead,
                            temp_f=_g("temperature_2m"),
                            humidity=_g("relative_humidity_2m"),
                            dewpoint_f=_g("dew_point_2m"),
                            wind_mph=_g("wind_speed_10m"),
                            wind_gust_mph=_g("wind_gusts_10m"),
                            wind_dir=_g("wind_direction_10m"),
                            pressure_mb=_g("pressure_msl"),
                            precip_in=_g("precipitation"),
                            cloud_cover=_g("cloud_cover"),
                            weather_code=_g("weather_code"))
                        count += 1
                    self.notify(f"[{loc['short']}] {our_key}: {count} forecast hours")
            except Exception as e:
                self.notify(f"[{loc['short']}] Forecast backfill error: {e}")
            time.sleep(1)

    # ── Retraining ────────────────────────────────────────────────────────

    def retrain(self):
        for loc_key, loc in LOCATIONS.items():
            self.notify(f"[{loc['short']}] Retraining ML models...")
            results = self.ml.train_all(loc_key)
            for key, info in results.items():
                if info.get("status") == "trained":
                    self.notify(
                        f"[{loc['short']}] {key}: "
                        f"MAE={info['mae']}, "
                        f"{info['improvement_pct']:+.1f}% vs {info['best_individual']}")
                else:
                    self.notify(
                        f"[{loc['short']}] {key}: "
                        f"need more data ({info.get('n_samples', 0)} samples)")

    # ── Run Cycle ─────────────────────────────────────────────────────────

    def run_cycle(self):
        try:
            self.notify("━━━ Collection cycle ━━━")
            self.collect_observations()
            self.collect_metar()           # KPWM conditions
            self.collect_forecasts()
            self.verify_predictions()      # Score past predictions (learning)
            self.verify_conditions()       # Score weather type accuracy
            self.generate_ensemble()       # Tiered prediction
            self.collect_alerts()
            if self.ha_publisher:
                self.ha_publisher.publish_all(self)
            self.notify("━━━ Cycle complete ━━━")
        except Exception as e:
            self.notify(f"[ERROR] Cycle failed: {e}")
            log.error(traceback.format_exc())

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.notify("WeatherOracle v2 started")

    def stop(self):
        self._running = False
        self.notify("WeatherOracle stopped")

    def _loop(self):
        self.run_cycle()
        self.collect_discussion()

        interval = self.config.get("collection_interval_min", 15) * 60
        retrain_interval = self.config.get("retrain_interval_hours", 6) * 3600
        advisor_interval = self.config.get("advisor_interval_hours", 4) * 3600
        last_retrain = 0
        last_discussion = time.time()
        last_advisor = 0

        while self._running:
            time.sleep(interval)
            if not self._running:
                break
            self.run_cycle()

            # Periodic retraining
            if time.time() - last_retrain >= retrain_interval:
                self.retrain()
                last_retrain = time.time()

            # Refresh AFD every 4 hours
            if time.time() - last_discussion >= 14400:
                self.collect_discussion()
                last_discussion = time.time()

            # Run Claude advisor periodically
            if time.time() - last_advisor >= advisor_interval:
                self.run_advisor()
                last_advisor = time.time()

    # ── Getters for GUI ───────────────────────────────────────────────────

    def get_current(self, location: str) -> Optional[dict]:
        return self.current_obs.get(location) or self.db.get_latest_obs(location)

    def get_crosscheck(self, location: str) -> dict:
        return self.crosscheck.get(location, {})

    def get_forecast(self, location: str) -> list:
        return self.db.get_ensemble_forecast(location)

    def get_raw_forecasts(self, location: str) -> list:
        return self.db.get_latest_forecasts(location)

    def get_alerts_for(self, location: str) -> list:
        return self.alerts.get(location, [])
