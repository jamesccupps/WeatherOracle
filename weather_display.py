#!/usr/bin/env python3
"""
WeatherOracle Display v1.0
Beautiful graphical weather dashboard served as a local web app.
Reads from the WeatherOracle SQLite database and displays:
  - Real-time conditions for both sites
  - Hourly forecast with temperature/precip/wind charts
  - Model accuracy comparison
  - NWS alerts
  - Sensor cross-check data

No extra dependencies — uses Python's built-in http.server.
"""

import http.server
import json
import os
import sqlite3
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Find the WeatherOracle database
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = SCRIPT_DIR / "data" / "weather_oracle.db"

# Import locations from config
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from core.config import LOCATIONS
except ImportError:
    LOCATIONS = {"location_1": {"name": "Location 1", "short": "LOC1"}}

if not DB_PATH.exists():
    # Try parent folder
    DB_PATH = SCRIPT_DIR.parent / "WeatherOracle" / "data" / "weather_oracle.db"

PORT = 8847


def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def query_db(sql, params=()):
    conn = get_db()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── API Endpoints ────────────────────────────────────────────────────────────

def api_current(location):
    """Get latest observation for a location."""
    rows = query_db("""
        SELECT * FROM observations
        WHERE location=? AND source='tempest'
        ORDER BY timestamp DESC LIMIT 1
    """, (location,))
    result = rows[0] if rows else {}

    # Cross-check sensors
    crosscheck = {}
    for src in ("yolink", "outdoor_sensor", "ha_average"):
        xc = query_db("""
            SELECT temp_f, humidity, timestamp FROM observations
            WHERE location=? AND source=?
            ORDER BY timestamp DESC LIMIT 1
        """, (location, src))
        if xc:
            crosscheck[src] = xc[0]
    result["crosscheck"] = crosscheck
    return result


def api_forecast(location):
    """Get ensemble forecast for a location."""
    rows = query_db("""
        SELECT * FROM ensemble_forecasts
        WHERE location=?
        ORDER BY issued_at DESC, valid_at ASC LIMIT 200
    """, (location,))
    if not rows:
        # Fallback: build forecast from raw model data if no ensemble exists
        raw = query_db("""
            SELECT valid_at,
                   AVG(temp_f) as temp_f,
                   AVG(humidity) as humidity,
                   AVG(wind_mph) as wind_mph,
                   AVG(precip_prob) as precip_prob,
                   AVG(cloud_cover) as cloud_cover,
                   MIN(weather_code) as weather_code,
                   MIN(lead_hours) as lead_hours,
                   75 as confidence,
                   'raw_avg' as method
            FROM forecasts
            WHERE location=?
              AND issued_at = (SELECT MAX(issued_at) FROM forecasts WHERE location=?)
            GROUP BY valid_at
            ORDER BY valid_at ASC
            LIMIT 72
        """, (location, location))
        return raw
    issued = rows[0]["issued_at"]
    return [r for r in rows if r["issued_at"] == issued]


def api_raw_forecasts(location):
    """Get raw model forecasts for comparison."""
    result = {}
    for model in ("gfs", "hrrr", "ecmwf", "icon", "gem", "jma", "tempest_bf"):
        # Get the latest issued_at for this model
        issued_row = query_db("""
            SELECT issued_at FROM forecasts
            WHERE location=? AND model=?
            ORDER BY issued_at DESC LIMIT 1
        """, (location, model))
        if not issued_row:
            continue
        issued = issued_row[0]["issued_at"]
        rows = query_db("""
            SELECT valid_at, temp_f, humidity, wind_mph, precip_prob,
                   weather_code, lead_hours
            FROM forecasts
            WHERE location=? AND model=? AND issued_at=?
            ORDER BY valid_at ASC LIMIT 80
        """, (location, model, issued))
        if rows:
            result[model] = rows
    return result


def api_recent_obs(location, hours=48):
    """Get recent observations for sparklines."""
    # Use multiple timestamp format strategies since the DB has mixed formats
    # (UTC with +00:00, local time without tz, etc)
    since_utc = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    since_local = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    # Use the earlier of the two to cast a wide net
    since = min(since_utc, since_local)
    rows = query_db("""
        SELECT timestamp, temp_f, humidity, wind_mph, pressure_mb,
               dewpoint_f, precip_in
        FROM observations
        WHERE location=? AND source IN ('tempest', 'openmeteo_archive', 'ha_outdoor_sensor', 'ha_yolink', 'ha_average')
          AND substr(timestamp, 1, 19) >= ? AND temp_f IS NOT NULL
        GROUP BY substr(timestamp, 1, 13)
        ORDER BY timestamp ASC
        LIMIT 200
    """, (location, since))
    # Fallback: if nothing found, just grab the latest 48 rows
    if not rows:
        rows = query_db("""
            SELECT timestamp, temp_f, humidity, wind_mph, pressure_mb,
                   dewpoint_f, precip_in
            FROM observations
            WHERE location=? AND source IN ('tempest', 'openmeteo_archive', 'ha_outdoor_sensor', 'ha_yolink', 'ha_average')
              AND temp_f IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (location, hours))
        rows.reverse()
    return rows


def api_stats():
    """Database statistics."""
    stats = {}
    for tbl in ("observations", "forecasts", "ensemble_forecasts"):
        r = query_db(f"SELECT COUNT(*) as n FROM {tbl}")
        stats[tbl] = r[0]["n"]
    for loc in LOCATIONS:
        r = query_db("SELECT COUNT(*) as n FROM observations WHERE location=?", (loc,))
        stats[f"obs_{loc}"] = r[0]["n"]

    # ML model info
    from pathlib import Path
    model_dir = SCRIPT_DIR / "models"
    if model_dir.exists():
        stats["ml_models"] = len(list(model_dir.glob("*.pkl")))
    else:
        stats["ml_models"] = 0

    # Training log
    log = query_db("""
        SELECT location, variable, bucket, n_samples, cv_mae,
               ensemble_mae, best_individual, improvement_pct, timestamp
        FROM training_log ORDER BY timestamp DESC LIMIT 20
    """)
    stats["training_log"] = log
    return stats


def api_debug():
    """Debug endpoint — shows table counts and sample rows."""
    debug = {}
    for tbl in ("observations", "forecasts", "ensemble_forecasts", "training_log"):
        r = query_db(f"SELECT COUNT(*) as n FROM {tbl}")
        debug[f"{tbl}_count"] = r[0]["n"]

    # Check ensemble by location
    for loc in LOCATIONS:
        r = query_db("""
            SELECT COUNT(*) as n FROM ensemble_forecasts WHERE location=?
        """, (loc,))
        debug[f"ensemble_{loc}"] = r[0]["n"]

        # Sample latest ensemble row
        sample = query_db("""
            SELECT issued_at, valid_at, temp_f, confidence, method
            FROM ensemble_forecasts WHERE location=?
            ORDER BY issued_at DESC LIMIT 1
        """, (loc,))
        debug[f"ensemble_{loc}_latest"] = sample[0] if sample else None

        # Recent obs count (48h window)
        since = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        r = query_db("""
            SELECT COUNT(*) as n FROM observations
            WHERE location=? AND source IN ('tempest','openmeteo_archive','ha_outdoor_sensor','ha_yolink','ha_average')
            AND timestamp >= ?
        """, (loc, since))
        debug[f"recent_obs_{loc}_48h"] = r[0]["n"]

        # Recent obs with local timezone
        since_local = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
        r2 = query_db("""
            SELECT COUNT(*) as n FROM observations
            WHERE location=? AND source IN ('tempest','openmeteo_archive','ha_outdoor_sensor','ha_yolink','ha_average')
            AND timestamp >= ?
        """, (loc, since_local))
        debug[f"recent_obs_{loc}_48h_local"] = r2[0]["n"]

        # Raw forecast counts per model
        for model in ("gfs", "hrrr", "ecmwf", "icon", "gem", "jma", "tempest_bf"):
            r = query_db("""
                SELECT COUNT(*) as n FROM forecasts
                WHERE location=? AND model=?
            """, (loc, model))
            debug[f"fc_{loc}_{model}"] = r[0]["n"]

    return debug


def api_debug():
    """Debug: show table counts and sample timestamps."""
    result = {}
    for tbl in ("observations", "forecasts", "ensemble_forecasts"):
        count = query_db(f"SELECT COUNT(*) as n FROM {tbl}")
        sample = query_db(f"SELECT * FROM {tbl} ORDER BY rowid DESC LIMIT 2")
        result[tbl] = {
            "count": count[0]["n"],
            "latest_rows": sample,
        }
    # Show distinct sources
    sources = query_db("SELECT source, COUNT(*) as n FROM observations GROUP BY source")
    result["sources"] = sources
    # Show distinct models in forecasts
    models = query_db("SELECT model, COUNT(*) as n FROM forecasts GROUP BY model")
    result["forecast_models"] = models
    return result


# ─── HTML Dashboard ──────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WeatherOracle — Southern Maine</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e17;
  --bg2: #111827;
  --card: #161f30;
  --card-hi: #1c2840;
  --border: #243049;
  --fg: #e2e8f0;
  --fg2: #94a3b8;
  --fg3: #64748b;
  --apt: #34d399;
  --apt-dim: #065f46;
  --apt-bg: rgba(52,211,153,0.08);
  --occ: #60a5fa;
  --occ-dim: #1e3a5f;
  --occ-bg: rgba(96,165,250,0.08);
  --warn: #f87171;
  --warm: #fb923c;
  --cool: #38bdf8;
  --precip: #818cf8;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--fg);
  min-height: 100vh;
}
.container { max-width: 1500px; margin: 0 auto; padding: 20px; }

/* Header */
.header {
  display: flex; align-items: baseline; gap: 16px;
  padding: 12px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.header h1 {
  font-size: 22px; font-weight: 700; letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--apt), var(--occ));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header .sub { color: var(--fg3); font-size: 13px; }
.header .updated { margin-left: auto; color: var(--fg3); font-size: 12px;
  font-family: 'JetBrains Mono', monospace; }

/* Grid layout */
.dual { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
@media (max-width: 900px) { .dual { grid-template-columns: 1fr; } }

/* Cards */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.card.apt::before { background: linear-gradient(90deg, var(--apt), transparent); }
.card.occ::before { background: linear-gradient(90deg, var(--occ), transparent); }

.card-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.5px; margin-bottom: 16px;
}
.card.apt .card-title { color: var(--apt); }
.card.occ .card-title { color: var(--occ); }

/* Current conditions */
.current-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 8px 24px;
  align-items: baseline;
}
.big-temp {
  font-size: 64px; font-weight: 700; line-height: 1;
  grid-row: span 2;
}
.card.apt .big-temp { color: var(--apt); }
.card.occ .big-temp { color: var(--occ); }
.condition { font-size: 18px; color: var(--fg); align-self: end; }
.feels { font-size: 13px; color: var(--fg3); }

.metrics {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 12px; margin-top: 20px;
  padding-top: 16px; border-top: 1px solid var(--border);
}
.metric { text-align: center; }
.metric .val {
  font-size: 20px; font-weight: 700; color: var(--fg);
  font-family: 'JetBrains Mono', monospace;
}
.metric .lbl { font-size: 10px; color: var(--fg3); text-transform: uppercase;
  letter-spacing: 0.8px; margin-top: 2px; }

/* Cross-check */
.crosscheck {
  margin-top: 16px; padding: 10px 12px;
  background: var(--bg2); border-radius: 8px;
  font-size: 12px; color: var(--fg3);
  font-family: 'JetBrains Mono', monospace;
}
.crosscheck .src { color: var(--fg2); }
.crosscheck .delta { padding: 2px 6px; border-radius: 4px; font-size: 11px; }
.crosscheck .delta.good { background: rgba(52,211,153,0.15); color: var(--apt); }
.crosscheck .delta.warn { background: rgba(248,113,113,0.15); color: var(--warn); }

/* Charts */
.chart-wrap {
  margin-top: 16px;
  height: 180px;
  position: relative;
}
canvas { width: 100% !important; height: 100% !important; }

/* Forecast table */
.fc-table {
  width: 100%; border-collapse: collapse;
  font-size: 13px; margin-top: 12px;
}
.fc-table th {
  text-align: center; padding: 8px 6px;
  font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.8px; color: var(--fg3);
  border-bottom: 1px solid var(--border);
  font-weight: 500;
}
.fc-table td {
  text-align: center; padding: 7px 6px;
  border-bottom: 1px solid rgba(36,48,73,0.5);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.fc-table tr:hover td { background: var(--card-hi); }
.fc-table .time-col { text-align: left; color: var(--fg2); font-size: 12px; }
.fc-table .temp-col { font-weight: 700; }
.fc-table .precip-col { color: var(--precip); }
.fc-table .sky-col { font-size: 14px; }
.fc-scroll { max-height: 520px; overflow-y: auto; }
.fc-scroll::-webkit-scrollbar { width: 4px; }
.fc-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Alerts */
.alert-bar {
  background: rgba(248,113,113,0.1);
  border: 1px solid rgba(248,113,113,0.3);
  border-radius: 8px; padding: 12px 16px;
  margin-bottom: 20px; display: none;
}
.alert-bar.visible { display: block; }
.alert-bar .alert-icon { color: var(--warn); font-size: 16px; margin-right: 8px; }
.alert-bar .alert-text { color: var(--warn); font-size: 13px; }

/* Stats bar */
.stats-bar {
  display: flex; gap: 24px; padding: 16px 0;
  border-top: 1px solid var(--border);
  margin-top: 24px;
  font-size: 11px; color: var(--fg3);
  font-family: 'JetBrains Mono', monospace;
}
.stats-bar .stat-val { color: var(--fg2); font-weight: 500; }

/* Confidence badge */
.conf {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 10px; font-weight: 700;
}
.conf.high { background: rgba(52,211,153,0.15); color: var(--apt); }
.conf.med { background: rgba(251,191,36,0.15); color: #fbbf24; }
.conf.low { background: rgba(248,113,113,0.15); color: var(--warn); }

/* Loading */
.loading {
  text-align: center; padding: 40px;
  color: var(--fg3); font-size: 14px;
}
.loading .spinner {
  width: 24px; height: 24px; border: 2px solid var(--border);
  border-top-color: var(--apt); border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Temp color */
.temp-cold { color: #38bdf8; }
.temp-cool { color: #67e8f9; }
.temp-mild { color: var(--fg); }
.temp-warm { color: #fb923c; }
.temp-hot { color: #ef4444; }

/* Tab system */
.tabs { display: flex; gap: 4px; margin-bottom: 20px; }
.tab {
  padding: 8px 18px; border-radius: 8px;
  font-size: 13px; font-weight: 500;
  background: transparent; border: 1px solid var(--border);
  color: var(--fg3); cursor: pointer;
  transition: all 0.2s;
}
.tab:hover { background: var(--card); color: var(--fg); }
.tab.active { background: var(--card); color: var(--fg);
  border-color: var(--apt); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Model comparison */
.model-row {
  display: grid; grid-template-columns: 140px repeat(3, 1fr) 80px;
  gap: 8px; padding: 8px 0;
  border-bottom: 1px solid rgba(36,48,73,0.5);
  font-size: 13px; align-items: center;
}
.model-row.hdr { font-size: 10px; color: var(--fg3); text-transform: uppercase;
  letter-spacing: 0.5px; font-weight: 500; }
.model-name { font-weight: 500; }
.model-bar { height: 6px; border-radius: 3px; background: var(--border); }
.model-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

/* Animation */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Daily forecast grid */
.daily-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.day-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px; text-align: center;
}
.day-name {
  font-size: 13px; font-weight: 700; color: var(--fg);
  margin-bottom: 2px;
}
.day-date { font-size: 11px; color: var(--fg3); margin-bottom: 10px; }
.day-icon { font-size: 32px; margin-bottom: 6px; }
.day-condition { font-size: 11px; color: var(--fg2); margin-bottom: 12px; min-height: 16px; }
.day-temps {
  display: flex; justify-content: center; gap: 16px;
  margin-bottom: 12px;
}
.day-hi { font-size: 22px; font-weight: 700; }
.day-lo { font-size: 22px; font-weight: 700; color: var(--fg3); }
.day-hi .arr { font-size: 11px; color: var(--warn); }
.day-lo .arr { font-size: 11px; color: var(--cool); }
.day-details {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
  font-size: 11px; color: var(--fg3);
  border-top: 1px solid var(--border); padding-top: 10px;
}
.day-detail-val {
  font-family: 'JetBrains Mono', monospace;
  color: var(--fg2); font-weight: 500; font-size: 12px;
}
.day-precip-bar {
  height: 4px; background: var(--border); border-radius: 2px;
  margin-top: 8px; overflow: hidden;
}
.day-precip-fill {
  height: 100%; border-radius: 2px;
  background: linear-gradient(90deg, var(--precip), #a78bfa);
  transition: width 0.5s;
}
.card { animation: fadeUp 0.4s ease both; }
.card:nth-child(2) { animation-delay: 0.05s; }
.card:nth-child(3) { animation-delay: 0.1s; }
.card:nth-child(4) { animation-delay: 0.15s; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>WeatherOracle</h1>
    <span class="sub">Southern Maine • ML-Powered Hyperlocal Forecast</span>
    <span class="updated" id="lastUpdate">Loading...</span>
  </div>

  <div id="alertBar" class="alert-bar">
    <span class="alert-icon">⚠</span>
    <span class="alert-text" id="alertText"></span>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('dashboard')">Dashboard</div>
    <div class="tab" onclick="switchTab('forecast')">72hr Forecast</div>
    <div class="tab" onclick="switchTab('models')">Model Comparison</div>
  </div>

  <!-- DASHBOARD TAB -->
  <div id="tab-dashboard" class="tab-content active">
    <div class="dual" id="currentCards">
      <div class="card apt" id="card-apartment"><div class="loading"><div class="spinner"></div>Loading location data...</div></div>
      <div class="card occ" id="card-occ"><div class="loading"><div class="spinner"></div>Loading location data...</div></div>
    </div>
    <div class="dual">
      <div class="card apt">
        <div class="card-title">Location 1 — 48hr Temperature</div>
        <div class="chart-wrap"><canvas id="chart-apt-temp"></canvas></div>
      </div>
      <div class="card occ">
        <div class="card-title">Location 2 — 48hr Temperature</div>
        <div class="chart-wrap"><canvas id="chart-occ-temp"></canvas></div>
      </div>
    </div>
    <div class="dual">
      <div class="card apt">
        <div class="card-title">Location 1 — 3-Day Outlook</div>
        <div id="daily-apt" class="daily-grid"></div>
      </div>
      <div class="card occ">
        <div class="card-title">Location 2 — 3-Day Outlook</div>
        <div id="daily-occ" class="daily-grid"></div>
      </div>
    </div>
  </div>
  <div id="tab-forecast" class="tab-content">
    <div class="dual">
      <div class="card apt">
        <div class="card-title">Location 1 — 72hr Ensemble Forecast</div>
        <div class="chart-wrap" style="height:200px"><canvas id="chart-apt-fc"></canvas></div>
        <div class="fc-scroll" id="fc-apt"></div>
      </div>
      <div class="card occ">
        <div class="card-title">Location 2 — 72hr Ensemble Forecast</div>
        <div class="chart-wrap" style="height:200px"><canvas id="chart-occ-fc"></canvas></div>
        <div class="fc-scroll" id="fc-occ"></div>
      </div>
    </div>
  </div>

  <!-- MODELS TAB -->
  <div id="tab-models" class="tab-content">
    <div class="dual">
      <div class="card apt">
        <div class="card-title">Apartment — Model Temperature Forecast Comparison</div>
        <div class="chart-wrap" style="height:250px"><canvas id="chart-apt-models"></canvas></div>
      </div>
      <div class="card occ">
        <div class="card-title">OCC — Model Temperature Forecast Comparison</div>
        <div class="chart-wrap" style="height:250px"><canvas id="chart-occ-models"></canvas></div>
      </div>
    </div>
    <div class="card" id="training-info">
      <div class="card-title" style="color:var(--apt)">ML Training Status</div>
      <div id="trainingLog"></div>
    </div>
  </div>

  <div class="stats-bar" id="statsBar"></div>
</div>

<script>
const WMO = {
  0:"Clear",1:"Mainly Clear",2:"Partly Cloudy",3:"Overcast",
  45:"Fog",48:"Rime Fog",51:"Lt Drizzle",53:"Drizzle",55:"Hvy Drizzle",
  56:"Frzg Drizzle",57:"Hvy Frzg Drizzle",
  61:"Lt Rain",63:"Rain",65:"Heavy Rain",66:"Frzg Rain",67:"Hvy Frzg Rain",
  71:"Lt Snow",73:"Snow",75:"Heavy Snow",77:"Snow Grains",
  80:"Lt Showers",81:"Showers",82:"Heavy Showers",
  85:"Lt Snow Shwrs",86:"Snow Showers",
  95:"Thunderstorm",96:"T-Storm+Hail",99:"Severe T-Storm"
};
const WMO_ICON = {
  0:"☀️",1:"🌤",2:"⛅",3:"☁️",45:"🌫",48:"🌫",
  51:"🌦",53:"🌧",55:"🌧",56:"🌧",57:"🌧",
  61:"🌦",63:"🌧",65:"🌧",66:"🌧",67:"🌧",
  71:"🌨",73:"❄️",75:"❄️",77:"❄️",80:"🌦",81:"🌧",82:"⛈",
  85:"🌨",86:"🌨",95:"⛈",96:"⛈",99:"⛈"
};
const MODEL_COLORS = {
  gfs:'#f87171', hrrr:'#fb923c', ecmwf:'#fbbf24',
  icon:'#34d399', gem:'#60a5fa', jma:'#a78bfa',
  tempest_bf:'#f472b6', ensemble:'#ffffff'
};
const MODEL_NAMES = {
  gfs:'GFS', hrrr:'HRRR', ecmwf:'ECMWF',
  icon:'ICON', gem:'GEM', jma:'JMA',
  tempest_bf:'Tempest'
};

let charts = {};

function tempColor(f) {
  if (f <= 20) return 'temp-cold';
  if (f <= 40) return 'temp-cool';
  if (f <= 70) return 'temp-mild';
  if (f <= 85) return 'temp-warm';
  return 'temp-hot';
}
function confClass(c) {
  if (c >= 70) return 'high';
  if (c >= 40) return 'med';
  return 'low';
}
function fmtTime(iso) {
  try {
    const d = new Date(iso);
    const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    let h = d.getHours();
    const ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12;
    return `${days[d.getDay()]} ${h}${ampm}`;
  } catch { return iso; }
}
function shortTime(iso) {
  try {
    const d = new Date(iso);
    let h = d.getHours();
    return `${h % 12 || 12}${h >= 12 ? 'p' : 'a'}`;
  } catch { return ''; }
}

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ── Build current conditions card ──
function buildCurrentCard(loc, data) {
  const el = document.getElementById('card-' + loc);
  if (!el) { console.warn('Missing element card-' + loc); return; }
  if (!data || !data.temp_f) {
    el.innerHTML = '<div class="loading">No data yet</div>';
    return;
  }
  const wc = data.weather_code;
  const icon = WMO_ICON[wc] || '';
  const cond = WMO[wc] || '';
  const locName = loc.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  let xcheckHtml = '';
  if (data.crosscheck) {
    const parts = [];
    for (const [src, xc] of Object.entries(data.crosscheck)) {
      if (xc.temp_f != null) {
        const delta = Math.abs(xc.temp_f - data.temp_f).toFixed(1);
        const cls = delta < 2 ? 'good' : 'warn';
        parts.push(`<span class="src">${src}:</span> ${xc.temp_f}°F <span class="delta ${cls}">Δ${delta}°</span>`);
      }
    }
    if (parts.length) {
      xcheckHtml = `<div class="crosscheck">${parts.join(' &nbsp;│&nbsp; ')}</div>`;
    }
  }

  el.innerHTML = `
    <div class="card-title">${locName}</div>
    <div class="current-grid">
      <div class="big-temp">${Math.round(data.temp_f)}°</div>
      <div class="condition">${icon} ${cond}</div>
      <div class="feels">Feels like ${data.feels_like_f || '--'}°F</div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="val">${data.humidity || '--'}%</div><div class="lbl">Humidity</div></div>
      <div class="metric"><div class="val">${data.dewpoint_f || '--'}°</div><div class="lbl">Dewpoint</div></div>
      <div class="metric"><div class="val">${data.wind_mph || '0'}</div><div class="lbl">Wind mph</div></div>
      <div class="metric"><div class="val">${data.pressure_mb || '--'}</div><div class="lbl">Pressure mb</div></div>
    </div>
    ${xcheckHtml}
    <div style="margin-top:12px;font-size:11px;color:var(--fg3);font-family:'JetBrains Mono',monospace">
      Tempest •
      ${data.timestamp ? data.timestamp.slice(0, 19) : ''}
    </div>
  `;
}

// ── Build forecast table ──
function buildForecastTable(loc, forecasts) {
  const prefix = (idx || 0) === 0 ? 'apt' : 'occ';
  const el = document.getElementById('fc-' + prefix);
  if (!el) { console.warn('Missing element fc-' + prefix); return; }
  if (!forecasts || !forecasts.length) {
    el.innerHTML = '<div class="loading">No forecast data</div>';
    return;
  }
  let html = `<table class="fc-table"><thead><tr>
    <th>Time</th><th>Temp</th><th>RH%</th><th>Wind</th>
    <th>Precip</th><th>Sky</th><th>Conf</th>
  </tr></thead><tbody>`;
  for (const fc of forecasts) {
    const wc = fc.weather_code;
    const icon = WMO_ICON[wc] || '';
    const sky = WMO[wc] || '';
    const conf = fc.confidence || 0;
    html += `<tr>
      <td class="time-col">${fmtTime(fc.valid_at)}</td>
      <td class="temp-col ${tempColor(fc.temp_f)}">${fc.temp_f != null ? fc.temp_f + '°' : '—'}</td>
      <td>${fc.humidity != null ? Math.round(fc.humidity) : '—'}</td>
      <td>${fc.wind_mph != null ? fc.wind_mph : '—'}</td>
      <td class="precip-col">${fc.precip_prob != null ? Math.round(fc.precip_prob) + '%' : '—'}</td>
      <td class="sky-col" title="${sky}">${icon}</td>
      <td><span class="conf ${confClass(conf)}">${Math.round(conf)}</span></td>
    </tr>`;
  }
  html += '</tbody></table>';
  el.innerHTML = html;
}

// ── Charts ──
function makeChart(canvasId, labels, datasets, opts = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (charts[canvasId]) charts[canvasId].destroy();

  const defaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: datasets.length > 1, labels: { color: '#94a3b8', font: { size: 10 } } },
    },
    scales: {
      x: { ticks: { color: '#64748b', font: { size: 10 }, maxRotation: 0, maxTicksLimit: 12 },
           grid: { color: 'rgba(36,48,73,0.5)' } },
      y: { ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } },
           grid: { color: 'rgba(36,48,73,0.5)' }, ...opts.yAxis },
    },
    elements: { point: { radius: 0, hoverRadius: 4 }, line: { tension: 0.3, borderWidth: 2 } },
    interaction: { mode: 'index', intersect: false },
  };

  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: { ...defaults, ...opts },
  });
}

function buildObsChart(loc, obs, idx) {
  if (!obs || !obs.length) return;
  const accent = (idx || 0) % 2 === 0 ? '#34d399' : '#60a5fa';
  const labels = obs.map(o => shortTime(o.timestamp));
  makeChart(`chart-${idx === 0 ? 'apt' : 'occ'}-temp`, labels, [
    {
      label: 'Temperature °F',
      data: obs.map(o => o.temp_f),
      borderColor: accent,
      backgroundColor: accent + '15',
      fill: true,
    }
  ]);
}

function buildForecastChart(loc, forecasts, idx) {
  const prefix = (idx || 0) === 0 ? 'apt' : 'occ';
  const accent = (idx || 0) % 2 === 0 ? '#34d399' : '#60a5fa';
  const labels = forecasts.map(f => fmtTime(f.valid_at));
  makeChart(`chart-${prefix}-fc`, labels, [
    {
      label: 'Temp °F',
      data: forecasts.map(f => f.temp_f),
      borderColor: accent,
      backgroundColor: accent + '15',
      fill: true,
      yAxisID: 'y',
    },
    {
      label: 'Precip %',
      data: forecasts.map(f => f.precip_prob),
      borderColor: '#818cf8',
      backgroundColor: '#818cf820',
      fill: true,
      yAxisID: 'y1',
    }
  ], {
    scales: {
      x: { ticks: { color: '#64748b', font: { size: 9 }, maxRotation: 45, maxTicksLimit: 16 },
           grid: { color: 'rgba(36,48,73,0.5)' } },
      y: { position: 'left', ticks: { color: '#64748b', font: { size: 10 } },
           grid: { color: 'rgba(36,48,73,0.5)' } },
      y1: { position: 'right', min: 0, max: 100,
            ticks: { color: '#818cf8', font: { size: 10 } },
            grid: { drawOnChartArea: false } },
    }
  });
}

function buildModelChart(loc, rawModels) {
  const prefix = (idx || 0) === 0 ? 'apt' : 'occ';
  const datasets = [];
  let labels = null;
  for (const [model, data] of Object.entries(rawModels)) {
    if (!data.length) continue;
    if (!labels) labels = data.map(d => shortTime(d.valid_at));
    datasets.push({
      label: MODEL_NAMES[model] || model,
      data: data.map(d => d.temp_f),
      borderColor: MODEL_COLORS[model] || '#888',
      borderWidth: 1.5,
    });
  }
  if (labels) {
    makeChart(`chart-${prefix}-models`, labels, datasets);
  }
}

function buildDailyForecast(loc, forecasts, observations) {
  const prefix = (idx || 0) === 0 ? 'apt' : 'occ';
  const el = document.getElementById('daily-' + prefix);
  if (!el) return;
  if (!forecasts || !forecasts.length) {
    el.innerHTML = '<div style="color:var(--fg3);font-size:13px;grid-column:span 3;text-align:center;padding:20px">Waiting for forecast data...</div>';
    return;
  }

  // Build today's observed temps for accurate hi/lo
  const todayStr = new Date().toISOString().slice(0, 10);
  const todayObsTemps = [];
  if (observations && observations.length) {
    for (const ob of observations) {
      try {
        const d = new Date(ob.timestamp || ob.time);
        if (d.toISOString().slice(0, 10) === todayStr && ob.temp_f != null) {
          todayObsTemps.push(ob.temp_f);
        }
      } catch(e) {}
    }
  }

  // Group forecasts by calendar date
  const byDate = {};
  for (const fc of forecasts) {
    try {
      const d = new Date(fc.valid_at);
      const key = d.toISOString().slice(0, 10);
      if (!byDate[key]) byDate[key] = [];
      byDate[key].push({ ...fc, _hour: d.getHours(), _date: d });
    } catch(e) {}
  }

  // Get 3 days (today + next 2, or next 3 if today has few hours left)
  const dateKeys = Object.keys(byDate).sort();
  const days = dateKeys.slice(0, 3);

  if (!days.length) {
    el.innerHTML = '<div style="color:var(--fg3);font-size:13px;grid-column:span 3;text-align:center;padding:20px">No daily data yet</div>';
    return;
  }

  const dayNames = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  let html = '';
  for (const dateKey of days) {
    const hours = byDate[dateKey];
    const d = new Date(dateKey + 'T12:00:00');

    // Aggregate
    let temps = hours.map(h => h.temp_f).filter(t => t != null);

    // For today: merge actual observed temps so hi/lo reflects the full day
    if (dateKey === todayStr && todayObsTemps.length) {
      temps = todayObsTemps.concat(temps);
    }

    const humids = hours.map(h => h.humidity).filter(h => h != null);
    const winds = hours.map(h => h.wind_mph).filter(w => w != null);
    const precips = hours.map(h => h.precip_prob).filter(p => p != null);
    const codes = hours.map(h => h.weather_code).filter(c => c != null);

    const hi = temps.length ? Math.round(Math.max(...temps)) : '--';
    const lo = temps.length ? Math.round(Math.min(...temps)) : '--';
    const avgHum = humids.length ? Math.round(humids.reduce((a,b)=>a+b,0)/humids.length) : '--';
    const maxWind = winds.length ? Math.round(Math.max(...winds)) : '--';
    const maxPrecip = precips.length ? Math.round(Math.max(...precips)) : 0;
    const avgPrecip = precips.length ? Math.round(precips.reduce((a,b)=>a+b,0)/precips.length) : 0;

    // Dominant weather code: pick the most common, biased toward midday
    const midday = hours.filter(h => h._hour >= 8 && h._hour <= 18);
    const codePool = (midday.length ? midday : hours).map(h => h.weather_code).filter(c => c != null);
    const codeCount = {};
    for (const c of codePool) codeCount[c] = (codeCount[c]||0) + 1;
    const dominantCode = Object.entries(codeCount).sort((a,b) => b[1]-a[1])[0];
    const wc = dominantCode ? parseInt(dominantCode[0]) : null;
    const icon = wc != null ? (WMO_ICON[wc] || '') : '—';
    const condition = wc != null ? (WMO[wc] || '') : '';

    html += `
      <div class="day-card">
        <div class="day-name">${dayNames[d.getDay()]}</div>
        <div class="day-date">${monthNames[d.getMonth()]} ${d.getDate()}</div>
        <div class="day-icon">${icon}</div>
        <div class="day-condition">${condition}</div>
        <div class="day-temps">
          <div class="day-hi ${tempColor(hi)}"><span class="arr">▲</span> ${hi}°</div>
          <div class="day-lo"><span class="arr">▼</span> ${lo}°</div>
        </div>
        <div class="day-details">
          <div>Humidity</div><div class="day-detail-val">${avgHum}%</div>
          <div>Wind</div><div class="day-detail-val">${maxWind} mph</div>
          <div>Precip</div><div class="day-detail-val">${maxPrecip}%</div>
        </div>
        <div class="day-precip-bar"><div class="day-precip-fill" style="width:${avgPrecip}%"></div></div>
      </div>
    `;
  }
  el.innerHTML = html;
}

function buildStats(stats) {
  const el = document.getElementById('statsBar');
  el.innerHTML = `
    <span>Observations: <span class="stat-val">${(stats.observations || 0).toLocaleString()}</span></span>
    <span>Forecasts: <span class="stat-val">${(stats.forecasts || 0).toLocaleString()}</span></span>
    <span>ML Models: <span class="stat-val">${stats.ml_models || 0}</span></span>
    <span>Observations: <span class="stat-val">${(stats.observations || 0).toLocaleString()}</span></span>
    <span>Forecasts: <span class="stat-val">${(stats.forecasts || 0).toLocaleString()}</span></span>
  `;

  // Training log
  const logEl = document.getElementById('trainingLog');
  if (stats.training_log && stats.training_log.length) {
    let html = `<div class="model-row hdr">
      <div>Model</div><div>Variable</div><div>Bucket</div><div>Samples</div><div>Improvement</div>
    </div>`;
    for (const e of stats.training_log.slice(0, 12)) {
      const imp = e.improvement_pct || 0;
      const color = imp > 0 ? 'var(--apt)' : 'var(--warn)';
      html += `<div class="model-row">
        <div class="model-name">${e.location}</div>
        <div>${e.variable}</div>
        <div>${e.bucket}</div>
        <div>${e.n_samples || 0}</div>
        <div style="color:${color};font-weight:700">${imp > 0 ? '+' : ''}${imp.toFixed(1)}%</div>
      </div>`;
    }
    logEl.innerHTML = html;
  } else {
    logEl.innerHTML = '<div style="color:var(--fg3);font-size:13px">No training runs yet</div>';
  }
}

// ── Main data load ──
async function loadAll() {
  const results = {};
  const endpoints = [
    ['aptCur', '/api/current/apartment'],
    ['occCur', '/api/current/occ'],
    ['aptFc', '/api/forecast/apartment'],
    ['occFc', '/api/forecast/occ'],
    ['aptObs', '/api/recent/apartment'],
    ['occObs', '/api/recent/occ'],
    ['aptRaw', '/api/raw/apartment'],
    ['occRaw', '/api/raw/occ'],
    ['stats', '/api/stats'],
  ];

  // Fetch all endpoints independently
  await Promise.all(endpoints.map(async ([key, url]) => {
    try {
      results[key] = await fetchJSON(url);
    } catch (e) {
      console.warn('API error for', url, e);
      results[key] = key.endsWith('Fc') || key.endsWith('Obs') ? [] : {};
    }
  }));

  console.log('API results:', Object.fromEntries(
    Object.entries(results).map(([k,v]) => [k, Array.isArray(v) ? v.length + ' items' : 'object'])
  ));

  // Build each section independently so one failure doesn't kill others
  const sections = [
    () => buildCurrentCard('apartment', results.aptCur),
    () => buildCurrentCard('occ', results.occCur),
    () => buildForecastTable('apartment', results.aptFc || []),
    () => buildForecastTable('occ', results.occFc || []),
    () => buildDailyForecast('apartment', results.aptFc || [], results.aptObs || []),
    () => buildDailyForecast('occ', results.occFc || [], results.occObs || []),
    () => buildObsChart('apartment', results.aptObs || []),
    () => buildObsChart('occ', results.occObs || []),
    () => buildForecastChart('apartment', results.aptFc || []),
    () => buildForecastChart('occ', results.occFc || []),
    () => buildModelChart('apartment', results.aptRaw || {}),
    () => buildModelChart('occ', results.occRaw || {}),
    () => buildStats(results.stats || {}),
  ];

  for (const fn of sections) {
    try { fn(); } catch (e) { console.error('Section error:', e); }
  }

  document.getElementById('lastUpdate').textContent =
    'Updated: ' + new Date().toLocaleTimeString();
}

// Initial load + auto-refresh every 5 min
loadAll();
setInterval(loadAll, 300000);
</script>
</body>
</html>"""


# ─── HTTP Server ──────────────────────────────────────────────────────────────

class WeatherHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif path.startswith("/api/"):
            self._handle_api(path)

        else:
            self.send_error(404)

    def _handle_api(self, path):
        try:
            if path.startswith("/api/current/"):
                loc = path.split("/")[-1]
                data = api_current(loc)
                print(f"  [API] current/{loc}: temp={data.get('temp_f', 'NONE')}")
            elif path.startswith("/api/forecast/"):
                loc = path.split("/")[-1]
                data = api_forecast(loc)
                print(f"  [API] forecast/{loc}: {len(data)} rows")
            elif path.startswith("/api/recent/"):
                loc = path.split("/")[-1]
                data = api_recent_obs(loc)
                print(f"  [API] recent/{loc}: {len(data)} rows")
            elif path.startswith("/api/raw/"):
                loc = path.split("/")[-1]
                data = api_raw_forecasts(loc)
                print(f"  [API] raw/{loc}: {sum(len(v) for v in data.values())} rows")
            elif path == "/api/stats":
                data = api_stats()
            elif path == "/api/debug":
                data = api_debug()
            else:
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        # Show errors, suppress normal GET requests
        if len(args) >= 2 and '200' not in str(args[1]):
            print(f"[HTTP] {format % args}")


def main():
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run WeatherOracle first to create the database.")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"WeatherOracle Display")
    print(f"Database: {DB_PATH}")
    print(f"Starting server on http://localhost:{PORT}")

    server = http.server.HTTPServer(("127.0.0.1", PORT), WeatherHandler)

    # Open browser after short delay
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"Dashboard: http://localhost:{PORT}")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
