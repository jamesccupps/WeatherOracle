# WeatherOracle

**Hyperlocal ML weather prediction** that fuses data from your WeatherFlow Tempest station, 7 weather models, METAR airport observations, and NWS forecasts — then learns which models perform best for your specific location over time.

![WeatherOracle Dashboard](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

WeatherOracle collects hourly forecasts from 7 weather models (GFS, HRRR, ECMWF, ICON, GEM, JMA, and Tempest BetterForecast), compares them against what actually happened at your Tempest station, and uses that verification data to build an ML ensemble that outperforms any individual model.

**Key features:**

- **Verification-driven learning** — every cycle, past predictions are scored against reality. Models that perform well get higher weight automatically.
- **Tiered prediction** — 0-6h uses persistence nowcasting (trend extrapolation from your Tempest), 6-12h blends obs with models, 12-72h uses pure ML ensemble.
- **Adaptive ML** — automatically scales model complexity to data size (Ridge regression for small datasets, Gradient Boosting for large ones). Includes an overfitting guard that rejects models where cross-validation MAE diverges too far from training MAE.
- **Exponential decay scoreboard** — recent model performance is weighted more heavily than old data (48h half-life), so the system responds quickly to weather regime changes.
- **Claude AI advisor** (optional) — periodically analyzes the NWS forecast discussion, current model performance, and observations to recommend model weight adjustments. Tracks whether its own advice helped or hurt accuracy.
- **Multi-location support** — monitor multiple Tempest stations simultaneously with independent ML models per location.
- **Home Assistant integration** — publishes forecasts as HA sensors with a custom Lovelace card.
- **METAR verification** — uses nearby airport observations to verify precipitation type and weather conditions.
- **Observation quality gate** — validates all incoming data against physical bounds before storing.

## Requirements

- **Python 3.10+**
- **WeatherFlow Tempest** weather station with API token ([get one here](https://tempestwx.com/settings/tokens))
- **Home Assistant** (optional) — for sensor cross-checking and publishing forecasts
- **Anthropic API key** (optional) — for Claude weather advisor

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/WeatherOracle.git
cd WeatherOracle

# Install dependencies
pip install -r requirements.txt

# Copy the example config
cp config.example.json data/weather_oracle_config.json

# Edit with your settings (see Configuration below)
# Then run:
python main.py
```

## Configuration

Edit `data/weather_oracle_config.json` with your details. The most important part is the `locations` section:

```json
{
  "tempest_api_token": "your-tempest-api-token",
  "ha_url": "http://your-ha-instance:8123",
  "ha_token": "your-long-lived-access-token",
  "claude_api_key": "",
  "locations": {
    "home": {
      "name": "My House",
      "short": "HOME",
      "tempest_station": 12345,
      "lat": 40.7128,
      "lon": -74.0060,
      "metar_station": "KJFK",
      "nws_office": "OKX"
    },
    "office": {
      "name": "Office Downtown",
      "short": "OFC",
      "tempest_station": 67890,
      "lat": 40.7580,
      "lon": -73.9855,
      "metar_station": "KLGA",
      "nws_office": "OKX"
    }
  }
}
```

### Location fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Display name for the location |
| `short` | Yes | Short abbreviation (3-4 chars) for log messages and HA entities |
| `tempest_station` | Yes | Your Tempest station ID (find it at tempestwx.com) |
| `lat` / `lon` | Yes | Latitude and longitude of the station |
| `metar_station` | No | Nearest airport ICAO code (e.g. `KJFK`) for condition verification |
| `nws_office` | No | NWS forecast office code for forecast discussion (e.g. `OKX`) |

### Finding your settings

- **Tempest station ID**: Go to [tempestwx.com](https://tempestwx.com), click your station → the number in the URL is your station ID
- **Tempest API token**: [tempestwx.com/settings/tokens](https://tempestwx.com/settings/tokens)
- **METAR station**: Find your nearest airport at [aviationweather.gov](https://aviationweather.gov/metar)
- **NWS office**: Look up at [weather.gov](https://www.weather.gov) — it's the 3-letter code in your local forecast URL

### API keys via environment variables

Instead of putting secrets in the config file, you can use environment variables:

```bash
export WEATHERORACLE_TEMPEST_TOKEN=your_token
export WEATHERORACLE_HA_TOKEN=your_ha_token
export WEATHERORACLE_CLAUDE_KEY=sk-ant-...
```

Or create a `.env` file (requires `python-dotenv`):

```
WEATHERORACLE_TEMPEST_TOKEN=your_token
WEATHERORACLE_HA_TOKEN=your_ha_token
WEATHERORACLE_CLAUDE_KEY=sk-ant-...
```

## How It Works

### Collection Cycle (every 5-15 min)

1. **Observe** — Pull current conditions from each Tempest station + HA cross-check sensors + METAR
2. **Forecast** — Fetch hourly forecasts from all 7 weather models via Open-Meteo and Tempest APIs
3. **Verify** — Score past predictions against what actually happened (this is the learning signal)
4. **Predict** — Generate ensemble forecast using tiered approach:
   - **0-6h**: Persistence (Tempest trend extrapolation) blended with models
   - **6-12h**: Model-dominant with observation sanity check
   - **12-72h**: Pure ML ensemble with verification-based dynamic weights
5. **Publish** — Push to Home Assistant, update GUI

### ML Training (every 1-6 hours)

- Pairs historical model forecasts with actual observations
- Trains per-variable (temp, humidity, wind, dewpoint, pressure), per-lead-time-bucket (0-6h, 6-12h, 12-24h, 24-48h, 48-72h) models
- Adapts complexity: Ridge regression for <100 samples, light GBR for 100-500, full GBR for 500+
- Rejects overfitted models automatically (CV/train MAE ratio > 10x)

### Weather Models Used

| Model | Source | Strength | Coverage |
|-------|--------|----------|----------|
| **HRRR** | NOAA | Best 0-18h, hourly updates | US only |
| **GFS** | NOAA | Reliable 3-7 day | Global |
| **ECMWF IFS** | European | Best overall skill | Global |
| **ICON** | DWD (Germany) | Good mesoscale | Global |
| **GEM** | Canada | Northern air masses | Global |
| **JMA** | Japan | Independent view | Global |
| **Tempest BF** | WeatherFlow | ML-enhanced, station-local | Per-station |

## Deep Backfill

For faster ML training, use the **Data & Stats** tab to run a deep backfill. This pulls historical observations and model forecasts from Open-Meteo's archive API, giving the ML months or years of training data instead of waiting days.

Recommended: Start with 30-90 days of backfill, then let live collection take over.

## Home Assistant Integration

WeatherOracle publishes sensors to HA via the REST API. A custom Lovelace card is included in `ha_card/weatheroracle-card.js`.

### Setup

1. Copy `ha_card/weatheroracle-card.js` to your HA `www/` directory
2. Add as a Lovelace resource: `/local/weatheroracle-card.js`
3. Configure your HA URL and long-lived access token in WeatherOracle settings
4. The sensors will be created automatically on the first collection cycle

> **Note:** The Lovelace card template assumes two locations. If you have a different number of locations or want to change the layout, edit the HTML template in `weatheroracle-card.js` — the entity IDs follow the pattern `sensor.weatheroracle_{short}_current` where `{short}` is your location's short code (lowercased).

## Web Display (Optional)

A standalone web dashboard is available via `python weather_display.py` (runs on `localhost:8847`). This is a secondary interface — the tkinter GUI is the primary app. The web display template supports up to two locations with fixed card layouts.

## Claude AI Advisor (Optional)

If you provide an Anthropic API key, WeatherOracle will periodically consult Claude to analyze:
- The NWS forecast discussion for your area
- Current model performance scores
- Recent observations

Claude returns model weight adjustment recommendations and tracks whether its advice improved or hurt accuracy over time. This is a closed-loop feedback system — bad advice gets flagged and Claude adjusts its strategy.

**Cost**: Typically <$1/month at default 4-hour intervals using Claude Sonnet.

## HRRR Note (Non-US Users)

HRRR data is only available for the contiguous United States. If your location is outside the US, HRRR will return no data and the ensemble will work with the remaining 6 models. You can optionally remove the `hrrr` entry from `WEATHER_MODELS` in `core/config.py` to skip the API call entirely.

## Project Structure

```
WeatherOracle/
├── main.py                 # Entry point
├── core/
│   ├── config.py           # Configuration, locations, model definitions
│   ├── database.py         # SQLite storage with thread safety + batch mode
│   └── orchestrator.py     # Main engine: collect → verify → predict → publish
├── collectors/
│   ├── tempest.py          # WeatherFlow Tempest API
│   ├── openmeteo.py        # Open-Meteo multi-model forecasts
│   ├── metar.py            # Airport METAR observations
│   ├── nws.py              # NWS alerts and forecast discussion
│   ├── homeassistant.py    # HA sensor reader (cross-check)
│   └── ha_publisher.py     # Push forecasts to HA as sensors
├── ml/
│   ├── engine_v2.py        # ML ensemble, scoreboard, persistence forecaster
│   ├── claude_advisor.py   # Claude API weather advisor with feedback loop
│   └── deep_backfill.py    # Historical data backfill pipeline
├── gui/
│   └── app.py              # Tkinter desktop GUI
├── ha_card/
│   └── weatheroracle-card.js  # Custom Lovelace card for HA
├── config.example.json     # Example configuration
├── .env.example            # Example environment variables
└── requirements.txt
```

## Troubleshooting

**"HRRR: FAILED"** — You're outside the US, or Open-Meteo is temporarily down. The ensemble works fine without it.

**"Obs validation: pressure_mb=34737 rejected"** — This was a bug in METAR pressure parsing (now fixed). If you see this, you're running an older version.

**ML shows "insufficient_data"** — Normal during the first 48 hours. Run a deep backfill or wait for enough live obs-forecast pairs to accumulate (need 48+ per bucket).

**"overfit_rejected"** — The ML detected that cross-validation error was >10x worse than training error. This is the system protecting itself from memorizing small datasets. It'll resolve as more data accumulates.

## License

MIT — see [LICENSE](LICENSE)

## Credits

- Weather model data via [Open-Meteo](https://open-meteo.com/) (free, no API key required)
- Station observations via [WeatherFlow Tempest API](https://weatherflow.github.io/Tempest/api/)
- Airport observations via [aviationweather.gov](https://aviationweather.gov/)
- NWS forecasts and alerts via [api.weather.gov](https://api.weather.gov/)
- AI advisor powered by [Anthropic Claude](https://anthropic.com/)
