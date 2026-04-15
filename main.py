#!/usr/bin/env python3
"""WeatherOracle v2.0 — Hyperlocal ML Weather Prediction

Fuses Tempest stations, HA sensors, 6 weather models, and NWS data
to produce ML-tuned forecasts for two southern Maine locations.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import APP_DIR
from gui import WeatherOracleGUI


def setup_logging():
    log = logging.getLogger("WeatherOracle")
    log.setLevel(logging.DEBUG)

    # File handler — rotates at 5MB, keeps 3 backups
    log_path = APP_DIR / "data" / "weather_oracle.log"
    log_path.parent.mkdir(exist_ok=True)
    fh = RotatingFileHandler(
        str(log_path), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(ch)

    return log


def main():
    log = setup_logging()
    log.info("WeatherOracle v2.0 starting")

    try:
        app = WeatherOracleGUI()
        app.run()
    except Exception as e:
        log.critical("Fatal error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
