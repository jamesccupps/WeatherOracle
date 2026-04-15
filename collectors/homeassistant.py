"""Home Assistant REST API collector for cross-check sensors."""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger("WeatherOracle.ha")


class HACollector:
    """Pull sensor readings and history from Home Assistant."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def test_connection(self) -> tuple:
        """Test HA connection. Returns (success, message)."""
        try:
            r = self.session.get(f"{self.url}/api/", timeout=5)
            if r.status_code == 200:
                return True, "Connected to Home Assistant"
            elif r.status_code == 401:
                return False, "Invalid token"
            return False, f"HTTP {r.status_code}"
        except requests.ConnectionError:
            return False, f"Cannot reach {self.url}"
        except Exception as e:
            return False, str(e)

    def get_sensor(self, entity_id: str) -> Optional[dict]:
        """Read a single sensor's current state."""
        try:
            r = self.session.get(
                f"{self.url}/api/states/{entity_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                state = data.get("state", "")
                if state in ("unavailable", "unknown", ""):
                    return None
                return {
                    "state": state,
                    "unit": data.get("attributes", {}).get("unit_of_measurement"),
                    "friendly_name": data.get("attributes", {}).get("friendly_name"),
                    "last_updated": data.get("last_updated"),
                }
            return None
        except Exception as e:
            log.error("HA sensor %s: %s", entity_id, e)
            return None

    def get_sensor_float(self, entity_id: str) -> Optional[float]:
        """Read a sensor and return its value as float."""
        data = self.get_sensor(entity_id)
        if data:
            try:
                return float(data["state"])
            except (ValueError, TypeError):
                pass
        return None

    def get_history(self, entity_id: str, hours_back: int = 48) -> list:
        """Pull HA history for a sensor."""
        try:
            start = (datetime.now(timezone.utc) - timedelta(hours=hours_back)
                     ).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = self.session.get(
                f"{self.url}/api/history/period/{start}",
                params={"filter_entity_id": entity_id,
                        "minimal_response": "true"},
                timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data and len(data) > 0:
                    return data[0]
            return []
        except Exception as e:
            log.error("HA history %s: %s", entity_id, e)
            return []

    def get_long_history(self, entity_id: str, days_back: int = 365,
                         chunk_days: int = 7) -> list:
        """Pull extended HA history in chunks. Returns list of
        {state, last_changed} dicts for the entity.

        HA's history API can be slow for large ranges so we chunk it.
        Typical recorder keeps 10 days by default but many configs keep more.
        """
        all_records = []
        end = datetime.now(timezone.utc)
        chunk_start = end - timedelta(days=days_back)

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            start_str = chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")

            try:
                r = self.session.get(
                    f"{self.url}/api/history/period/{start_str}",
                    params={
                        "filter_entity_id": entity_id,
                        "end_time": end_str,
                        "minimal_response": "true",
                        "significant_changes_only": "false",
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    if data and len(data) > 0:
                        for entry in data[0]:
                            state = entry.get("state", "")
                            if state in ("unavailable", "unknown", ""):
                                continue
                            try:
                                float(state)
                                all_records.append({
                                    "state": float(state),
                                    "last_changed": entry.get("last_changed", ""),
                                })
                            except (ValueError, TypeError):
                                pass
                elif r.status_code == 404:
                    break  # No data that far back
            except Exception as e:
                log.warning("HA long history %s chunk %s: %s",
                            entity_id, start_str[:10], e)

            chunk_start = chunk_end
            time.sleep(0.2)  # Don't hammer HA

        log.info("HA history %s: %d records over %d days",
                 entity_id, len(all_records), days_back)
        return all_records

    def get_outdoor_crosscheck(self, config: dict) -> dict:
        """Read all outdoor cross-check sensors and return summary."""
        result = {}
        sensor_pairs = [
            ("yolink", config.get("yolink_temp_entity"),
             config.get("yolink_humidity_entity")),
            ("outdoor_sensor", config.get("outdoor_temp_entity"),
             config.get("outdoor_humidity_entity")),
            ("ha_average", config.get("avg_outdoor_temp_entity"),
             config.get("avg_outdoor_humidity_entity")),
        ]
        for name, temp_eid, hum_eid in sensor_pairs:
            if not temp_eid:
                continue
            temp = self.get_sensor_float(temp_eid)
            hum = self.get_sensor_float(hum_eid) if hum_eid else None
            if temp is not None:
                result[name] = {"temp_f": temp, "humidity": hum}
        return result
