"""National Weather Service alerts and forecast discussion."""

import logging
from typing import Optional

import requests

log = logging.getLogger("WeatherOracle.nws")


class NWSCollector:
    """Pull NWS alerts and Area Forecast Discussion for GYX (Gray, ME)."""

    BASE = "https://api.weather.gov"
    HEADERS = {"User-Agent": "(WeatherOracle/2.0, github.com/weatheroracle)",
               "Accept": "application/geo+json"}

    def get_alerts(self, lat: float, lon: float) -> list:
        """Active alerts for a lat/lon point."""
        try:
            r = requests.get(
                f"{self.BASE}/alerts/active",
                params={"point": f"{lat},{lon}"},
                headers=self.HEADERS, timeout=15)
            r.raise_for_status()
            features = r.json().get("features", [])
            return [{
                "event": f["properties"]["event"],
                "headline": f["properties"].get("headline", ""),
                "description": f["properties"].get("description", "")[:500],
                "severity": f["properties"].get("severity", ""),
                "certainty": f["properties"].get("certainty", ""),
                "onset": f["properties"].get("onset", ""),
                "expires": f["properties"].get("expires", ""),
            } for f in features]
        except Exception as e:
            log.error("NWS alerts error: %s", e)
            return []

    def get_forecast_discussion(self, office: str = "GYX") -> Optional[str]:
        """Latest Area Forecast Discussion text."""
        try:
            r = requests.get(
                f"{self.BASE}/products/types/AFD/locations/{office}",
                headers=self.HEADERS, timeout=15)
            r.raise_for_status()
            products = r.json().get("@graph", [])
            if not products:
                return None
            url = products[0].get("@id", "")
            if not url:
                return None
            r2 = requests.get(url, headers=self.HEADERS, timeout=15)
            r2.raise_for_status()
            return r2.json().get("productText", "")
        except Exception as e:
            log.error("NWS AFD error: %s", e)
            return None

    def get_gridpoint_forecast(self, lat: float, lon: float) -> Optional[dict]:
        """NWS 7-day gridpoint forecast."""
        try:
            # First get gridpoint
            r = requests.get(f"{self.BASE}/points/{lat},{lon}",
                             headers=self.HEADERS, timeout=15)
            r.raise_for_status()
            props = r.json().get("properties", {})
            fc_url = props.get("forecast")
            fc_hourly_url = props.get("forecastHourly")

            result = {"office": props.get("gridId"),
                      "gridX": props.get("gridX"),
                      "gridY": props.get("gridY")}

            if fc_url:
                r2 = requests.get(fc_url, headers=self.HEADERS, timeout=15)
                if r2.status_code == 200:
                    periods = r2.json().get("properties", {}).get("periods", [])
                    result["periods"] = [{
                        "name": p["name"],
                        "temp": p["temperature"],
                        "unit": p["temperatureUnit"],
                        "wind": p["windSpeed"],
                        "wind_dir": p["windDirection"],
                        "short": p["shortForecast"],
                        "detail": p["detailedForecast"],
                    } for p in periods[:14]]

            if fc_hourly_url:
                r3 = requests.get(fc_hourly_url, headers=self.HEADERS, timeout=15)
                if r3.status_code == 200:
                    periods = r3.json().get("properties", {}).get("periods", [])
                    result["hourly"] = [{
                        "start": p["startTime"],
                        "temp": p["temperature"],
                        "wind": p["windSpeed"],
                        "wind_dir": p["windDirection"],
                        "short": p["shortForecast"],
                    } for p in periods[:72]]

            return result
        except Exception as e:
            log.error("NWS gridpoint error: %s", e)
            return None
