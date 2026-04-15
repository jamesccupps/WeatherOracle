"""METAR collector for Portland Jetport (KPWM).

Pulls real observed weather conditions from the airport ~10 miles from both
WeatherOracle locations. Provides ground truth for:
  - Precipitation type (rain, snow, drizzle, freezing rain)
  - Sky cover (clear, few, scattered, broken, overcast)
  - Visibility (for fog/mist detection)
  - Observed temperature and dewpoint (independent verification)

Uses the free NOAA aviationweather.gov API (no key needed).
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("WeatherOracle.metar")

# METAR weather codes → our simplified categories
WX_CATEGORIES = {
    # Precipitation types
    "RA": "rain", "DZ": "drizzle", "SN": "snow", "SG": "snow_grains",
    "PL": "ice_pellets", "GR": "hail", "GS": "small_hail",
    "FZRA": "freezing_rain", "FZDZ": "freezing_drizzle",
    "TSRA": "thunderstorm_rain", "TSSN": "thunderstorm_snow",
    "TS": "thunderstorm", "SH": "showers",
    # Obscuration
    "FG": "fog", "BR": "mist", "HZ": "haze", "FU": "smoke",
    # Other
    "UP": "unknown_precip",
}

# Intensity prefixes in METAR
INTENSITY = {"-": "light", "+": "heavy", "": "moderate"}

# Sky cover codes
SKY_COVER = {
    "CLR": "clear", "SKC": "clear", "FEW": "few",
    "SCT": "scattered", "BKN": "broken", "OVC": "overcast",
    "VV": "obscured",
}

# Map our categories to WMO weather codes for comparison with model forecasts
CATEGORY_TO_WMO = {
    "clear": 0, "few": 1, "scattered": 2, "broken": 3, "overcast": 3,
    "fog": 45, "mist": 45, "haze": 48,
    "drizzle": 51, "light_drizzle": 51, "heavy_drizzle": 55,
    "rain": 61, "light_rain": 61, "heavy_rain": 65,
    "freezing_drizzle": 66, "freezing_rain": 67,
    "snow": 71, "light_snow": 71, "heavy_snow": 75,
    "showers": 80, "thunderstorm": 95, "thunderstorm_rain": 95,
}


class METARCollector:
    """Pulls and parses METAR observations from any airport station."""

    API_URL = "https://aviationweather.gov/api/data/metar"

    def __init__(self, station: str = ""):
        self.station = station.upper() if station else ""
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "WeatherOracle/2.0"
        self.last_metar = None
        self.last_parsed = None

    def get_current(self, station: str = None) -> Optional[dict]:
        """Fetch and parse the latest METAR for a station."""
        station = (station or self.station or "").upper()
        if not station:
            return None
        try:
            r = self.session.get(
                self.API_URL,
                params={"ids": station, "format": "json"},
                timeout=10,
            )
            if r.status_code != 200:
                log.warning("METAR API HTTP %d", r.status_code)
                return None

            data = r.json()
            if not data:
                return None

            metar = data[0] if isinstance(data, list) else data
            self.last_metar = metar
            parsed = self._parse(metar)
            self.last_parsed = parsed
            return parsed

        except Exception as e:
            log.error("METAR fetch error: %s", e)
            return None

    def get_recent(self, hours: int = 6) -> list:
        """Fetch recent METARs for verification window."""
        try:
            r = self.session.get(
                self.API_URL,
                params={
                    "ids": self.station,
                    "format": "json",
                    "hours": hours,
                },
                timeout=15,
            )
            if r.status_code != 200:
                return []

            data = r.json()
            if not data:
                return []

            results = []
            for m in (data if isinstance(data, list) else [data]):
                parsed = self._parse(m)
                if parsed:
                    results.append(parsed)

            return results

        except Exception as e:
            log.error("METAR recent fetch error: %s", e)
            return []

    def _parse(self, metar: dict) -> Optional[dict]:
        """Parse aviationweather.gov JSON METAR into our format."""
        try:
            raw = metar.get("rawOb", "")

            # Temperature (already in the JSON in some formats)
            temp_c = metar.get("temp")
            dewp_c = metar.get("dewp")
            temp_f = round(temp_c * 9/5 + 32, 1) if temp_c is not None else None
            dewp_f = round(dewp_c * 9/5 + 32, 1) if dewp_c is not None else None

            # Wind
            wind_kts = metar.get("wspd")
            gust_kts = metar.get("wgst")
            wind_mph = round(wind_kts * 1.151, 1) if wind_kts is not None else None
            gust_mph = round(gust_kts * 1.151, 1) if gust_kts is not None else None
            wind_dir = metar.get("wdir")

            # Visibility (statute miles)
            visib = metar.get("visib")

            # Altimeter / pressure
            # aviationweather.gov JSON API returns altim in hectopascals (hPa/mb)
            # NOT inches of mercury — no conversion needed
            altim = metar.get("altim")
            if altim is not None:
                # Sanity check: if value is in inHg range (27-32), convert
                # If already in hPa/mb range (870-1084), use directly
                if altim < 50:
                    pressure_mb = round(altim * 33.8639, 1)
                else:
                    pressure_mb = round(altim, 1)
            else:
                pressure_mb = None

            # Parse weather phenomena from raw string
            wx_list = self._parse_wx(raw)

            # Parse sky cover from raw string
            sky_cover, cloud_base = self._parse_sky(raw)

            # Determine primary condition
            primary_condition = self._determine_condition(wx_list, sky_cover, visib)

            # Map to WMO code
            wmo_code = CATEGORY_TO_WMO.get(primary_condition, 2)

            # Precipitation flag
            has_precip = any(w["category"] in (
                "rain", "drizzle", "snow", "freezing_rain", "freezing_drizzle",
                "ice_pellets", "hail", "showers", "thunderstorm_rain",
                "thunderstorm_snow", "thunderstorm"
            ) for w in wx_list)

            # Observation time
            obs_time = metar.get("reportTime") or metar.get("obsTime", "")

            result = {
                "station": self.station,
                "raw": raw,
                "timestamp": obs_time,
                "temp_f": temp_f,
                "dewpoint_f": dewp_f,
                "humidity": self._calc_rh(temp_c, dewp_c) if temp_c and dewp_c else None,
                "wind_mph": wind_mph,
                "wind_gust_mph": gust_mph,
                "wind_dir": wind_dir,
                "pressure_mb": pressure_mb,
                "visibility_miles": visib,
                "sky_cover": sky_cover,
                "cloud_base_ft": cloud_base,
                "weather": wx_list,
                "primary_condition": primary_condition,
                "wmo_code": wmo_code,
                "has_precip": has_precip,
                "is_fog": any(w["category"] in ("fog", "mist") for w in wx_list),
                "precip_type": next(
                    (w["category"] for w in wx_list if w["category"] in
                     ("rain", "drizzle", "snow", "freezing_rain",
                      "ice_pellets", "hail")),
                    None
                ),
            }

            return result

        except Exception as e:
            log.error("METAR parse error: %s (raw: %s)",
                      e, metar.get("rawOb", "?")[:80])
            return None

    def _parse_wx(self, raw: str) -> list:
        """Extract weather phenomena from raw METAR string."""
        wx_list = []
        # Weather groups appear between wind and sky condition
        # Pattern: optional intensity + descriptor + phenomenon
        wx_pattern = re.compile(
            r'(?:^|\s)([+-]?)(MI|PR|BC|DR|BL|SH|TS|FZ)?'
            r'(DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)'
            r'(?=\s|$)'
        )

        for match in wx_pattern.finditer(raw):
            intensity_code = match.group(1)
            descriptor = match.group(2) or ""
            phenomenon = match.group(3)

            # Build full code for lookup
            full_code = f"{descriptor}{phenomenon}"
            if full_code in WX_CATEGORIES:
                category = WX_CATEGORIES[full_code]
            elif phenomenon in WX_CATEGORIES:
                category = WX_CATEGORIES[phenomenon]
            else:
                category = phenomenon.lower()

            intensity = INTENSITY.get(intensity_code, "moderate")

            # Prefix intensity for important distinctions
            if intensity != "moderate" and category in ("rain", "snow", "drizzle"):
                display_category = f"{intensity}_{category}"
            else:
                display_category = category

            wx_list.append({
                "code": match.group(0).strip(),
                "category": display_category if display_category in CATEGORY_TO_WMO else category,
                "intensity": intensity,
                "raw_category": category,
            })

        return wx_list

    def _parse_sky(self, raw: str) -> tuple:
        """Extract sky condition. Returns (cover_str, lowest_base_ft)."""
        sky_pattern = re.compile(r'(CLR|SKC|FEW|SCT|BKN|OVC|VV)(\d{3})?')
        covers = []
        lowest_base = None

        for match in sky_pattern.finditer(raw):
            code = match.group(1)
            base = int(match.group(2)) * 100 if match.group(2) else None
            cover_name = SKY_COVER.get(code, code.lower())
            covers.append(cover_name)
            if base is not None and (lowest_base is None or base < lowest_base):
                lowest_base = base

        # Return the most significant (most covered) layer
        priority = ["overcast", "broken", "obscured", "scattered", "few", "clear"]
        for p in priority:
            if p in covers:
                return p, lowest_base

        return covers[0] if covers else "unknown", lowest_base

    def _determine_condition(self, wx_list: list, sky_cover: str,
                              visibility: float = None) -> str:
        """Determine the primary weather condition."""
        # Precipitation takes priority
        for w in wx_list:
            cat = w.get("raw_category", "")
            if cat in ("rain", "snow", "drizzle", "freezing_rain",
                       "freezing_drizzle", "ice_pellets", "hail",
                       "thunderstorm", "showers"):
                intensity = w.get("intensity", "moderate")
                if intensity != "moderate":
                    return f"{intensity}_{cat}"
                return cat

        # Then obscuration
        for w in wx_list:
            if w.get("raw_category") in ("fog", "mist", "haze"):
                return w["raw_category"]

        # Fall back to sky cover
        return sky_cover

    @staticmethod
    def _calc_rh(temp_c: float, dewp_c: float) -> float:
        """Calculate relative humidity from temp and dewpoint (Magnus formula)."""
        a, b = 17.625, 243.04
        rh = 100 * (
            pow(2.71828, (a * dewp_c) / (b + dewp_c)) /
            pow(2.71828, (a * temp_c) / (b + temp_c))
        )
        return round(min(100, max(0, rh)), 1)
