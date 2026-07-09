"""Reverse-geocoding for GPS coordinates — offline and online backends.

Offline: uses the `reverse_geocoder` package (bundled dataset, city-level).
Online: uses Nominatim (OpenStreetMap) via stdlib urllib.

Both backends share a JSON file cache to avoid repeat lookups.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .metadata import GPSCoordinate, LocationInfo

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "video-chunker" / "geocode.json"

# Nominatim usage policy: max 1 req/s, valid User-Agent
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_TIMEOUT = 10  # seconds
_NOMINATIM_USER_AGENT = "video-chunker/0.1 (https://github.com/min-hsao/video-chunker)"


# ── Cache ─────────────────────────────────────────────────────────────────────


class GeocodeCache:
    """JSON file cache for reverse-geocode results.

    Keys are "lat,lon" rounded to 3 decimal places (~110m precision).
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self._cache: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Geocode cache read failed: %s — starting fresh", e)
                self._cache = {}
        else:
            self._cache = {}

        return self._cache

    def _save(self) -> None:
        if self._cache is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Geocode cache write failed: %s", e)

    @staticmethod
    def _make_key(gps: GPSCoordinate) -> str:
        """Round to 3 decimals (~110m) for cache key."""
        return f"{round(gps.latitude, 3)},{round(gps.longitude, 3)}"

    def get(self, gps: GPSCoordinate) -> dict[str, Any] | None:
        key = self._make_key(gps)
        return self._load().get(key)

    def put(self, gps: GPSCoordinate, data: dict[str, Any]) -> None:
        cache = self._load()
        cache[self._make_key(gps)] = data
        self._save()


# ── Offline backend (reverse_geocoder) ────────────────────────────────────────


def _geocode_offline(gps: GPSCoordinate) -> LocationInfo | None:
    """Reverse-geocode using the reverse_geocoder package.

    Requires: pip install reverse_geocoder (included in [geo] extra).
    Uses a bundled cities dataset — no network needed.
    """
    try:
        import reverse_geocoder as rg
    except ImportError:
        logger.warning(
            "reverse_geocoder not installed. Install with: pip install 'video-chunker[geo]'"
        )
        return None

    try:
        results = rg.search((gps.latitude, gps.longitude), verbose=False)
        if not results:
            return None

        r = results[0]
        name_parts = []
        if r.get("name"):
            name_parts.append(r["name"])
        if r.get("admin1") and r["admin1"] != r.get("name"):
            name_parts.append(r["admin1"])
        if r.get("cc"):
            name_parts.append(r["cc"])

        name = ", ".join(name_parts) if name_parts else f"{gps.latitude:.4f}, {gps.longitude:.4f}"

        return LocationInfo(
            name=name,
            source="offline",
            admin1=r.get("admin1", ""),
            country=r.get("cc", ""),
            country_code=r.get("cc", ""),
        )
    except Exception as e:
        logger.warning("Offline geocode failed: %s", e)
        return None


# ── Online backend (Nominatim) ────────────────────────────────────────────────


def _geocode_online(gps: GPSCoordinate) -> LocationInfo | None:
    """Reverse-geocode using Nominatim (OpenStreetMap).

    Respects usage policy: real User-Agent, 1 req/s rate limit.
    """
    params = urllib.parse.urlencode({
        "format": "jsonv2",
        "lat": gps.latitude,
        "lon": gps.longitude,
        "zoom": 14,       # neighborhood level
        "addressdetails": 1,
    })
    url = f"{_NOMINATIM_URL}?{params}"

    req = urllib.request.Request(url, headers={
        "User-Agent": _NOMINATIM_USER_AGENT,
        "Accept": "application/json",
    })

    try:
        import time
        # Rate limit: at least 1 second between requests
        time.sleep(1.0)

        with urllib.request.urlopen(req, timeout=_NOMINATIM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        address = data.get("address", {})

        # Build a human-readable name
        name_parts = []
        # Prefer neighborhood, then suburb, then city/town/village
        for key in ("neighbourhood", "suburb", "city_district", "city", "town", "village", "hamlet"):
            val = address.get(key)
            if val:
                name_parts.append(val)
                break
        if address.get("state") and address["state"] not in name_parts:
            name_parts.append(address["state"])
        if address.get("country"):
            name_parts.append(address["country"])

        name = ", ".join(name_parts) if name_parts else data.get("display_name", "")

        return LocationInfo(
            name=name,
            source="online",
            admin1=address.get("state", ""),
            country=address.get("country", ""),
            country_code=address.get("country_code", "").upper(),
        )
    except Exception as e:
        logger.warning("Online geocode failed: %s", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────


def reverse_geocode(
    gps: GPSCoordinate,
    *,
    mode: str = "offline",
    cache: GeocodeCache | None = None,
) -> LocationInfo | None:
    """Reverse-geocode GPS coordinates to a human-readable location.

    Args:
        gps: Valid GPSCoordinate with latitude/longitude.
        mode: "offline" (reverse_geocoder, default) or "online" (Nominatim).
        cache: Optional GeocodeCache. If None, uses default path.
            Pass a cache with cache_path=None equivalent to disable.

    Returns:
        LocationInfo with place name, or None if geocoding failed.
    """
    if not gps.is_valid():
        return None

    # Use cache if available
    if cache is not None:
        cached = cache.get(gps)
        if cached:
            return LocationInfo(**cached)

    # Try the requested backend
    result: LocationInfo | None = None
    if mode == "offline":
        result = _geocode_offline(gps)
    elif mode == "online":
        result = _geocode_online(gps)
        # Fallback to offline if online fails and offline is available
        if result is None:
            result = _geocode_offline(gps)
    else:
        logger.warning("Unknown geocode mode: %s", mode)
        return None

    # Cache the result
    if result and cache is not None:
        cache.put(gps, {
            "name": result.name,
            "source": result.source,
            "admin1": result.admin1,
            "country": result.country,
            "country_code": result.country_code,
        })

    return result
