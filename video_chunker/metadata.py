"""EXIF / embedded metadata extraction from video files.

Tier 1: ffprobe (always available)
Tier 2: exiftool (auto-detected, fills gaps)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from .utils import _run

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class GPSCoordinate:
    latitude: float
    longitude: float
    altitude: float | None = None

    def is_valid(self) -> bool:
        """Reject (0,0) and obviously bogus coordinates."""
        return not (self.latitude == 0.0 and self.longitude == 0.0)


@dataclass
class LocationInfo:
    """Human-readable location derived from reverse-geocoding."""
    name: str           # e.g. "Shoreditch, London, UK"
    source: str         # "offline" | "online" | "user"
    admin1: str = ""    # state/region
    country: str = ""   # country name
    country_code: str = ""


@dataclass
class VideoMetadata:
    # Timestamps
    recorded_at: datetime | None = None     # local time (when offset is known)
    recorded_at_utc: datetime | None = None  # always UTC

    # GPS
    gps: GPSCoordinate | None = None
    location: LocationInfo | None = None

    # Device
    make: str = ""
    model: str = ""
    lens: str = ""
    software: str = ""

    # Provenance
    sources: list[str] = field(default_factory=list)
    """Which tools contributed data: ["ffprobe"], ["ffprobe", "exiftool"], etc."""

    # Raw tags for debugging / sidecar
    raw_tags: dict = field(default_factory=dict)

    def has_any(self) -> bool:
        """True if any metadata was extracted beyond defaults."""
        return bool(
            self.recorded_at_utc
            or self.gps
            or self.make
            or self.model
            or self.lens
        )

    def prompt_context(self, *, redact_gps: bool = False) -> str:
        """Build a human-readable context string for the LLM labeler.

        Returns empty string if no metadata is available.
        """
        parts: list[str] = []

        # Camera
        camera = self.model or self.make
        if camera:
            lens_part = f" + {self.lens}" if self.lens else ""
            parts.append(f"Shot on {camera}{lens_part}")

        # Date/time
        if self.recorded_at:
            parts.append(f"Recorded {self.recorded_at.strftime('%Y-%m-%d at %H:%M')}")
        elif self.recorded_at_utc:
            parts.append(f"Recorded {self.recorded_at_utc.strftime('%Y-%m-%d UTC')}")

        # Location — only send derived place name to LLM, NEVER raw coordinates
        if self.location and self.location.name:
            parts.append(f"Location: {self.location.name}")
        # Privacy-first: if no reverse-geocoded place name, omit location entirely.
        # Raw GPS coordinates are never sent to the LLM regardless of redact_gps.

        return " | ".join(parts) if parts else ""

    def to_dict(self, *, redact_gps: bool = False) -> dict:
        """Serialize to JSON-safe dict for manifest/sidecar."""
        d: dict = {}

        if self.recorded_at:
            d["recorded_at"] = self.recorded_at.isoformat()
        if self.recorded_at_utc:
            d["recorded_at_utc"] = self.recorded_at_utc.isoformat()

        if self.gps and not redact_gps:
            d["gps"] = {
                "latitude": self.gps.latitude,
                "longitude": self.gps.longitude,
            }
            if self.gps.altitude is not None:
                d["gps"]["altitude"] = self.gps.altitude

        if self.location:
            d["location"] = {
                "name": self.location.name,
                "source": self.location.source,
            }
            if self.location.admin1:
                d["location"]["admin1"] = self.location.admin1
            if self.location.country:
                d["location"]["country"] = self.location.country
            if self.location.country_code:
                d["location"]["country_code"] = self.location.country_code

        if self.make:
            d["make"] = self.make
        if self.model:
            d["model"] = self.model
        if self.lens:
            d["lens"] = self.lens
        if self.software:
            d["software"] = self.software

        if self.sources:
            d["sources"] = self.sources

        return d


# ── ISO 6709 parser ───────────────────────────────────────────────────────────


def parse_iso6709(value: str) -> GPSCoordinate | None:
    """Parse an ISO 6709 coordinate string from ffprobe.

    Examples (iPhone QuickTime):
      "+51.5074-000.1276+015.200/"
      "+51.5074-000.1276/"
      "51.5074,-0.1276,15.2"

    Returns None if parsing fails.
    """
    if not value or not value.strip():
        return None

    value = value.strip()

    try:
        # Try comma-separated format first (some cameras)
        if "," in value:
            parts = [p.strip() for p in value.split(",")]
            lat = float(parts[0])
            lon = float(parts[1])
            alt = float(parts[2]) if len(parts) >= 3 and parts[2] else None
            return GPSCoordinate(latitude=lat, longitude=lon, altitude=alt)

        # ISO 6709 compact format: +DD.DDDD±DDD.DDDD±HHHH.HHH/
        # Strip trailing slash
        value = value.rstrip("/")

        # Match: sign, digits.digits, sign, digits.digits, optional sign+digits
        m = re.match(
            r"^([+-]\d{1,3}(?:\.\d+)?)([+-]\d{1,4}(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?$",
            value,
        )
        if m:
            lat = float(m.group(1))
            lon = float(m.group(2))
            alt = float(m.group(3)) if m.group(3) else None
            return GPSCoordinate(latitude=lat, longitude=lon, altitude=alt)

    except (ValueError, IndexError):
        pass

    logger.debug("Could not parse ISO 6709 value: %r", value)
    return None


# ── ffprobe extraction (Tier 1) ────────────────────────────────────────────────


# ffprobe format tags where GPS data may live (vendor-specific)
_GPS_TAG_KEYS = (
    "com.apple.quicktime.location.ISO6709",
    "location",  # Android / generic
    "com.apple.quicktime.location.altitude",
    "xyz",  # Some 360 cameras
)

_CREATION_TAG_KEYS = (
    "com.apple.quicktime.creationdate",  # Has UTC offset — prefer this
    "creation_time",                      # UTC, generic
    "date",                               # Some cameras
)

_DEVICE_TAG_KEYS = {
    "make": ("com.apple.quicktime.make", "make"),
    "model": ("com.apple.quicktime.model", "model"),
    "lens": ("com.apple.quicktime.lens", "lens", "com.apple.quicktime.camera.lens_model"),
    "software": ("com.apple.quicktime.software", "software", "encoder"),
}


def _parse_creation_date(value: str) -> tuple[datetime | None, datetime | None]:
    """Parse a creation date string, returning (local_time, utc_time).

    Returns (None, None) for both if parsing fails.
    Apple QuickTime creationdate includes timezone offset.
    Generic creation_time is assumed UTC.
    """
    if not value:
        return None, None

    # Try ISO 8601 with timezone offset (e.g. "2024-06-15T18:42:30+0100")
    dt_local: datetime | None = None
    dt_utc: datetime | None = None

    # Common formats to try
    formats = [
        # ISO with numeric offset
        ("%Y-%m-%dT%H:%M:%S%z", True),  # has tz
        ("%Y-%m-%dT%H:%M:%S.%f%z", True),
        # ISO without offset (assume UTC)
        ("%Y-%m-%dT%H:%M:%SZ", False),
        ("%Y-%m-%dT%H:%M:%S", False),
        # Compact
        ("%Y%m%dT%H%M%S%z", True),
        # Date only
        ("%Y-%m-%d", False),
        ("%Y:%m:%d %H:%M:%S", False),  # EXIF-style
        ("%Y:%m:%d %H:%M:%S%z", True),
    ]

    for fmt, has_tz in formats:
        try:
            dt = datetime.strptime(value.strip(), fmt)
            if has_tz and dt.tzinfo is not None:
                dt_local = dt
                dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
            else:
                dt_utc = dt
            break
        except ValueError:
            continue

    return dt_local, dt_utc


def extract_metadata(video_path: Path) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe (Tier 1).

    Reads format-level tags for GPS, creation time, and device info.
    This is always available (ffprobe is already required).

    Returns VideoMetadata with whatever was found. Caller checks .has_any().
    """
    metadata = VideoMetadata(sources=["ffprobe"])

    try:
        result = _run([
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ])
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        tags = fmt.get("tags", {})

        if not tags:
            logger.debug("No format tags found in %s", video_path.name)
            return metadata

        metadata.raw_tags = dict(tags)

        # ── Creation time ──────────────────────────────────────────────────
        for key in _CREATION_TAG_KEYS:
            if key in tags and tags[key]:
                dt_local, dt_utc = _parse_creation_date(tags[key])
                if dt_utc:
                    metadata.recorded_at_utc = dt_utc
                if dt_local:
                    metadata.recorded_at = dt_local
                if dt_utc or dt_local:
                    logger.debug("  creation date from %s: %s", key, tags[key])
                    break

        # ── GPS ────────────────────────────────────────────────────────────
        # Primary: ISO 6709 tag (iPhone, most QuickTime cameras)
        iso6709_key = None
        for key in _GPS_TAG_KEYS:
            if key in tags and tags[key]:
                iso6709_key = key
                break

        if iso6709_key:
            gps = parse_iso6709(tags[iso6709_key])
            if gps and gps.is_valid():
                metadata.gps = gps
                logger.debug("  GPS from %s: %.5f, %.5f", iso6709_key,
                             gps.latitude, gps.longitude)

        # Altitude may be a separate tag on some cameras
        alt_key = "com.apple.quicktime.location.altitude"
        if alt_key in tags and tags[alt_key] and metadata.gps:
            try:
                metadata.gps.altitude = float(tags[alt_key])
            except ValueError:
                pass

        # ── Device info ────────────────────────────────────────────────────
        for field, keys in _DEVICE_TAG_KEYS.items():
            for key in keys:
                if key in tags and tags[key]:
                    setattr(metadata, field, tags[key])
                    break

        if metadata.has_any():
            logger.info("Metadata: %s %s %s",
                        metadata.model or metadata.make or "unknown camera",
                        metadata.recorded_at or metadata.recorded_at_utc or "",
                        metadata.gps or "")

    except Exception as e:
        logger.warning("Metadata extraction failed for %s: %s", video_path.name, e)
        metadata.sources = []

    return metadata


# ── Quick one-liner for CLI display ────────────────────────────────────────────


def metadata_summary(metadata: VideoMetadata) -> str:
    """One-line summary for the CLI progress output."""
    if not metadata.has_any():
        return "No embedded metadata found"

    parts: list[str] = []
    camera = metadata.model or metadata.make
    if camera:
        parts.append(camera)
    if metadata.gps and metadata.gps.is_valid():
        parts.append(f"{metadata.gps.latitude:.4f},{metadata.gps.longitude:.4f}")
    if metadata.recorded_at:
        parts.append(metadata.recorded_at.strftime("%Y-%m-%d %H:%M"))
    elif metadata.recorded_at_utc:
        parts.append(f"{metadata.recorded_at_utc.strftime('%Y-%m-%d %H:%M')} UTC")

    return " | ".join(parts) if parts else "Metadata present (no details)"


# ── Tier 2: exiftool enrichment ───────────────────────────────────────────────


import shutil


def _is_exiftool_available() -> bool:
    """Check if exiftool is installed and on PATH."""
    return shutil.which("exiftool") is not None


def enrich_with_exiftool(
    video_path: Path,
    metadata: VideoMetadata,
) -> VideoMetadata:
    """Enrich metadata using exiftool (Tier 2).

    Only fills gaps left by ffprobe — never overwrites ffprobe data.
    Returns the same metadata object, modified in place.

    exiftool provides:
    - Lens info (Fuji maker notes, Canon LensModel, etc.)
    - More GPS variants (GoPro, DJI, Garmin)
    - OffsetTime* for proper timezone handling
    - MakerNote camera/lens details
    """
    if not _is_exiftool_available():
        logger.debug("exiftool not found — skipping Tier 2 enrichment")
        return metadata

    try:
        result = _run([
            "exiftool",
            "-json",
            "-n",  # Numeric (no pretty-printing of values)
            str(video_path),
        ])
        data = json.loads(result.stdout)
        if not data or not isinstance(data, list):
            return metadata

        tags = data[0] if data else {}
        if not tags:
            return metadata

        if "ffprobe" not in metadata.sources:
            metadata.sources.append("ffprobe")
        if "exiftool" not in metadata.sources:
            metadata.sources.append("exiftool")

        # Merge raw tags (exiftool keys are prefixed with group:)
        for k, v in tags.items():
            # Strip group prefix for cleaner keys: "EXIF:GPSLatitude" -> "GPSLatitude"
            clean_key = k.split(":", 1)[-1] if ":" in k else k
            metadata.raw_tags.setdefault(clean_key, v)

        # ── GPS (fill gap or enrich altitude) ──────────────────────────────
        if not metadata.gps or metadata.gps.altitude is None:
            gps_lat = tags.get("GPSLatitude") or tags.get("EXIF:GPSLatitude")
            gps_lon = tags.get("GPSLongitude") or tags.get("EXIF:GPSLongitude")
            gps_alt = tags.get("GPSAltitude") or tags.get("EXIF:GPSAltitude")

            if gps_lat is not None and gps_lon is not None:
                try:
                    lat = float(gps_lat)
                    lon = float(gps_lon)
                    alt = float(gps_alt) if gps_alt is not None else None

                    if not metadata.gps:
                        coord = GPSCoordinate(latitude=lat, longitude=lon, altitude=alt)
                        if coord.is_valid():
                            metadata.gps = coord
                            logger.debug("  GPS from exiftool: %.5f, %.5f", lat, lon)
                    elif metadata.gps.altitude is None and alt is not None:
                        metadata.gps.altitude = alt
                        logger.debug("  Altitude from exiftool: %.1f", alt)
                except (ValueError, TypeError):
                    pass

        # ── Creation time (fill gap) ────────────────────────────────────────
        if not metadata.recorded_at_utc:
            # exiftool keys for creation time
            for key in ("DateTimeOriginal", "CreateDate", "TrackCreateDate",
                        "MediaCreateDate"):
                val = tags.get(key) or tags.get(f"EXIF:{key}") or tags.get(f"QuickTime:{key}")
                if val:
                    dt_local, dt_utc = _parse_creation_date(str(val))
                    if dt_utc:
                        metadata.recorded_at_utc = dt_utc
                    if dt_local:
                        metadata.recorded_at = dt_local
                    if dt_utc or dt_local:
                        logger.debug("  Creation time from exiftool %s", key)
                        break

            # Check for offset time tags to upgrade UTC to local
            if metadata.recorded_at_utc and not metadata.recorded_at:
                offset_str = (
                    tags.get("OffsetTime") or tags.get("EXIF:OffsetTime")
                    or tags.get("TimeZone") or tags.get("QuickTime:TimeZone")
                )
                if offset_str:
                    # Format: "+01:00" or "+0100"
                    offset_clean = offset_str.replace(":", "")
                    try:
                        from datetime import timedelta
                        # Parse ±HHMM
                        sign = 1 if offset_clean[0] == "+" else -1
                        hours = int(offset_clean[1:3])
                        mins = int(offset_clean[3:5])
                        offset = timedelta(hours=sign * hours, minutes=sign * mins)
                        metadata.recorded_at = metadata.recorded_at_utc.replace(
                            tzinfo=timezone.utc
                        ).astimezone(timezone(offset))
                        metadata.recorded_at = metadata.recorded_at.replace(tzinfo=None)
                        logger.debug("  Timezone offset from exiftool: %s", offset_str)
                    except (ValueError, IndexError):
                        pass

        # ── Device info (fill gaps, exiftool wins on lens) ─────────────────
        if not metadata.make:
            metadata.make = (
                tags.get("Make") or tags.get("EXIF:Make") or ""
            )
        if not metadata.model:
            metadata.model = (
                tags.get("Model") or tags.get("EXIF:Model")
                or tags.get("QuickTime:Model") or ""
            )

        # Lens — exiftool is the primary source for this
        lens = (
            tags.get("LensModel") or tags.get("EXIF:LensModel")
            or tags.get("LensInfo") or tags.get("EXIF:LensInfo")
            or tags.get("LensSpec") or tags.get("FujiFilm:LensModel")
            or tags.get("Canon:LensModel") or tags.get("Nikon:LensModel")
        )
        if lens:
            metadata.lens = str(lens)
            logger.debug("  Lens from exiftool: %s", lens)

        if not metadata.software:
            metadata.software = (
                tags.get("Software") or tags.get("QuickTime:Software") or ""
            )

        logger.info("exiftool enrichment complete: %s %s %s",
                    metadata.model or metadata.make or "unknown camera",
                    metadata.lens or "no lens",
                    metadata.gps or "no GPS")

    except Exception as e:
        logger.warning("exiftool enrichment failed for %s: %s", video_path.name, e)

    return metadata
