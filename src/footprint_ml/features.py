"""Feature extraction: extract_features(polygon, **signals) → flat dict.

Produces a flat dictionary of numeric features ready for model inference.
Geometry features are always present. Categorical signals (zone, OSM, ANZSIC)
are optional — missing values are encoded as NaN so the downstream
HistGradientBoostingClassifier can handle them natively.
"""

from __future__ import annotations

import math
from typing import Any

from pyproj import CRS
from shapely.geometry import Polygon

from footprint_ml.geometry import compute_metrics

# Sentinel for missing optional features — HGBC handles NaN natively
_MISSING = float("nan")

# Stable ordered list of all feature names (defines model input column order)
FEATURE_NAMES: list[str] = [
    # Geometry (always present)
    "building_area_m2",
    "building_perimeter_m",
    "building_compactness",
    "aspect_ratio",
    "bbox_length_m",
    "bbox_width_m",
    "edge_count",
    "log_area",
    "elongation",
    # Zoning (optional)
    "zone_code_encoded",
    # OSM (optional)
    "osm_amenity_encoded",
    "osm_building_use_encoded",
    "has_osm_shop",
    "has_osm_office",
    # ABR/ANZSIC (optional)
    "anzsic_primary_division_encoded",
    "anzsic_count",
]

# ---------------------------------------------------------------------------
# OSM tag encoding tables
# ---------------------------------------------------------------------------

# building=* values mapped to a numeric code (0 = unknown/missing)
_OSM_BUILDING_USE_MAP: dict[str, float] = {
    "warehouse": 1.0,
    "industrial": 2.0,
    "retail": 3.0,
    "office": 4.0,
    "hospital": 5.0,
    "clinic": 5.0,
    "hotel": 6.0,
    "motel": 6.0,
    "school": 7.0,
    "university": 7.0,
    "college": 7.0,
    "kindergarten": 8.0,
    "commercial": 9.0,
    "supermarket": 3.0,
    "shop": 3.0,
    "yes": 0.0,
}

# amenity=* values mapped to a numeric code (0 = unknown/missing)
_OSM_AMENITY_MAP: dict[str, float] = {
    "hospital": 1.0,
    "clinic": 1.0,
    "doctors": 1.0,
    "dentist": 1.0,
    "pharmacy": 1.0,
    "school": 2.0,
    "university": 2.0,
    "college": 2.0,
    "kindergarten": 3.0,
    "childcare": 3.0,
    "restaurant": 4.0,
    "cafe": 4.0,
    "bar": 4.0,
    "pub": 4.0,
    "fast_food": 4.0,
    "hotel": 5.0,
    "office": 6.0,
}

# ANZSIC primary division letter → numeric code
_ANZSIC_DIVISION_MAP: dict[str, float] = {
    "A": 1.0,   # Agriculture, Forestry and Fishing
    "B": 2.0,   # Mining
    "C": 3.0,   # Manufacturing
    "D": 4.0,   # Electricity, Gas, Water and Waste Services
    "E": 5.0,   # Construction
    "F": 6.0,   # Wholesale Trade
    "G": 7.0,   # Retail Trade
    "H": 8.0,   # Accommodation and Food Services
    "I": 9.0,   # Transport, Postal and Warehousing
    "J": 10.0,  # Information Media and Telecommunications
    "K": 11.0,  # Financial and Insurance Services
    "L": 12.0,  # Rental, Hiring and Real Estate Services
    "M": 13.0,  # Professional, Scientific and Technical Services
    "N": 14.0,  # Administrative and Support Services
    "O": 15.0,  # Public Administration and Safety
    "P": 16.0,  # Education and Training
    "Q": 17.0,  # Health Care and Social Assistance
    "R": 18.0,  # Arts and Recreation Services
    "S": 19.0,  # Other Services
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_zone(zone_code: str | None) -> float:
    """Encode a zone code string to a float.

    Currently a simple hash-based ordinal that is stable across runs.
    A fitted OrdinalEncoder (encoders.py) will replace this at inference time;
    this fallback is used when no fitted encoder is available.
    """
    if zone_code is None:
        return _MISSING
    return float(abs(hash(zone_code.upper().strip())) % 1000)


def _encode_osm_building(osm_tags: dict[str, Any] | None) -> float:
    """Encode OSM building=* tag to a numeric code."""
    if not osm_tags:
        return _MISSING
    val = osm_tags.get("building")
    if val is None:
        return _MISSING
    return _OSM_BUILDING_USE_MAP.get(str(val).lower(), 0.0)


def _encode_osm_amenity(osm_tags: dict[str, Any] | None) -> float:
    """Encode OSM amenity=* tag to a numeric code."""
    if not osm_tags:
        return _MISSING
    val = osm_tags.get("amenity")
    if val is None:
        return _MISSING
    return _OSM_AMENITY_MAP.get(str(val).lower(), 0.0)


def _has_osm_shop(osm_tags: dict[str, Any] | None) -> float:
    """1.0 if OSM shop=* is present and non-null, else 0.0."""
    if not osm_tags:
        return 0.0
    return 1.0 if osm_tags.get("shop") not in (None, "") else 0.0


def _has_osm_office(osm_tags: dict[str, Any] | None) -> float:
    """1.0 if OSM office=* is present and non-null, else 0.0."""
    if not osm_tags:
        return 0.0
    return 1.0 if osm_tags.get("office") not in (None, "") else 0.0


def _encode_anzsic(anzsic_divisions: list[str] | None) -> tuple[float, float]:
    """Return (primary_division_encoded, count).

    Primary division is the first entry in the list.
    Count is the total number of distinct divisions present.
    """
    if not anzsic_divisions:
        return _MISSING, 0.0
    primary = _ANZSIC_DIVISION_MAP.get(anzsic_divisions[0].upper(), 0.0)
    count = float(len(set(d.upper() for d in anzsic_divisions)))
    return primary, count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    polygon: Polygon,
    *,
    zone_code: str | None = None,
    osm_tags: dict[str, Any] | None = None,
    anzsic_divisions: list[str] | None = None,
    src_crs: CRS | None = None,
) -> dict[str, float]:
    """Extract a flat feature dict from a building footprint polygon.

    Geometry features are always computed. Optional signals are encoded when
    provided; absent signals are represented as NaN so the underlying
    HistGradientBoostingClassifier handles them without imputation.

    Args:
        polygon: Building footprint in WGS84 (or *src_crs*).
        zone_code: Zoning code string, e.g. ``"IND"`` or ``"B2"``.
        osm_tags: Dict of raw OSM key→value tags for the building/area.
            Recognised keys: ``building``, ``amenity``, ``shop``, ``office``.
        anzsic_divisions: List of ANZSIC primary division letters present at
            the address, e.g. ``["F", "I"]``.
        src_crs: CRS of *polygon*. Defaults to WGS84 (EPSG:4326).

    Returns:
        Flat ``dict[str, float]`` with keys matching :data:`FEATURE_NAMES`.
        Missing optional features are ``float('nan')``.
    """
    geom = compute_metrics(polygon, src_crs=src_crs)

    anzsic_primary, anzsic_count = _encode_anzsic(anzsic_divisions)

    return {
        # Geometry
        "building_area_m2": geom["building_area_m2"],
        "building_perimeter_m": geom["building_perimeter_m"],
        "building_compactness": geom["building_compactness"],
        "aspect_ratio": geom["aspect_ratio"],
        "bbox_length_m": geom["bbox_length_m"],
        "bbox_width_m": geom["bbox_width_m"],
        "edge_count": geom["edge_count"],
        "log_area": geom["log_area"],
        "elongation": geom["elongation"],
        # Zoning
        "zone_code_encoded": _encode_zone(zone_code),
        # OSM
        "osm_amenity_encoded": _encode_osm_amenity(osm_tags),
        "osm_building_use_encoded": _encode_osm_building(osm_tags),
        "has_osm_shop": _has_osm_shop(osm_tags),
        "has_osm_office": _has_osm_office(osm_tags),
        # ANZSIC
        "anzsic_primary_division_encoded": anzsic_primary,
        "anzsic_count": anzsic_count,
    }


def feature_vector(
    polygon: Polygon,
    *,
    zone_code: str | None = None,
    osm_tags: dict[str, Any] | None = None,
    anzsic_divisions: list[str] | None = None,
    src_crs: CRS | None = None,
) -> list[float]:
    """Return features as an ordered list matching :data:`FEATURE_NAMES`.

    Convenience wrapper around :func:`extract_features` for callers that need
    a plain list (e.g. for numpy array construction).
    """
    feats = extract_features(
        polygon,
        zone_code=zone_code,
        osm_tags=osm_tags,
        anzsic_divisions=anzsic_divisions,
        src_crs=src_crs,
    )
    return [feats[name] for name in FEATURE_NAMES]
