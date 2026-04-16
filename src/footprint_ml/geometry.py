"""Standalone polygon geometry metrics using Shapely + pyproj.

All metric calculations are performed in a metric CRS (UTM or user-supplied).
The input polygon is expected to be in WGS84 (EPSG:4326) unless a source CRS
is provided.
"""

from __future__ import annotations

import math

from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform


def _utm_crs_for_polygon(polygon: Polygon) -> CRS:
    """Return the UTM CRS whose zone contains the polygon centroid."""
    centroid = polygon.centroid
    lon, lat = centroid.x, centroid.y
    # UTM zone number: 1-based, each zone is 6° wide
    zone = int((lon + 180) / 6) % 60 + 1
    hemisphere = "north" if lat >= 0 else "south"
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": hemisphere == "south"})


def _project(polygon: Polygon, src_crs: CRS, dst_crs: CRS) -> Polygon:
    """Reproject a Shapely polygon between two pyproj CRS objects."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, polygon)


def polygon_to_metric(
    polygon: Polygon,
    src_crs: CRS | None = None,
    dst_crs: CRS | None = None,
) -> tuple[Polygon, CRS]:
    """Reproject *polygon* into a metric CRS suitable for distance calculations.

    Args:
        polygon: Input polygon (coordinates in *src_crs*).
        src_crs: CRS of the input polygon. Defaults to WGS84 (EPSG:4326).
        dst_crs: Target metric CRS. Defaults to the UTM zone of the centroid.

    Returns:
        (projected_polygon, dst_crs) — the reprojected polygon and the CRS used.
    """
    if src_crs is None:
        src_crs = CRS.from_epsg(4326)
    if dst_crs is None:
        dst_crs = _utm_crs_for_polygon(polygon)
    projected = _project(polygon, src_crs, dst_crs)
    return projected, dst_crs


def building_area_m2(polygon: Polygon) -> float:
    """Area of *polygon* in square metres (metric projection)."""
    metric, _ = polygon_to_metric(polygon)
    return float(metric.area)


def building_perimeter_m(polygon: Polygon) -> float:
    """Perimeter of *polygon* in metres (metric projection)."""
    metric, _ = polygon_to_metric(polygon)
    return float(metric.length)


def building_compactness(polygon: Polygon) -> float:
    """Polsby-Popper compactness score: 4π·area / perimeter².

    Returns a value in (0, 1] where 1.0 is a perfect circle.
    Calculation is done in a metric CRS so units are consistent.
    """
    metric, _ = polygon_to_metric(polygon)
    area = metric.area
    perim = metric.length
    if perim == 0:
        return 0.0
    return float(4 * math.pi * area / (perim**2))


def _min_rotated_rect_dims(polygon: Polygon) -> tuple[float, float]:
    """Return (length, width) of the minimum rotated bounding rectangle in metres.

    Length >= width.  The rectangle is computed in a metric CRS.
    """
    metric, _ = polygon_to_metric(polygon)
    rect = metric.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    # The minimum_rotated_rectangle exterior has 5 coords (closed ring)
    # Compute the two distinct edge lengths
    dx0 = coords[1][0] - coords[0][0]
    dy0 = coords[1][1] - coords[0][1]
    dx1 = coords[2][0] - coords[1][0]
    dy1 = coords[2][1] - coords[1][1]
    side_a = math.hypot(dx0, dy0)
    side_b = math.hypot(dx1, dy1)
    length = max(side_a, side_b)
    width = min(side_a, side_b)
    return length, width


def aspect_ratio(polygon: Polygon) -> float:
    """Aspect ratio of the minimum rotated bounding rectangle: length / width.

    Returns 1.0 for a square, higher values for elongated buildings.
    Returns 1.0 if width is zero (degenerate polygon).
    """
    length, width = _min_rotated_rect_dims(polygon)
    if width == 0:
        return 1.0
    return float(length / width)


def bbox_length_m(polygon: Polygon) -> float:
    """Longer side of the minimum rotated bounding rectangle in metres."""
    length, _ = _min_rotated_rect_dims(polygon)
    return float(length)


def bbox_width_m(polygon: Polygon) -> float:
    """Shorter side of the minimum rotated bounding rectangle in metres."""
    _, width = _min_rotated_rect_dims(polygon)
    return float(width)


def edge_count(polygon: Polygon) -> int:
    """Number of exterior edges (vertices) of the polygon.

    For a rectangular building this is 4.
    """
    coords = list(polygon.exterior.coords)
    # Shapely closes the ring: last coord == first coord
    return len(coords) - 1


def elongation(polygon: Polygon) -> float:
    """Elongation: 1 - (width / length) of the minimum rotated bounding rectangle.

    Returns 0.0 for a square, approaching 1.0 for a very thin building.
    Returns 0.0 for degenerate polygons.
    """
    length, width = _min_rotated_rect_dims(polygon)
    if length == 0:
        return 0.0
    return float(1.0 - width / length)


def compute_metrics(
    polygon: Polygon,
    src_crs: CRS | None = None,
) -> dict[str, float]:
    """Compute all geometry metrics for *polygon* in one pass.

    Reprojects once to the UTM zone of the centroid (or *src_crs* if given),
    then derives all metrics from the projected geometry.

    Args:
        polygon: Input polygon in WGS84 (or *src_crs*).
        src_crs: CRS of the input polygon. Defaults to WGS84 (EPSG:4326).

    Returns:
        dict with keys: building_area_m2, building_perimeter_m,
        building_compactness, aspect_ratio, bbox_length_m, bbox_width_m,
        edge_count, log_area, elongation.
    """
    if src_crs is None:
        src_crs = CRS.from_epsg(4326)

    metric, _ = polygon_to_metric(polygon, src_crs=src_crs)

    area = float(metric.area)
    perim = float(metric.length)
    compactness = (4 * math.pi * area / perim**2) if perim > 0 else 0.0

    rect = metric.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    dx0 = coords[1][0] - coords[0][0]
    dy0 = coords[1][1] - coords[0][1]
    dx1 = coords[2][0] - coords[1][0]
    dy1 = coords[2][1] - coords[1][1]
    side_a = math.hypot(dx0, dy0)
    side_b = math.hypot(dx1, dy1)
    length = max(side_a, side_b)
    width = min(side_a, side_b)

    ar = (length / width) if width > 0 else 1.0
    elong = (1.0 - width / length) if length > 0 else 0.0
    log_area = math.log(area) if area > 0 else 0.0
    n_edges = len(list(polygon.exterior.coords)) - 1

    return {
        "building_area_m2": area,
        "building_perimeter_m": perim,
        "building_compactness": compactness,
        "aspect_ratio": ar,
        "bbox_length_m": length,
        "bbox_width_m": width,
        "edge_count": float(n_edges),
        "log_area": log_area,
        "elongation": elong,
    }
