"""Tests for footprint_ml.geometry polygon metrics."""

from __future__ import annotations

import math

from pyproj import CRS
from shapely.geometry import Polygon

from footprint_ml.geometry import (
    aspect_ratio,
    bbox_length_m,
    bbox_width_m,
    building_area_m2,
    building_compactness,
    building_perimeter_m,
    compute_metrics,
    edge_count,
    elongation,
    polygon_to_metric,
)

# ---------------------------------------------------------------------------
# Test fixtures — WGS84 polygons in Sydney, Australia (UTM zone 56S)
# ---------------------------------------------------------------------------


def _rect_polygon(lon: float, lat: float, width_deg: float, height_deg: float) -> Polygon:
    """Axis-aligned rectangle in WGS84."""
    return Polygon(
        [
            (lon, lat),
            (lon + width_deg, lat),
            (lon + width_deg, lat + height_deg),
            (lon, lat + height_deg),
        ]
    )


def _metric_square_polygon(lon: float, lat: float, side_m: float) -> Polygon:
    """Build a true square in UTM coordinates then back-project to WGS84.

    This guarantees the polygon is a square in metric space regardless of
    the latitude distortion of degree-based coordinates.
    """
    from pyproj import CRS, Transformer

    src = CRS.from_epsg(4326)
    # Determine UTM zone for the given point
    zone = int((lon + 180) / 6) % 60 + 1
    utm = CRS.from_dict({"proj": "utm", "zone": zone, "south": lat < 0})

    fwd = Transformer.from_crs(src, utm, always_xy=True)
    inv = Transformer.from_crs(utm, src, always_xy=True)

    cx, cy = fwd.transform(lon, lat)
    half = side_m / 2
    corners_utm = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]
    corners_wgs84 = [inv.transform(x, y) for x, y in corners_utm]
    return Polygon(corners_wgs84)


# ~100m x 50m rectangle near Sydney CBD
SYDNEY_RECT = _rect_polygon(151.2093, -33.8688, 0.0009, 0.00045)

# True 100m x 100m square in metric space
SYDNEY_SQUARE = _metric_square_polygon(151.2093, -33.8688, 100.0)

# An L-shaped polygon (8 exterior edges)
L_SHAPE = Polygon(
    [
        (151.2093, -33.8688),
        (151.2102, -33.8688),
        (151.2102, -33.8693),
        (151.2098, -33.8693),
        (151.2098, -33.8695),
        (151.2093, -33.8695),
    ]
)


# ---------------------------------------------------------------------------
# polygon_to_metric
# ---------------------------------------------------------------------------


class TestPolygonToMetric:
    def test_returns_polygon_and_crs(self) -> None:
        proj, crs = polygon_to_metric(SYDNEY_RECT)
        assert isinstance(proj, Polygon)
        assert isinstance(crs, CRS)

    def test_projected_is_metric(self) -> None:
        proj, crs = polygon_to_metric(SYDNEY_RECT)
        # In a metric CRS the area should be in m², not tiny degree² fractions
        assert proj.area > 1000  # at least 1000 m²

    def test_accepts_custom_dst_crs(self) -> None:
        gda2020 = CRS.from_epsg(7855)  # GDA2020 MGA Zone 55 (metric)
        src = CRS.from_epsg(4326)
        proj, crs_out = polygon_to_metric(SYDNEY_RECT, src_crs=src, dst_crs=gda2020)
        assert crs_out == gda2020
        assert proj.area > 1000


# ---------------------------------------------------------------------------
# building_area_m2
# ---------------------------------------------------------------------------


class TestBuildingAreaM2:
    def test_reasonable_range(self) -> None:
        area = building_area_m2(SYDNEY_RECT)
        # Roughly 100m * 50m = 5000 m², allow ±30% for projection distortion at test coords
        assert 3500 < area < 6500

    def test_larger_polygon_has_larger_area(self) -> None:
        small = _rect_polygon(151.2093, -33.8688, 0.0005, 0.0005)
        large = _rect_polygon(151.2093, -33.8688, 0.002, 0.002)
        assert building_area_m2(large) > building_area_m2(small)


# ---------------------------------------------------------------------------
# building_perimeter_m
# ---------------------------------------------------------------------------


class TestBuildingPerimeterM:
    def test_reasonable_range(self) -> None:
        perim = building_perimeter_m(SYDNEY_RECT)
        # Roughly 2*(100+50) = 300m, allow ±30%
        assert 200 < perim < 400

    def test_square_has_equal_sides(self) -> None:
        perim = building_perimeter_m(SYDNEY_SQUARE)
        area = building_area_m2(SYDNEY_SQUARE)
        # For a square: perimeter ≈ 4 * sqrt(area)
        side = math.sqrt(area)
        assert abs(perim - 4 * side) / perim < 0.05  # within 5%


# ---------------------------------------------------------------------------
# building_compactness
# ---------------------------------------------------------------------------


class TestBuildingCompactness:
    def test_range(self) -> None:
        c = building_compactness(SYDNEY_RECT)
        assert 0.0 < c <= 1.0

    def test_square_more_compact_than_rect(self) -> None:
        c_square = building_compactness(SYDNEY_SQUARE)
        c_rect = building_compactness(SYDNEY_RECT)
        assert c_square > c_rect

    def test_square_approaches_pi_over_4(self) -> None:
        # A square has compactness π/4 ≈ 0.785
        c = building_compactness(SYDNEY_SQUARE)
        assert abs(c - math.pi / 4) < 0.05


# ---------------------------------------------------------------------------
# aspect_ratio
# ---------------------------------------------------------------------------


class TestAspectRatio:
    def test_square_is_near_one(self) -> None:
        ar = aspect_ratio(SYDNEY_SQUARE)
        assert abs(ar - 1.0) < 0.05

    def test_rect_is_greater_than_one(self) -> None:
        ar = aspect_ratio(SYDNEY_RECT)
        assert ar > 1.0

    def test_rect_ratio_approximately_two(self) -> None:
        # 100m x 50m → aspect ~2
        ar = aspect_ratio(SYDNEY_RECT)
        assert 1.6 < ar < 2.4


# ---------------------------------------------------------------------------
# bbox_length_m / bbox_width_m
# ---------------------------------------------------------------------------


class TestBboxDims:
    def test_length_gte_width(self) -> None:
        assert bbox_length_m(SYDNEY_RECT) >= bbox_width_m(SYDNEY_RECT)

    def test_square_length_approx_equals_width(self) -> None:
        length = bbox_length_m(SYDNEY_SQUARE)
        width = bbox_width_m(SYDNEY_SQUARE)
        assert abs(length - width) / length < 0.05

    def test_rect_length_approx_100m(self) -> None:
        length = bbox_length_m(SYDNEY_RECT)
        assert 70 < length < 130

    def test_rect_width_approx_50m(self) -> None:
        w = bbox_width_m(SYDNEY_RECT)
        assert 35 < w < 65


# ---------------------------------------------------------------------------
# edge_count
# ---------------------------------------------------------------------------


class TestEdgeCount:
    def test_rectangle_has_four_edges(self) -> None:
        assert edge_count(SYDNEY_RECT) == 4

    def test_l_shape_has_six_edges(self) -> None:
        assert edge_count(L_SHAPE) == 6


# ---------------------------------------------------------------------------
# elongation
# ---------------------------------------------------------------------------


class TestElongation:
    def test_square_near_zero(self) -> None:
        e = elongation(SYDNEY_SQUARE)
        assert e < 0.1

    def test_rect_positive(self) -> None:
        e = elongation(SYDNEY_RECT)
        assert e > 0.0

    def test_range(self) -> None:
        e = elongation(SYDNEY_RECT)
        assert 0.0 <= e < 1.0


# ---------------------------------------------------------------------------
# compute_metrics — single-pass consistency
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_all_keys(self) -> None:
        expected_keys = {
            "building_area_m2",
            "building_perimeter_m",
            "building_compactness",
            "aspect_ratio",
            "bbox_length_m",
            "bbox_width_m",
            "edge_count",
            "log_area",
            "elongation",
        }
        metrics = compute_metrics(SYDNEY_RECT)
        assert set(metrics.keys()) == expected_keys

    def test_all_values_are_float(self) -> None:
        metrics = compute_metrics(SYDNEY_RECT)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} is {type(v)}"

    def test_log_area_consistent(self) -> None:
        metrics = compute_metrics(SYDNEY_RECT)
        assert abs(metrics["log_area"] - math.log(metrics["building_area_m2"])) < 1e-9

    def test_consistent_with_individual_functions(self) -> None:
        metrics = compute_metrics(SYDNEY_RECT)
        assert abs(metrics["building_area_m2"] - building_area_m2(SYDNEY_RECT)) < 1.0
        assert abs(metrics["building_perimeter_m"] - building_perimeter_m(SYDNEY_RECT)) < 1.0
        assert abs(metrics["aspect_ratio"] - aspect_ratio(SYDNEY_RECT)) < 0.01
        assert metrics["edge_count"] == float(edge_count(SYDNEY_RECT))

    def test_accepts_custom_src_crs(self) -> None:
        metrics = compute_metrics(SYDNEY_RECT, src_crs=CRS.from_epsg(4326))
        assert metrics["building_area_m2"] > 0

    def test_l_shape_edge_count(self) -> None:
        metrics = compute_metrics(L_SHAPE)
        assert metrics["edge_count"] == 6.0
