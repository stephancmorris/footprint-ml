"""Tests for footprint_ml.features feature extraction."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from footprint_ml.features import (
    FEATURE_NAMES,
    _encode_anzsic,
    _encode_osm_amenity,
    _encode_osm_building,
    _encode_zone,
    _has_osm_office,
    _has_osm_shop,
    extract_features,
    feature_vector,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _rect(lon: float, lat: float, dlon: float, dlat: float) -> Polygon:
    return Polygon([
        (lon, lat),
        (lon + dlon, lat),
        (lon + dlon, lat + dlat),
        (lon, lat + dlat),
    ])


# ~100m x 50m near Sydney
POLY = _rect(151.2093, -33.8688, 0.0009, 0.00045)


# ---------------------------------------------------------------------------
# FEATURE_NAMES contract
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_count(self) -> None:
        assert len(FEATURE_NAMES) == 16

    def test_geometry_features_first(self) -> None:
        geom = FEATURE_NAMES[:9]
        assert "building_area_m2" in geom
        assert "log_area" in geom
        assert "elongation" in geom

    def test_no_duplicates(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Internal encoders
# ---------------------------------------------------------------------------

class TestEncodeZone:
    def test_none_returns_nan(self) -> None:
        assert math.isnan(_encode_zone(None))

    def test_same_input_same_output(self) -> None:
        assert _encode_zone("IND") == _encode_zone("IND")

    def test_case_insensitive(self) -> None:
        assert _encode_zone("ind") == _encode_zone("IND")

    def test_whitespace_stripped(self) -> None:
        assert _encode_zone("  IND  ") == _encode_zone("IND")

    def test_different_codes_differ(self) -> None:
        # Not guaranteed for all pairs, but these shouldn't collide
        assert _encode_zone("IND") != _encode_zone("B2")

    def test_returns_float(self) -> None:
        assert isinstance(_encode_zone("IND"), float)


class TestEncodeOsmBuilding:
    def test_none_tags_returns_nan(self) -> None:
        assert math.isnan(_encode_osm_building(None))

    def test_empty_dict_returns_nan(self) -> None:
        assert math.isnan(_encode_osm_building({}))

    def test_missing_building_key_returns_nan(self) -> None:
        assert math.isnan(_encode_osm_building({"amenity": "hospital"}))

    def test_warehouse(self) -> None:
        assert _encode_osm_building({"building": "warehouse"}) == 1.0

    def test_industrial(self) -> None:
        assert _encode_osm_building({"building": "industrial"}) == 2.0

    def test_retail(self) -> None:
        assert _encode_osm_building({"building": "retail"}) == 3.0

    def test_unknown_value_returns_zero(self) -> None:
        assert _encode_osm_building({"building": "castle"}) == 0.0

    def test_case_insensitive(self) -> None:
        assert _encode_osm_building({"building": "WAREHOUSE"}) == 1.0


class TestEncodeOsmAmenity:
    def test_none_tags_returns_nan(self) -> None:
        assert math.isnan(_encode_osm_amenity(None))

    def test_missing_amenity_key_returns_nan(self) -> None:
        assert math.isnan(_encode_osm_amenity({"building": "warehouse"}))

    def test_hospital(self) -> None:
        assert _encode_osm_amenity({"amenity": "hospital"}) == 1.0

    def test_school(self) -> None:
        assert _encode_osm_amenity({"amenity": "school"}) == 2.0

    def test_unknown_returns_zero(self) -> None:
        assert _encode_osm_amenity({"amenity": "bench"}) == 0.0


class TestHasOsmShop:
    def test_none_tags_returns_zero(self) -> None:
        assert _has_osm_shop(None) == 0.0

    def test_shop_present(self) -> None:
        assert _has_osm_shop({"shop": "supermarket"}) == 1.0

    def test_shop_none_value(self) -> None:
        assert _has_osm_shop({"shop": None}) == 0.0

    def test_shop_empty_string(self) -> None:
        assert _has_osm_shop({"shop": ""}) == 0.0

    def test_no_shop_key(self) -> None:
        assert _has_osm_shop({"building": "retail"}) == 0.0


class TestHasOsmOffice:
    def test_none_tags_returns_zero(self) -> None:
        assert _has_osm_office(None) == 0.0

    def test_office_present(self) -> None:
        assert _has_osm_office({"office": "yes"}) == 1.0

    def test_office_none_value(self) -> None:
        assert _has_osm_office({"office": None}) == 0.0

    def test_no_office_key(self) -> None:
        assert _has_osm_office({"building": "office"}) == 0.0


class TestEncodeAnzsic:
    def test_none_returns_nan_and_zero(self) -> None:
        primary, count = _encode_anzsic(None)
        assert math.isnan(primary)
        assert count == 0.0

    def test_empty_list_returns_nan_and_zero(self) -> None:
        primary, count = _encode_anzsic([])
        assert math.isnan(primary)
        assert count == 0.0

    def test_single_known_division(self) -> None:
        primary, count = _encode_anzsic(["F"])
        assert primary == 6.0   # Wholesale Trade
        assert count == 1.0

    def test_multiple_divisions(self) -> None:
        primary, count = _encode_anzsic(["F", "I"])
        assert primary == 6.0   # first entry is primary
        assert count == 2.0

    def test_deduplication(self) -> None:
        _, count = _encode_anzsic(["F", "F", "I"])
        assert count == 2.0

    def test_case_insensitive(self) -> None:
        primary_lower, _ = _encode_anzsic(["f"])
        primary_upper, _ = _encode_anzsic(["F"])
        assert primary_lower == primary_upper

    def test_unknown_division_zero(self) -> None:
        primary, _ = _encode_anzsic(["Z"])
        assert primary == 0.0


# ---------------------------------------------------------------------------
# extract_features — output shape and types
# ---------------------------------------------------------------------------

class TestExtractFeaturesShape:
    def test_returns_dict(self) -> None:
        result = extract_features(POLY)
        assert isinstance(result, dict)

    def test_has_all_feature_names(self) -> None:
        result = extract_features(POLY)
        assert set(result.keys()) == set(FEATURE_NAMES)

    def test_all_values_are_float(self) -> None:
        result = extract_features(POLY)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_geometry_features_not_nan(self) -> None:
        result = extract_features(POLY)
        geom_keys = FEATURE_NAMES[:9]
        for k in geom_keys:
            assert not math.isnan(result[k]), f"{k} should not be NaN"

    def test_optional_features_nan_when_absent(self) -> None:
        result = extract_features(POLY)
        assert math.isnan(result["zone_code_encoded"])
        assert math.isnan(result["osm_amenity_encoded"])
        assert math.isnan(result["osm_building_use_encoded"])
        assert math.isnan(result["anzsic_primary_division_encoded"])

    def test_boolean_flags_zero_when_absent(self) -> None:
        result = extract_features(POLY)
        assert result["has_osm_shop"] == 0.0
        assert result["has_osm_office"] == 0.0

    def test_anzsic_count_zero_when_absent(self) -> None:
        result = extract_features(POLY)
        assert result["anzsic_count"] == 0.0


# ---------------------------------------------------------------------------
# extract_features — with optional signals
# ---------------------------------------------------------------------------

class TestExtractFeaturesWithSignals:
    def test_zone_code_encoded(self) -> None:
        result = extract_features(POLY, zone_code="IND")
        assert not math.isnan(result["zone_code_encoded"])

    def test_osm_warehouse_building(self) -> None:
        result = extract_features(POLY, osm_tags={"building": "warehouse"})
        assert result["osm_building_use_encoded"] == 1.0
        assert math.isnan(result["osm_amenity_encoded"])

    def test_osm_shop_flag(self) -> None:
        result = extract_features(POLY, osm_tags={"shop": "supermarket"})
        assert result["has_osm_shop"] == 1.0

    def test_osm_office_flag(self) -> None:
        result = extract_features(POLY, osm_tags={"office": "yes"})
        assert result["has_osm_office"] == 1.0

    def test_anzsic_divisions(self) -> None:
        result = extract_features(POLY, anzsic_divisions=["F", "I"])
        assert result["anzsic_primary_division_encoded"] == 6.0
        assert result["anzsic_count"] == 2.0

    def test_all_signals_together(self) -> None:
        result = extract_features(
            POLY,
            zone_code="IND",
            osm_tags={"building": "warehouse", "amenity": None},
            anzsic_divisions=["F", "I"],
        )
        assert not math.isnan(result["zone_code_encoded"])
        assert result["osm_building_use_encoded"] == 1.0
        assert result["anzsic_count"] == 2.0


# ---------------------------------------------------------------------------
# feature_vector
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_returns_list(self) -> None:
        vec = feature_vector(POLY)
        assert isinstance(vec, list)

    def test_length_matches_feature_names(self) -> None:
        vec = feature_vector(POLY)
        assert len(vec) == len(FEATURE_NAMES)

    def test_order_matches_feature_names(self) -> None:
        feats = extract_features(POLY, zone_code="IND", anzsic_divisions=["G"])
        vec = feature_vector(POLY, zone_code="IND", anzsic_divisions=["G"])
        for i, name in enumerate(FEATURE_NAMES):
            expected = feats[name]
            actual = vec[i]
            if math.isnan(expected):
                assert math.isnan(actual), f"position {i} ({name}): expected NaN, got {actual}"
            else:
                assert actual == expected, f"position {i} ({name}) mismatch"

    def test_all_floats(self) -> None:
        vec = feature_vector(POLY)
        assert all(isinstance(v, float) for v in vec)
