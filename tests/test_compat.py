"""Tests for footprint_ml._compat — from_pulse_signals / to_pulse_result."""

from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon

from footprint_ml._compat import _IGNORED_PULSE_KEYS, from_pulse_signals, to_pulse_result
from footprint_ml.types import Prediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _poly() -> Polygon:
    return Polygon([(151.0, -33.0), (151.001, -33.0), (151.001, -33.001), (151.0, -33.001)])


def _minimal_signals() -> dict:
    return {"geometry": _poly()}


def _full_signals() -> dict:
    return {
        "geometry": _poly(),
        "zone_code": "IND",
        "osm_tags": {"building": "warehouse"},
        "anzsic_divisions": ["F", "I"],
        "confidence_source": "osm",
        "gnaf_pid": "GANSW123",
        "property_id": "abc-456",
    }


def _pred() -> Prediction:
    return Prediction(
        asset_class="warehouse",
        confidence=0.78,
        probabilities={"warehouse": 0.78, "industrial": 0.12, "retail": 0.10},
        model_version="test_v1",
    )


# ---------------------------------------------------------------------------
# from_pulse_signals
# ---------------------------------------------------------------------------


class TestFromPulseSignals:
    def test_minimal_signals(self) -> None:
        result = from_pulse_signals(_minimal_signals())
        assert "geometry" in result
        assert isinstance(result["geometry"], Polygon)

    def test_returns_all_expected_keys(self) -> None:
        result = from_pulse_signals(_minimal_signals())
        assert set(result.keys()) == {"geometry", "zone_code", "osm_tags", "anzsic_divisions"}

    def test_optional_signals_none_when_absent(self) -> None:
        result = from_pulse_signals(_minimal_signals())
        assert result["zone_code"] is None
        assert result["osm_tags"] is None
        assert result["anzsic_divisions"] is None

    def test_full_signals_passed_through(self) -> None:
        result = from_pulse_signals(_full_signals())
        assert result["zone_code"] == "IND"
        assert result["osm_tags"] == {"building": "warehouse"}
        assert result["anzsic_divisions"] == ["F", "I"]

    def test_pulse_internal_keys_not_in_result(self) -> None:
        result = from_pulse_signals(_full_signals())
        assert "confidence_source" not in result
        assert "gnaf_pid" not in result
        assert "property_id" not in result

    def test_missing_geometry_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="geometry"):
            from_pulse_signals({"zone_code": "IND"})

    def test_non_polygon_geometry_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            from_pulse_signals({"geometry": Point(151.0, -33.0)})

    def test_result_is_unpackable_to_predict(self) -> None:
        """Keys in the result must exactly match FootprintClassifier.predict() kwargs."""
        import inspect

        from footprint_ml.classifier import FootprintClassifier

        sig = inspect.signature(FootprintClassifier.predict)
        valid_kwargs = set(sig.parameters.keys()) - {"self"}
        result = from_pulse_signals(_full_signals())
        assert set(result.keys()).issubset(valid_kwargs)


# ---------------------------------------------------------------------------
# to_pulse_result
# ---------------------------------------------------------------------------


class TestToPulseResult:
    def test_returns_dict(self) -> None:
        result = to_pulse_result(_pred())
        assert isinstance(result, dict)

    def test_core_fields_present(self) -> None:
        result = to_pulse_result(_pred())
        assert result["asset_class"] == "warehouse"
        assert result["confidence"] == 0.78
        assert result["model_version"] == "test_v1"
        assert isinstance(result["probabilities"], dict)

    def test_passthrough_keys_copied(self) -> None:
        signals = _full_signals()
        result = to_pulse_result(_pred(), signals=signals)
        assert result["gnaf_pid"] == "GANSW123"
        assert result["property_id"] == "abc-456"
        assert result["confidence_source"] == "osm"

    def test_geometry_not_in_result(self) -> None:
        result = to_pulse_result(_pred(), signals=_full_signals())
        assert "geometry" not in result

    def test_no_signals_no_passthrough(self) -> None:
        result = to_pulse_result(_pred(), signals=None)
        assert "gnaf_pid" not in result

    def test_signals_without_ignored_keys(self) -> None:
        signals = {"geometry": _poly(), "zone_code": "IND"}
        result = to_pulse_result(_pred(), signals=signals)
        # zone_code is not in _IGNORED_PULSE_KEYS so not copied
        assert "zone_code" not in result

    def test_ignored_pulse_keys_constant(self) -> None:
        assert "gnaf_pid" in _IGNORED_PULSE_KEYS
        assert "property_id" in _IGNORED_PULSE_KEYS
        assert "confidence_source" in _IGNORED_PULSE_KEYS
