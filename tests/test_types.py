"""Tests for footprint_ml.types — AssetClass enum and Prediction dataclass."""

import pytest

from footprint_ml.types import AssetClass, Prediction

EXPECTED_ASSET_CLASSES = {
    "industrial",
    "warehouse",
    "retail",
    "office",
    "medical",
    "hospitality",
    "education",
    "childcare",
    "mixed_use",
    "other_commercial",
}


class TestAssetClass:
    def test_all_classes_present(self) -> None:
        values = {ac.value for ac in AssetClass}
        assert values == EXPECTED_ASSET_CLASSES

    def test_count(self) -> None:
        assert len(AssetClass) == 10

    def test_str_returns_value(self) -> None:
        assert str(AssetClass.WAREHOUSE) == "warehouse"
        assert str(AssetClass.OTHER_COMMERCIAL) == "other_commercial"

    def test_is_str_subclass(self) -> None:
        assert isinstance(AssetClass.RETAIL, str)

    def test_lookup_by_value(self) -> None:
        assert AssetClass("warehouse") is AssetClass.WAREHOUSE
        assert AssetClass("other_commercial") is AssetClass.OTHER_COMMERCIAL

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            AssetClass("car_park")

    def test_equality_with_string(self) -> None:
        assert AssetClass.OFFICE == "office"
        assert AssetClass.INDUSTRIAL == "industrial"


class TestPrediction:
    def _make_prediction(self, **kwargs: object) -> Prediction:
        defaults: dict[str, object] = {
            "asset_class": "warehouse",
            "confidence": 0.78,
            "probabilities": {"warehouse": 0.78, "industrial": 0.12, "retail": 0.10},
            "model_version": "au_commercial_v1",
        }
        defaults.update(kwargs)
        return Prediction(**defaults)  # type: ignore[arg-type]

    def test_basic_construction(self) -> None:
        pred = self._make_prediction()
        assert pred.asset_class == "warehouse"
        assert pred.confidence == 0.78
        assert pred.probabilities["warehouse"] == 0.78
        assert pred.model_version == "au_commercial_v1"

    def test_is_frozen(self) -> None:
        pred = self._make_prediction()
        with pytest.raises(AttributeError):
            pred.asset_class = "retail"  # type: ignore[misc]

    def test_equality(self) -> None:
        pred_a = self._make_prediction()
        pred_b = self._make_prediction()
        assert pred_a == pred_b

    def test_inequality(self) -> None:
        pred_a = self._make_prediction(confidence=0.78)
        pred_b = self._make_prediction(confidence=0.50)
        assert pred_a != pred_b

    def test_probabilities_are_dict(self) -> None:
        pred = self._make_prediction()
        assert isinstance(pred.probabilities, dict)
        assert all(isinstance(v, float) for v in pred.probabilities.values())

    def test_asset_class_can_be_enum_value(self) -> None:
        pred = self._make_prediction(asset_class=AssetClass.INDUSTRIAL.value)
        assert pred.asset_class == "industrial"
