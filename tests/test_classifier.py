"""Tests for footprint_ml.classifier — FootprintClassifier."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
from shapely.geometry import Polygon

from footprint_ml.classifier import FootprintClassifier, _is_missing
from footprint_ml.features import FEATURE_NAMES
from footprint_ml.model_registry import ModelArtifact
from footprint_ml.types import Prediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_CLASSES = [
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
]

# Probabilities that sum to 1.0; warehouse is the top class
_WAREHOUSE_IDX = ALL_CLASSES.index("warehouse")
_PROBA = np.full(len(ALL_CLASSES), 0.02)
_PROBA[_WAREHOUSE_IDX] = 1.0 - 0.02 * (len(ALL_CLASSES) - 1)


def _fake_pipeline(top_class_idx: int = _WAREHOUSE_IDX) -> MagicMock:
    pipeline = MagicMock()
    pipeline.classes_ = np.array(ALL_CLASSES)
    proba = np.full(len(ALL_CLASSES), 0.02)
    proba[top_class_idx] = 1.0 - 0.02 * (len(ALL_CLASSES) - 1)
    pipeline.predict_proba.return_value = proba.reshape(1, -1)
    return pipeline


def _fake_artifact(top_class_idx: int = _WAREHOUSE_IDX) -> ModelArtifact:
    return ModelArtifact(
        pipeline=_fake_pipeline(top_class_idx),
        meta={
            "version": "test_v1",
            "feature_names": FEATURE_NAMES,
            "asset_classes": ALL_CLASSES,
        },
    )


def _make_clf(top_class_idx: int = _WAREHOUSE_IDX) -> FootprintClassifier:
    return FootprintClassifier(_artifact=_fake_artifact(top_class_idx))


def _rect(lon: float = 151.2093, lat: float = -33.8688) -> Polygon:
    return Polygon(
        [
            (lon, lat),
            (lon + 0.001, lat),
            (lon + 0.001, lat + 0.001),
            (lon, lat + 0.001),
        ]
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFootprintClassifierInit:
    def test_inject_artifact(self) -> None:
        clf = _make_clf()
        assert clf.model_version == "test_v1"

    def test_model_version_property(self) -> None:
        clf = _make_clf()
        assert clf.model_version == "test_v1"

    def test_asset_classes_property(self) -> None:
        clf = _make_clf()
        assert clf.asset_classes == ALL_CLASSES


# ---------------------------------------------------------------------------
# predict — return type and shape
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_prediction(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert isinstance(pred, Prediction)

    def test_asset_class_is_string(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert isinstance(pred.asset_class, str)

    def test_asset_class_is_known(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert pred.asset_class in ALL_CLASSES

    def test_confidence_in_range(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert 0.0 <= pred.confidence <= 1.0

    def test_probabilities_keys_match_classes(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert set(pred.probabilities.keys()) == set(ALL_CLASSES)

    def test_probabilities_sum_to_one(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

    def test_confidence_matches_top_probability(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert abs(pred.confidence - pred.probabilities[pred.asset_class]) < 1e-9

    def test_model_version_in_prediction(self) -> None:
        clf = _make_clf()
        pred = clf.predict(geometry=_rect())
        assert pred.model_version == "test_v1"

    def test_top_class_is_warehouse(self) -> None:
        clf = _make_clf(top_class_idx=_WAREHOUSE_IDX)
        pred = clf.predict(geometry=_rect())
        assert pred.asset_class == "warehouse"

    def test_top_class_changes_with_pipeline(self) -> None:
        office_idx = ALL_CLASSES.index("office")
        clf = _make_clf(top_class_idx=office_idx)
        pred = clf.predict(geometry=_rect())
        assert pred.asset_class == "office"

    def test_accepts_optional_signals(self) -> None:
        clf = _make_clf()
        pred = clf.predict(
            geometry=_rect(),
            zone_code="IND",
            osm_tags={"building": "warehouse"},
            anzsic_divisions=["F", "I"],
        )
        assert isinstance(pred, Prediction)

    def test_pipeline_called_once_per_predict(self) -> None:
        artifact = _fake_artifact()
        clf = FootprintClassifier(_artifact=artifact)
        clf.predict(geometry=_rect())
        artifact.pipeline.predict_proba.assert_called_once()

    def test_pipeline_receives_correct_feature_count(self) -> None:
        artifact = _fake_artifact()
        clf = FootprintClassifier(_artifact=artifact)
        clf.predict(geometry=_rect())
        call_args = artifact.pipeline.predict_proba.call_args
        X = call_args[0][0]
        assert X.shape == (1, len(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    def _make_df(self, n: int = 3, include_optional: bool = False) -> Any:
        import pandas as pd

        data: dict[str, Any] = {"geometry": [_rect() for _ in range(n)]}
        if include_optional:
            data["zone_code"] = ["IND"] * n
            data["osm_tags"] = [{"building": "warehouse"}] * n
            data["anzsic_divisions"] = [["F"]] * n
        return pd.DataFrame(data)

    def test_returns_list_of_predictions(self) -> None:
        clf = _make_clf()
        preds = clf.predict_batch(self._make_df())
        assert isinstance(preds, list)
        assert all(isinstance(p, Prediction) for p in preds)

    def test_length_matches_dataframe(self) -> None:
        clf = _make_clf()
        df = self._make_df(n=5)
        preds = clf.predict_batch(df)
        assert len(preds) == 5

    def test_with_optional_columns(self) -> None:
        clf = _make_clf()
        df = self._make_df(include_optional=True)
        preds = clf.predict_batch(df)
        assert len(preds) == 3

    def test_handles_nan_optional_fields(self) -> None:
        import pandas as pd

        clf = _make_clf()
        df = pd.DataFrame(
            {
                "geometry": [_rect(), _rect()],
                "zone_code": [None, "IND"],
            }
        )
        preds = clf.predict_batch(df)
        assert len(preds) == 2

    def test_empty_dataframe_returns_empty_list(self) -> None:
        import pandas as pd

        clf = _make_clf()
        df = pd.DataFrame({"geometry": []})
        preds = clf.predict_batch(df)
        assert preds == []


# ---------------------------------------------------------------------------
# _is_missing helper
# ---------------------------------------------------------------------------


class TestIsMissing:
    def test_none_is_missing(self) -> None:
        assert _is_missing(None)

    def test_string_not_missing(self) -> None:
        assert not _is_missing("IND")

    def test_zero_not_missing(self) -> None:
        assert not _is_missing(0)

    def test_empty_list_not_missing(self) -> None:
        assert not _is_missing([])

    def test_pandas_nan_is_missing(self) -> None:
        import pandas as pd

        assert _is_missing(pd.NA)
        assert _is_missing(float("nan"))
