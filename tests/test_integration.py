"""End-to-end integration tests: polygon in → prediction out.

These tests exercise the full stack — feature extraction, model loading,
classifier predict, batch predict, Pulse compat — using a real trained
model artifact built in the test session (no bundled model required).

They are intentionally slow (5–10s) because they train a tiny real model
rather than mocking the pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Polygon

from footprint_ml._compat import from_pulse_signals, to_pulse_result
from footprint_ml.classifier import FootprintClassifier
from footprint_ml.model_registry import load_from_path
from footprint_ml.trainer import FootprintTrainer
from footprint_ml.types import AssetClass, Prediction


# ---------------------------------------------------------------------------
# Session-scoped real model artifact
# ---------------------------------------------------------------------------

_CLASSES = [
    "warehouse", "industrial", "retail", "office", "medical",
    "hospitality", "education", "childcare", "mixed_use", "other_commercial",
]
_N_PER_CLASS = 6  # minimal; enough for 2-fold CV


def _rect(lon: float, lat: float, size: float = 0.002) -> Polygon:
    return Polygon([
        (lon, lat), (lon + size, lat),
        (lon + size, lat + size), (lon, lat + size),
    ])


def _build_training_df() -> pd.DataFrame:
    rows = []
    for i, cls in enumerate(_CLASSES):
        for j in range(_N_PER_CLASS):
            rows.append({
                "geometry": _rect(151.0 + i * 0.05 + j * 0.005, -33.0 + j * 0.005),
                "asset_class": cls,
                "zone_code": "IND" if cls in ("warehouse", "industrial") else "B2",
                "osm_tags": {"building": cls if cls in ("warehouse", "industrial", "retail", "office") else "yes"},
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_artifact(tmp_path_factory: pytest.TempPathFactory):
    """Train a real (tiny) model once per test session and return its artifact."""
    tmp = tmp_path_factory.mktemp("model")
    trainer = FootprintTrainer(
        hgbc_params={"max_iter": 20, "max_depth": 3, "random_state": 42},
        cv_folds=2,
        version="integration_test_v1",
    )
    trainer.fit(_build_training_df())
    trainer.save(tmp)
    return load_from_path(tmp)


@pytest.fixture(scope="session")
def clf(trained_artifact):
    return FootprintClassifier(_artifact=trained_artifact)


# ---------------------------------------------------------------------------
# Core predict contract
# ---------------------------------------------------------------------------

class TestPredictContract:
    def test_returns_prediction(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert isinstance(pred, Prediction)

    def test_asset_class_is_valid(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert pred.asset_class in {ac.value for ac in AssetClass}

    def test_confidence_in_range(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert 0.0 <= pred.confidence <= 1.0

    def test_probabilities_sum_to_one(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

    def test_confidence_is_max_probability(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert abs(pred.confidence - max(pred.probabilities.values())) < 1e-9

    def test_asset_class_matches_argmax(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        best = max(pred.probabilities, key=lambda k: pred.probabilities[k])
        assert pred.asset_class == best

    def test_model_version_set(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert pred.model_version == "integration_test_v1"

    def test_all_asset_classes_in_probabilities(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8))
        assert set(pred.probabilities.keys()) == set(_CLASSES)


# ---------------------------------------------------------------------------
# Enriched predict (optional signals)
# ---------------------------------------------------------------------------

class TestPredictWithSignals:
    def test_zone_code_accepted(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(geometry=_rect(151.2, -33.8), zone_code="IND")
        assert isinstance(pred, Prediction)

    def test_osm_tags_accepted(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(
            geometry=_rect(151.2, -33.8),
            osm_tags={"building": "warehouse", "amenity": None},
        )
        assert isinstance(pred, Prediction)

    def test_anzsic_divisions_accepted(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(
            geometry=_rect(151.2, -33.8),
            anzsic_divisions=["F", "I"],
        )
        assert isinstance(pred, Prediction)

    def test_all_signals_together(self, clf: FootprintClassifier) -> None:
        pred = clf.predict(
            geometry=_rect(151.2, -33.8),
            zone_code="IND",
            osm_tags={"building": "warehouse"},
            anzsic_divisions=["F", "I"],
        )
        assert isinstance(pred, Prediction)


# ---------------------------------------------------------------------------
# Batch predict
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def _df(self, n: int = 5) -> pd.DataFrame:
        return pd.DataFrame({
            "geometry": [_rect(151.0 + i * 0.01, -33.0) for i in range(n)],
            "zone_code": ["IND"] * n,
        })

    def test_returns_list(self, clf: FootprintClassifier) -> None:
        preds = clf.predict_batch(self._df())
        assert isinstance(preds, list)

    def test_length_matches_input(self, clf: FootprintClassifier) -> None:
        preds = clf.predict_batch(self._df(n=7))
        assert len(preds) == 7

    def test_all_predictions_valid(self, clf: FootprintClassifier) -> None:
        preds = clf.predict_batch(self._df())
        for pred in preds:
            assert isinstance(pred, Prediction)
            assert pred.asset_class in {ac.value for ac in AssetClass}
            assert 0.0 <= pred.confidence <= 1.0

    def test_empty_df_returns_empty(self, clf: FootprintClassifier) -> None:
        df = pd.DataFrame({"geometry": []})
        assert clf.predict_batch(df) == []


# ---------------------------------------------------------------------------
# Pulse compat round-trip
# ---------------------------------------------------------------------------

class TestPulseCompat:
    def _signals(self) -> dict:
        return {
            "geometry": _rect(151.2, -33.8),
            "zone_code": "IND",
            "osm_tags": {"building": "warehouse"},
            "anzsic_divisions": ["F"],
            "gnaf_pid": "GANSW12345",
            "property_id": "prop-abc",
        }

    def test_from_pulse_signals_roundtrip(self, clf: FootprintClassifier) -> None:
        kwargs = from_pulse_signals(self._signals())
        pred = clf.predict(**kwargs)
        assert isinstance(pred, Prediction)

    def test_to_pulse_result_has_core_fields(self, clf: FootprintClassifier) -> None:
        signals = self._signals()
        pred = clf.predict(**from_pulse_signals(signals))
        result = to_pulse_result(pred, signals=signals)
        assert "asset_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "model_version" in result

    def test_to_pulse_result_passthrough_keys(self, clf: FootprintClassifier) -> None:
        signals = self._signals()
        pred = clf.predict(**from_pulse_signals(signals))
        result = to_pulse_result(pred, signals=signals)
        assert result["gnaf_pid"] == "GANSW12345"
        assert result["property_id"] == "prop-abc"

    def test_geometry_not_leaked_to_result(self, clf: FootprintClassifier) -> None:
        signals = self._signals()
        pred = clf.predict(**from_pulse_signals(signals))
        result = to_pulse_result(pred, signals=signals)
        assert "geometry" not in result


# ---------------------------------------------------------------------------
# Model registry round-trip: train → save → load → predict
# ---------------------------------------------------------------------------

class TestRegistryRoundTrip:
    def test_load_from_path_and_predict(self, trained_artifact) -> None:
        clf = FootprintClassifier(_artifact=trained_artifact)
        pred = clf.predict(geometry=_rect(151.5, -33.5))
        assert pred.asset_class in {ac.value for ac in AssetClass}

    def test_meta_json_survives_round_trip(self, tmp_path: Path) -> None:
        trainer = FootprintTrainer(
            hgbc_params={"max_iter": 10, "max_depth": 2, "random_state": 0},
            cv_folds=2,
            version="rt_test_v1",
        )
        trainer.fit(_build_training_df())
        out = trainer.save(tmp_path / "model")
        meta = json.loads((out / "meta.json").read_text())
        assert meta["version"] == "rt_test_v1"
        assert meta["asset_classes"] == sorted(_CLASSES)
        assert len(meta["feature_names"]) == 16

    def test_loaded_artifact_version_matches(self, tmp_path: Path) -> None:
        trainer = FootprintTrainer(
            hgbc_params={"max_iter": 10, "max_depth": 2, "random_state": 0},
            cv_folds=2,
            version="version_check_v1",
        )
        trainer.fit(_build_training_df())
        out = trainer.save(tmp_path / "model")
        artifact = load_from_path(out)
        assert artifact.version == "version_check_v1"


# ---------------------------------------------------------------------------
# Geometry variety: different building shapes
# ---------------------------------------------------------------------------

class TestGeometryVariety:
    _SHAPES = {
        "small_square": Polygon([(151.2, -33.8), (151.2005, -33.8), (151.2005, -33.8005), (151.2, -33.8005)]),
        "large_rect": Polygon([(151.2, -33.8), (151.205, -33.8), (151.205, -33.801), (151.2, -33.801)]),
        "l_shape": Polygon([
            (151.2, -33.8), (151.204, -33.8), (151.204, -33.802),
            (151.202, -33.802), (151.202, -33.803), (151.2, -33.803),
        ]),
    }

    @pytest.mark.parametrize("name,poly", _SHAPES.items())
    def test_predicts_for_shape(self, clf: FootprintClassifier, name: str, poly: Polygon) -> None:
        pred = clf.predict(geometry=poly)
        assert isinstance(pred, Prediction)
        assert pred.asset_class in {ac.value for ac in AssetClass}
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6
