"""Tests for footprint_ml.trainer — FootprintTrainer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from footprint_ml.model_registry import _META_FILENAME, _MODEL_FILENAME, load_from_path
from footprint_ml.trainer import FootprintTrainer, _build_pipeline, _min_class_count

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rect(lon: float, lat: float, size: float = 0.002) -> Polygon:
    """Small WGS84 rectangle — area well above the 200m² filter."""
    return Polygon(
        [
            (lon, lat),
            (lon + size, lat),
            (lon + size, lat + size),
            (lon, lat + size),
        ]
    )


# 8 samples per class × 5 classes = 40 rows (enough for 2-fold CV)
_CLASSES = ["warehouse", "industrial", "retail", "office", "medical"]
_N_PER_CLASS = 8


def _make_df(
    include_optional: bool = False,
    classes: list[str] | None = None,
    n_per_class: int = _N_PER_CLASS,
) -> pd.DataFrame:
    """Minimal labelled DataFrame for trainer tests."""
    used_classes = classes or _CLASSES
    rows = []
    for i, cls in enumerate(used_classes):
        for j in range(n_per_class):
            lon = 151.0 + i * 0.05 + j * 0.005
            lat = -33.0 + j * 0.005
            row: dict = {"geometry": _rect(lon, lat), "asset_class": cls}
            if include_optional:
                row["zone_code"] = "IND" if cls in ("warehouse", "industrial") else "B2"
                row["osm_tags"] = {"building": cls}
                row["anzsic_divisions"] = ["F"] if cls == "warehouse" else ["G"]
            rows.append(row)
    return pd.DataFrame(rows)


def _fast_trainer(**kwargs) -> FootprintTrainer:
    """Trainer with minimal hyperparams for speed in tests."""
    return FootprintTrainer(
        hgbc_params={"max_iter": 10, "max_depth": 3, "random_state": 0},
        cv_folds=2,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFootprintTrainerInit:
    def test_not_fitted_initially(self) -> None:
        trainer = FootprintTrainer()
        assert not trainer.is_fitted

    def test_classes_empty_before_fit(self) -> None:
        trainer = FootprintTrainer()
        assert trainer.classes == []

    def test_custom_hgbc_params(self) -> None:
        trainer = FootprintTrainer(hgbc_params={"max_iter": 50})
        assert trainer._hgbc_params["max_iter"] == 50

    def test_default_params_preserved_when_partial_override(self) -> None:
        trainer = FootprintTrainer(hgbc_params={"max_iter": 50})
        assert "max_depth" in trainer._hgbc_params


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_returns_self(self) -> None:
        trainer = _fast_trainer()
        result = trainer.fit(_make_df())
        assert result is trainer

    def test_is_fitted_after_fit(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        assert trainer.is_fitted

    def test_classes_populated_after_fit(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        assert set(trainer.classes) == set(_CLASSES)

    def test_classes_are_sorted(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        assert trainer.classes == sorted(trainer.classes)

    def test_fit_with_optional_columns(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df(include_optional=True))
        assert trainer.is_fitted

    def test_missing_geometry_column_raises(self) -> None:
        df = _make_df().rename(columns={"geometry": "geom"})
        trainer = _fast_trainer()
        with pytest.raises(ValueError, match="geometry"):
            trainer.fit(df, geometry_column="geometry")

    def test_missing_label_column_raises(self) -> None:
        trainer = _fast_trainer()
        with pytest.raises(ValueError, match="asset_class"):
            trainer.fit(_make_df(), label_column="asset_class_WRONG")

    def test_invalid_label_raises(self) -> None:
        df = _make_df()
        df.loc[0, "asset_class"] = "car_park"
        trainer = _fast_trainer()
        with pytest.raises(ValueError, match="car_park"):
            trainer.fit(df)

    def test_empty_df_raises(self) -> None:
        trainer = _fast_trainer()
        with pytest.raises(ValueError, match="empty"):
            trainer.fit(pd.DataFrame({"geometry": [], "asset_class": []}))


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_expected_keys(self) -> None:
        trainer = _fast_trainer()
        metrics = trainer.evaluate(_make_df())
        assert "macro_f1" in metrics
        assert "per_class_f1" in metrics
        assert "cv_scores" in metrics
        assert "classification_report" in metrics

    def test_macro_f1_in_range(self) -> None:
        trainer = _fast_trainer()
        metrics = trainer.evaluate(_make_df())
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_cv_scores_length(self) -> None:
        trainer = _fast_trainer()
        metrics = trainer.evaluate(_make_df())
        assert len(metrics["cv_scores"]) == 2  # cv_folds=2

    def test_per_class_f1_contains_all_classes(self) -> None:
        trainer = _fast_trainer()
        metrics = trainer.evaluate(_make_df())
        for cls in _CLASSES:
            assert cls in metrics["per_class_f1"]

    def test_evaluate_does_not_require_prior_fit(self) -> None:
        # evaluate() fits its own folds — no dependency on fit()
        trainer = _fast_trainer()
        metrics = trainer.evaluate(_make_df())
        assert isinstance(metrics["macro_f1"], float)


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_before_fit_raises(self) -> None:
        trainer = _fast_trainer()
        with pytest.raises(RuntimeError, match="fit"):
            trainer.save("/tmp/no_model")

    def test_creates_model_joblib(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            assert (out / _MODEL_FILENAME).exists()

    def test_creates_meta_json(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            assert (out / _META_FILENAME).exists()

    def test_meta_json_contents(self) -> None:
        trainer = _fast_trainer(version="test_v1")
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            meta = json.loads((out / _META_FILENAME).read_text())
            assert meta["version"] == "test_v1"
            assert meta["asset_classes"] == sorted(_CLASSES)
            assert "feature_names" in meta
            assert "training_date" in meta

    def test_returns_path_object(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            result = trainer.save(tmp)
            assert isinstance(result, Path)

    def test_creates_output_dir_if_absent(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "new_subdir" / "model"
            trainer.save(out_dir)
            assert out_dir.exists()

    def test_custom_training_date(self) -> None:
        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp, training_date="2026-01-01")
            meta = json.loads((out / _META_FILENAME).read_text())
            assert meta["training_date"] == "2026-01-01"


# ---------------------------------------------------------------------------
# Round-trip: save → load_from_path → classify
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_saved_model_loadable(self) -> None:
        trainer = _fast_trainer(version="rt_v1")
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            artifact = load_from_path(out)
            assert artifact.version == "rt_v1"

    def test_loaded_model_can_predict(self) -> None:
        from footprint_ml.classifier import FootprintClassifier

        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            artifact = load_from_path(out)
            clf = FootprintClassifier(_artifact=artifact)
            pred = clf.predict(geometry=_rect(151.2, -33.8))
            assert pred.asset_class in _CLASSES
            assert 0.0 <= pred.confidence <= 1.0

    def test_probabilities_sum_to_one(self) -> None:
        from footprint_ml.classifier import FootprintClassifier

        trainer = _fast_trainer()
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            artifact = load_from_path(out)
            clf = FootprintClassifier(_artifact=artifact)
            pred = clf.predict(geometry=_rect(151.2, -33.8))
            assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

    def test_model_version_in_prediction(self) -> None:
        from footprint_ml.classifier import FootprintClassifier

        trainer = _fast_trainer(version="round_trip_v1")
        trainer.fit(_make_df())
        with tempfile.TemporaryDirectory() as tmp:
            out = trainer.save(tmp)
            artifact = load_from_path(out)
            clf = FootprintClassifier(_artifact=artifact)
            pred = clf.predict(geometry=_rect(151.2, -33.8))
            assert pred.model_version == "round_trip_v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_min_class_count(self) -> None:
        y = np.array(["a", "a", "a", "b", "b", "c"])
        assert _min_class_count(y) == 1

    def test_build_pipeline_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline

        p = _build_pipeline(
            {"max_iter": 10, "random_state": 0},
            calibration_method="isotonic",
            cv=2,
        )
        assert isinstance(p, Pipeline)
