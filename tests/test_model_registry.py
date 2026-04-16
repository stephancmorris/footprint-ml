"""Tests for footprint_ml.model_registry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from footprint_ml.features import FEATURE_NAMES
from footprint_ml.model_registry import (
    ModelArtifact,
    _FALLBACK_META,
    _META_FILENAME,
    _MODEL_FILENAME,
    load,
    load_from_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pipeline(classes: list[str] | None = None) -> Pipeline:
    """Minimal real sklearn pipeline that is picklable by joblib."""
    if classes is None:
        classes = ["industrial", "warehouse", "retail"]
    n = len(classes)
    # DummyClassifier fitted on trivial data — picklable, predict_proba works
    clf = DummyClassifier(strategy="uniform")
    X = np.zeros((n, len(FEATURE_NAMES)))
    y = classes
    clf.fit(X, y)
    return clf  # type: ignore[return-value]


def _write_artifact(directory: Path, meta: dict[str, Any] | None = None) -> None:
    """Write a minimal model artifact to *directory*."""
    pipeline = _fake_pipeline()
    joblib.dump(pipeline, directory / _MODEL_FILENAME)
    effective_meta = meta or {
        "version": "test_v1",
        "feature_names": FEATURE_NAMES,
        "asset_classes": ["industrial", "warehouse", "retail"],
    }
    (directory / _META_FILENAME).write_text(json.dumps(effective_meta))


# ---------------------------------------------------------------------------
# ModelArtifact
# ---------------------------------------------------------------------------

class TestModelArtifact:
    def test_version(self) -> None:
        meta = {"version": "au_commercial_v1", "feature_names": [], "asset_classes": []}
        art = ModelArtifact(pipeline=_fake_pipeline(), meta=meta)
        assert art.version == "au_commercial_v1"

    def test_version_fallback(self) -> None:
        art = ModelArtifact(pipeline=_fake_pipeline(), meta={})
        assert art.version == "unknown"

    def test_feature_names(self) -> None:
        meta = {"version": "v1", "feature_names": FEATURE_NAMES, "asset_classes": []}
        art = ModelArtifact(pipeline=_fake_pipeline(), meta=meta)
        assert art.feature_names == FEATURE_NAMES

    def test_feature_names_fallback(self) -> None:
        art = ModelArtifact(pipeline=_fake_pipeline(), meta={})
        assert art.feature_names == FEATURE_NAMES

    def test_asset_classes(self) -> None:
        classes = ["warehouse", "retail"]
        meta = {"version": "v1", "feature_names": [], "asset_classes": classes}
        art = ModelArtifact(pipeline=_fake_pipeline(), meta=meta)
        assert art.asset_classes == classes


# ---------------------------------------------------------------------------
# load_from_path — directory
# ---------------------------------------------------------------------------

class TestLoadFromPathDirectory:
    def test_loads_model_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_artifact(d)
            art = load_from_path(d)
            assert art.version == "test_v1"
            assert art.feature_names == FEATURE_NAMES

    def test_missing_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                load_from_path(Path(tmp))

    def test_missing_meta_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            joblib.dump(_fake_pipeline(), d / _MODEL_FILENAME)
            art = load_from_path(d)
            assert art.version == _FALLBACK_META["version"]

    def test_returns_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_artifact(d)
            art = load_from_path(d)
            assert isinstance(art, ModelArtifact)


# ---------------------------------------------------------------------------
# load_from_path — direct .joblib file
# ---------------------------------------------------------------------------

class TestLoadFromPathJoblib:
    def test_loads_joblib_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            joblib.dump(_fake_pipeline(), d / _MODEL_FILENAME)
            art = load_from_path(d / _MODEL_FILENAME)
            assert isinstance(art, ModelArtifact)

    def test_loads_adjacent_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            joblib.dump(_fake_pipeline(), d / _MODEL_FILENAME)
            meta = {"version": "file_v1", "feature_names": FEATURE_NAMES, "asset_classes": []}
            (d / _META_FILENAME).write_text(json.dumps(meta))
            art = load_from_path(d / _MODEL_FILENAME)
            assert art.version == "file_v1"

    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_from_path("/nonexistent/path/model.joblib")


# ---------------------------------------------------------------------------
# load() unified entry point
# ---------------------------------------------------------------------------

class TestLoad:
    def test_path_takes_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_artifact(d)
            art = load(path=d)
            assert art.version == "test_v1"

    def test_no_args_calls_load_bundled(self) -> None:
        with patch("footprint_ml.model_registry.load_bundled") as mock_lb:
            mock_lb.return_value = ModelArtifact(pipeline=_fake_pipeline(), meta={"version": "bundled"})
            art = load()
            mock_lb.assert_called_once()
            assert art.version == "bundled"

    def test_version_calls_download(self) -> None:
        with patch("footprint_ml.model_registry.download") as mock_dl:
            mock_dl.return_value = ModelArtifact(pipeline=_fake_pipeline(), meta={"version": "dl_v1"})
            art = load(version="au_commercial_v1")
            mock_dl.assert_called_once_with("au_commercial_v1")
            assert art.version == "dl_v1"
