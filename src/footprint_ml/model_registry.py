"""Model registry: load bundled model, load from path, download from GitHub Releases.

A model artifact is a directory (or zip extracted to a directory) containing:
  - ``model.joblib``  — serialised sklearn Pipeline (HGBC + calibration)
  - ``meta.json``     — version string, feature names, asset class labels,
                        training date, and optional encoder state

The bundled model lives at ``src/footprint_ml/models/`` inside the package
wheel and is loaded by default when no path is specified.
"""

from __future__ import annotations

import importlib.resources
import json
import urllib.request
from pathlib import Path
from typing import Any

import joblib

from footprint_ml.features import FEATURE_NAMES

# GitHub Releases base URL — updated when new model versions are published
_GITHUB_RELEASES_URL = (
    "https://github.com/stephancmorris/footprint-ml/releases/download"
)

# Filename convention inside every artifact directory
_MODEL_FILENAME = "model.joblib"
_META_FILENAME = "meta.json"

# Default meta used when no bundled model exists yet (scaffold / dev mode)
_FALLBACK_META: dict[str, Any] = {
    "version": "none",
    "feature_names": FEATURE_NAMES,
    "asset_classes": [
        "industrial", "warehouse", "retail", "office", "medical",
        "hospitality", "education", "childcare", "mixed_use", "other_commercial",
    ],
    "training_date": None,
}


# ---------------------------------------------------------------------------
# ModelArtifact
# ---------------------------------------------------------------------------

class ModelArtifact:
    """Container for a loaded model pipeline and its metadata."""

    def __init__(self, pipeline: Any, meta: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.meta = meta

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "unknown"))

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta.get("feature_names", FEATURE_NAMES))

    @property
    def asset_classes(self) -> list[str]:
        return list(self.meta.get("asset_classes", []))


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_meta(directory: Path) -> dict[str, Any]:
    meta_path = directory / _META_FILENAME
    if not meta_path.exists():
        return dict(_FALLBACK_META)
    with meta_path.open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _load_from_directory(directory: Path) -> ModelArtifact:
    model_path = directory / _MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model file found at {model_path}. "
            "Run scripts/train_bundled_model.py to generate a bundled model."
        )
    pipeline = joblib.load(model_path)
    meta = _load_meta(directory)
    return ModelArtifact(pipeline=pipeline, meta=meta)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_bundled() -> ModelArtifact:
    """Load the model bundled inside the installed package wheel.

    Raises ``FileNotFoundError`` if no bundled model has been trained yet
    (expected during development before Phase C is complete).
    """
    pkg_models = importlib.resources.files("footprint_ml") / "models"
    # importlib.resources gives us a Traversable; resolve to a real Path
    with importlib.resources.as_file(pkg_models) as models_dir:  # type: ignore[arg-type]
        return _load_from_directory(Path(models_dir))


def load_from_path(path: str | Path) -> ModelArtifact:
    """Load a model artifact from an explicit filesystem path.

    *path* may point to either:
    - A directory containing ``model.joblib`` + ``meta.json``, or
    - A ``.joblib`` file directly (meta defaults to fallback values).
    """
    p = Path(path)
    if p.is_dir():
        return _load_from_directory(p)
    if p.suffix == ".joblib" and p.exists():
        pipeline = joblib.load(p)
        meta_path = p.parent / _META_FILENAME
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(_FALLBACK_META)
        return ModelArtifact(pipeline=pipeline, meta=meta)
    raise FileNotFoundError(f"No model artifact found at {p}")


def download(version: str, dest: str | Path | None = None) -> ModelArtifact:
    """Download a model artifact from GitHub Releases and load it.

    Args:
        version: Release tag, e.g. ``"au_commercial_v1"``.
        dest: Directory to save the downloaded files. Defaults to a
            ``models_cache/<version>/`` directory alongside the package.

    Returns:
        Loaded :class:`ModelArtifact`.
    """
    if dest is None:
        cache_root = Path.home() / ".cache" / "footprint_ml" / "models"
    else:
        cache_root = Path(dest)

    version_dir = cache_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

    for filename in (_MODEL_FILENAME, _META_FILENAME):
        target = version_dir / filename
        if target.exists():
            continue  # already cached
        url = f"{_GITHUB_RELEASES_URL}/{version}/{filename}"
        try:
            urllib.request.urlretrieve(url, target)  # noqa: S310
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {filename} from {url}: {exc}"
            ) from exc

    return _load_from_directory(version_dir)


def load(
    *,
    path: str | Path | None = None,
    version: str | None = None,
) -> ModelArtifact:
    """Unified loader — selects the right strategy based on arguments.

    Priority:
    1. ``path`` provided → :func:`load_from_path`
    2. ``version`` provided → :func:`download` (with local cache)
    3. Neither → :func:`load_bundled`
    """
    if path is not None:
        return load_from_path(path)
    if version is not None:
        return download(version)
    return load_bundled()
