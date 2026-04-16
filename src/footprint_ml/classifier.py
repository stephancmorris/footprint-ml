"""FootprintClassifier: predict asset class from building footprint polygon."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from footprint_ml.features import extract_features
from footprint_ml.model_registry import ModelArtifact, load
from footprint_ml.types import Prediction

if TYPE_CHECKING:
    from pyproj import CRS
    from shapely.geometry import Polygon


class FootprintClassifier:
    """Classify building footprint polygons into commercial property asset classes.

    The classifier wraps a trained sklearn pipeline (HistGradientBoostingClassifier
    + CalibratedClassifierCV) and handles feature extraction internally.

    Usage::

        clf = FootprintClassifier()                  # loads bundled model
        pred = clf.predict(geometry=polygon)
        # pred.asset_class   → "warehouse"
        # pred.confidence    → 0.78
        # pred.probabilities → {"warehouse": 0.78, "industrial": 0.12, ...}

        # With optional signals
        pred = clf.predict(
            geometry=polygon,
            zone_code="IND",
            osm_tags={"building": "warehouse"},
            anzsic_divisions=["F", "I"],
        )

        # Batch prediction from a DataFrame
        preds = clf.predict_batch(df)
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model_version: str | None = None,
        _artifact: ModelArtifact | None = None,  # for testing / injection
    ) -> None:
        """Load a model artifact.

        Args:
            model_path: Path to a model directory or ``.joblib`` file.
                Defaults to the bundled model.
            model_version: GitHub Releases version tag to download,
                e.g. ``"au_commercial_v1"``.
            _artifact: Inject a pre-loaded :class:`~footprint_ml.model_registry.ModelArtifact`
                directly (used in tests).
        """
        if _artifact is not None:
            self._artifact = _artifact
        else:
            self._artifact = load(path=model_path, version=model_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        geometry: Polygon,
        *,
        zone_code: str | None = None,
        osm_tags: dict[str, Any] | None = None,
        anzsic_divisions: list[str] | None = None,
        src_crs: CRS | None = None,
    ) -> Prediction:
        """Predict the asset class for a single building footprint.

        Args:
            geometry: Building footprint polygon in WGS84 (or *src_crs*).
            zone_code: Zoning code string, e.g. ``"IND"``.
            osm_tags: Raw OSM key→value tag dict for the building.
            anzsic_divisions: List of ANZSIC primary division letters.
            src_crs: CRS of *geometry*. Defaults to WGS84 (EPSG:4326).

        Returns:
            :class:`~footprint_ml.types.Prediction` with asset class,
            confidence, full probability distribution, and model version.
        """
        features = extract_features(
            geometry,
            zone_code=zone_code,
            osm_tags=osm_tags,
            anzsic_divisions=anzsic_divisions,
            src_crs=src_crs,
        )
        return self._predict_from_features(features)

    def predict_batch(self, dataframe: Any) -> list[Prediction]:
        """Predict asset classes for a batch of buildings from a DataFrame.

        The DataFrame must contain a ``geometry`` column of Shapely polygons.
        Optional signal columns — ``zone_code``, ``osm_tags``,
        ``anzsic_divisions`` — are used when present.

        Args:
            dataframe: ``pandas.DataFrame`` with at least a ``geometry`` column.

        Returns:
            List of :class:`~footprint_ml.types.Prediction` in the same row
            order as the input DataFrame.
        """
        try:
            import pandas  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "predict_batch() requires pandas. Install with: pip install footprint-ml[train]"
            ) from exc

        results: list[Prediction] = []
        for _, row in dataframe.iterrows():
            geom = row["geometry"]
            zone = row.get("zone_code") if "zone_code" in dataframe.columns else None
            osm = row.get("osm_tags") if "osm_tags" in dataframe.columns else None
            anzsic = (
                row.get("anzsic_divisions") if "anzsic_divisions" in dataframe.columns else None
            )

            # pandas may return NaN for missing optional fields — normalise to None
            zone = None if _is_missing(zone) else zone
            osm = None if _is_missing(osm) else osm
            anzsic = None if _is_missing(anzsic) else anzsic

            results.append(
                self.predict(
                    geometry=geom,
                    zone_code=zone,
                    osm_tags=osm,
                    anzsic_divisions=anzsic,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        """Version string of the loaded model artifact."""
        return self._artifact.version

    @property
    def asset_classes(self) -> list[str]:
        """Ordered list of asset class labels the model was trained on."""
        return self._artifact.asset_classes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_from_features(self, features: dict[str, float]) -> Prediction:
        """Run inference on a pre-extracted feature dict."""
        pipeline = self._artifact.pipeline
        feature_names = self._artifact.feature_names

        # Build input row in the column order the model expects
        row = np.array(
            [features.get(name, float("nan")) for name in feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        proba: np.ndarray = pipeline.predict_proba(row)[0]
        classes: list[str] = list(pipeline.classes_)

        top_idx = int(np.argmax(proba))
        asset_class = classes[top_idx]
        confidence = float(proba[top_idx])
        probabilities = {cls: float(p) for cls, p in zip(classes, proba, strict=True)}

        return Prediction(
            asset_class=asset_class,
            confidence=confidence,
            probabilities=probabilities,
            model_version=self._artifact.version,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_missing(val: Any) -> bool:
    """Return True if *val* is None or a pandas/numpy NA sentinel."""
    if val is None:
        return True
    try:
        import pandas as pd

        if pd.isna(val):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    return False
