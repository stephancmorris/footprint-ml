"""FootprintTrainer: fit, evaluate, and save footprint classification models.

Training pipeline:
  1. Accept a pandas DataFrame with a ``geometry`` column (Shapely polygons)
     and a label column (asset class strings).
  2. Extract features for every row via ``extract_features()``.
  3. Fit a ``HistGradientBoostingClassifier`` wrapped in
     ``CalibratedClassifierCV`` for well-calibrated probabilities.
  4. Evaluate with stratified k-fold cross-validation.
  5. Save the fitted pipeline + ``meta.json`` to a directory suitable for
     ``model_registry.load_from_path()``.

The trainer is intentionally decoupled from the classifier — it only produces
artifacts that the registry and classifier know how to load.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from footprint_ml.features import FEATURE_NAMES, extract_features
from footprint_ml.types import AssetClass

logger = logging.getLogger(__name__)

# Default hyperparameters — tuned for a mixed-feature commercial property dataset
_DEFAULT_HGBC_PARAMS: dict[str, Any] = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 42,
}

_MODEL_FILENAME = "model.joblib"
_META_FILENAME = "meta.json"


class FootprintTrainer:
    """Fit, evaluate, and save a footprint classification model.

    Usage::

        trainer = FootprintTrainer()
        trainer.fit(df, label_column="asset_class")
        metrics = trainer.evaluate(df, label_column="asset_class")
        print(metrics["macro_f1"])
        trainer.save("models/au_commercial_v1")
    """

    def __init__(
        self,
        *,
        hgbc_params: dict[str, Any] | None = None,
        cv_folds: int = 5,
        calibration_method: str = "isotonic",
        version: str = "custom_v1",
    ) -> None:
        """
        Args:
            hgbc_params: Override default HistGradientBoostingClassifier params.
            cv_folds: Number of stratified folds for cross-validation.
            calibration_method: ``"isotonic"`` or ``"sigmoid"``.
            version: Version string written to ``meta.json``.
        """
        self._hgbc_params = {**_DEFAULT_HGBC_PARAMS, **(hgbc_params or {})}
        self._cv_folds = cv_folds
        self._calibration_method = calibration_method
        self._version = version

        self._pipeline: Pipeline | None = None
        self._label_classes: list[str] = []
        self._feature_names: list[str] = list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: Any,
        *,
        label_column: str = "asset_class",
        geometry_column: str = "geometry",
    ) -> FootprintTrainer:
        """Fit the model on *df*.

        Args:
            df: ``pandas.DataFrame`` with at least *geometry_column* and
                *label_column* columns. Optional signal columns
                ``zone_code``, ``osm_tags``, ``anzsic_divisions`` are used
                when present.
            label_column: Name of the target column.
            geometry_column: Name of the column holding Shapely polygons.

        Returns:
            ``self`` for chaining.
        """
        _require_pandas()

        df = df.copy()
        _validate_df(df, label_column, geometry_column)

        logger.info("Extracting features for %d rows…", len(df))
        X, y = self._build_Xy(df, label_column=label_column, geometry_column=geometry_column)

        self._label_classes = sorted(df[label_column].unique().tolist())

        logger.info(
            "Fitting HGBC + calibration on %d samples, %d classes…",
            len(y),
            len(self._label_classes),
        )
        self._pipeline = _build_pipeline(
            self._hgbc_params,
            calibration_method=self._calibration_method,
            cv=min(self._cv_folds, _min_class_count(y)),
        )
        self._pipeline.fit(X, y)
        logger.info("Training complete.")
        return self

    def evaluate(
        self,
        df: Any,
        *,
        label_column: str = "asset_class",
        geometry_column: str = "geometry",
        cv_folds: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate the model with stratified k-fold cross-validation.

        Fits *and* evaluates fresh folds on *df* (does not use the already-
        fitted pipeline, so results are unbiased).

        Args:
            df: Labelled DataFrame.
            label_column: Target column name.
            geometry_column: Geometry column name.
            cv_folds: Override the instance's ``cv_folds``. Defaults to
                ``self._cv_folds``.

        Returns:
            Dict with keys:
            - ``"macro_f1"`` — mean macro-F1 across folds
            - ``"per_class_f1"`` — dict of class → mean F1
            - ``"cv_scores"`` — raw array of per-fold macro-F1 scores
            - ``"classification_report"`` — sklearn text report on full set
        """
        _require_pandas()
        _validate_df(df, label_column, geometry_column)

        X, y = self._build_Xy(df, label_column=label_column, geometry_column=geometry_column)
        n_folds = cv_folds or self._cv_folds
        n_folds = min(n_folds, _min_class_count(y))

        pipeline = _build_pipeline(
            self._hgbc_params,
            calibration_method=self._calibration_method,
            cv=n_folds,
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=skf,
            scoring="f1_macro",
            return_train_score=False,
        )

        # Full-set classification report (in-sample — for diagnostics only)
        pipeline_full = _build_pipeline(
            self._hgbc_params,
            calibration_method=self._calibration_method,
            cv=n_folds,
        )
        pipeline_full.fit(X, y)
        y_pred = pipeline_full.predict(X)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        # Use classes present in y, not self._label_classes (evaluate may run without fit)
        present_classes = sorted(np.unique(y).tolist())
        per_class_f1 = {
            cls: float(report[cls]["f1-score"]) for cls in present_classes if cls in report
        }

        return {
            "macro_f1": float(np.mean(cv_results["test_score"])),
            "per_class_f1": per_class_f1,
            "cv_scores": cv_results["test_score"].tolist(),
            "classification_report": report,
        }

    def save(
        self,
        output_dir: str | Path,
        *,
        training_date: str | None = None,
    ) -> Path:
        """Save the fitted pipeline and ``meta.json`` to *output_dir*.

        Args:
            output_dir: Directory to write artifacts. Created if absent.
            training_date: ISO date string for ``meta.json``. Defaults to now.

        Returns:
            Path to the output directory.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before save().")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_path = out / _MODEL_FILENAME
        joblib.dump(self._pipeline, model_path)

        meta: dict[str, Any] = {
            "version": self._version,
            "feature_names": self._feature_names,
            "asset_classes": self._label_classes,
            "training_date": training_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "hgbc_params": self._hgbc_params,
            "cv_folds": self._cv_folds,
            "calibration_method": self._calibration_method,
        }
        (out / _META_FILENAME).write_text(json.dumps(meta, indent=2))

        logger.info("Saved model artifact to %s", out)
        return out

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    @property
    def classes(self) -> list[str]:
        """Asset class labels seen during training (sorted)."""
        return list(self._label_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_Xy(
        self,
        df: Any,
        *,
        label_column: str,
        geometry_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from every row and return (X, y) arrays."""
        rows: list[list[float]] = []
        for _, row in df.iterrows():
            geom = row[geometry_column]
            zone = _opt(row, "zone_code")
            osm = _opt(row, "osm_tags")
            anzsic = _opt(row, "anzsic_divisions")
            feats = extract_features(
                geom,
                zone_code=zone,
                osm_tags=osm,
                anzsic_divisions=anzsic,
            )
            rows.append([feats[name] for name in self._feature_names])

        X = np.array(rows, dtype=np.float64)
        y = np.array(df[label_column].tolist())
        return X, y


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FootprintTrainer requires pandas. Install with: pip install footprint-ml[train]"
        ) from exc


def _validate_df(df: Any, label_column: str, geometry_column: str) -> None:
    missing = [c for c in (label_column, geometry_column) if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("DataFrame must not be empty.")

    invalid = [v for v in df[label_column].unique() if v not in AssetClass._value2member_map_]
    if invalid:
        raise ValueError(
            f"Unknown asset class labels: {invalid}. "
            f"Valid values: {list(AssetClass._value2member_map_)}"
        )


def _min_class_count(y: np.ndarray) -> int:
    """Return the smallest per-class sample count — used to cap CV folds."""
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _build_pipeline(
    hgbc_params: dict[str, Any],
    *,
    calibration_method: str,
    cv: int,
) -> Pipeline:
    """Construct a fresh unfitted HGBC + calibration pipeline."""
    base = HistGradientBoostingClassifier(**hgbc_params)
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method=calibration_method,
        cv=cv,
    )
    return Pipeline([("clf", calibrated)])


def _opt(row: Any, key: str) -> Any:
    """Return ``row[key]`` if the key exists and value is not NA, else None."""
    if key not in row.index:
        return None
    val = row[key]
    try:
        import pandas as pd

        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val
