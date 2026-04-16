"""Pulse compatibility helpers.

``from_pulse_signals()`` converts the dict-of-signals that Pulse's
data-engine passes around into the keyword arguments that
:func:`~footprint_ml.features.extract_features` and
:class:`~footprint_ml.classifier.FootprintClassifier` expect.

This keeps the translation logic in one place so Pulse's classifier wrapper
never needs to know about footprint-ml's internal conventions.

Pulse signal dict shape (all keys optional except ``geometry``)::

    {
        "geometry":          <shapely Polygon in WGS84>,
        "zone_code":         "IND",           # str | None
        "osm_tags":          {"building": "warehouse", ...},  # dict | None
        "anzsic_divisions":  ["F", "I"],      # list[str] | None
        "confidence_source": "osm",           # ignored by footprint-ml
        "gnaf_pid":          "GANSW123",      # ignored by footprint-ml
    }
"""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon


# Keys that Pulse includes but footprint-ml has no use for
_IGNORED_PULSE_KEYS: frozenset[str] = frozenset(
    ["confidence_source", "gnaf_pid", "property_id", "lot_plan", "address_id"]
)


def from_pulse_signals(signals: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pulse signal dict into footprint-ml keyword arguments.

    Args:
        signals: Dict produced by Pulse's data-engine enrichment pipeline.
            Must contain a ``"geometry"`` key with a Shapely ``Polygon``.

    Returns:
        Dict of keyword arguments suitable for passing directly to
        :meth:`~footprint_ml.classifier.FootprintClassifier.predict` via
        ``**``, e.g.::

            clf.predict(**from_pulse_signals(signals))

    Raises:
        KeyError: If ``"geometry"`` is missing from *signals*.
        TypeError: If ``"geometry"`` is not a Shapely ``Polygon``.
    """
    if "geometry" not in signals:
        raise KeyError("Pulse signals dict must contain a 'geometry' key.")

    geometry = signals["geometry"]
    if not isinstance(geometry, Polygon):
        raise TypeError(
            f"Expected geometry to be a shapely Polygon, got {type(geometry).__name__}."
        )

    return {
        "geometry": geometry,
        "zone_code": signals.get("zone_code"),
        "osm_tags": signals.get("osm_tags"),
        "anzsic_divisions": signals.get("anzsic_divisions"),
    }


def to_pulse_result(prediction: Any, signals: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a :class:`~footprint_ml.types.Prediction` back to a Pulse-shaped result dict.

    Merges prediction fields with any pass-through keys from *signals*
    (e.g. ``gnaf_pid``, ``property_id``) so the result can be written
    directly to Pulse's scoring output.

    Args:
        prediction: A :class:`~footprint_ml.types.Prediction` instance.
        signals: Original Pulse signal dict (optional). Pass-through keys
            are copied to the result unchanged.

    Returns:
        Dict with ``asset_class``, ``confidence``, ``probabilities``,
        ``model_version``, plus any pass-through keys from *signals*.
    """
    result: dict[str, Any] = {
        "asset_class": prediction.asset_class,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "model_version": prediction.model_version,
    }

    if signals is not None:
        for key in _IGNORED_PULSE_KEYS:
            if key in signals:
                result[key] = signals[key]

    return result
