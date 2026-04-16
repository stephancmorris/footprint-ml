"""Example: integrating footprint-ml into Pulse via from_pulse_signals().

Shows how Pulse's data-engine should call footprint-ml using the compat
helper — keeping all footprint-ml API knowledge in one place.

This is the pattern to use in Pulse's data-engine/scoring/classifier.py.
footprint-ml must be installed separately:

    pip install footprint-ml

No Pulse imports here — this file is intentionally self-contained so it
can be run outside the Pulse repo for testing.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from footprint_ml import FootprintClassifier
from footprint_ml._compat import from_pulse_signals, to_pulse_result

# ---------------------------------------------------------------------------
# Simulate the dict that Pulse's enrichment pipeline produces
# ---------------------------------------------------------------------------
pulse_signals = {
    # Core — always present
    "geometry": Polygon([
        (150.8720, -33.8650),
        (150.8740, -33.8650),
        (150.8740, -33.8660),
        (150.8720, -33.8660),
    ]),

    # Enrichment signals — present when available
    "zone_code": "IN1",
    "osm_tags": {
        "building": "warehouse",
        "source": "cadastre",
    },
    "anzsic_divisions": ["I"],      # Transport, Postal and Warehousing

    # Pulse internal fields — passed through unchanged to the output
    "gnaf_pid": "GANSW703212938",
    "property_id": "9a3f1c2d-...",
    "confidence_source": "osm",
}

# ---------------------------------------------------------------------------
# 1. Convert Pulse signals → footprint-ml kwargs
# ---------------------------------------------------------------------------
predict_kwargs = from_pulse_signals(pulse_signals)

# ---------------------------------------------------------------------------
# 2. Classify
# ---------------------------------------------------------------------------
clf = FootprintClassifier()
prediction = clf.predict(**predict_kwargs)

# ---------------------------------------------------------------------------
# 3. Convert result back to Pulse-shaped output
# ---------------------------------------------------------------------------
result = to_pulse_result(prediction, signals=pulse_signals)

print("=== Pulse-shaped result ===")
for key, value in result.items():
    if key == "probabilities":
        print(f"  {key}:")
        for cls, prob in sorted(value.items(), key=lambda x: -x[1])[:4]:
            print(f"      {cls:<20} {prob:.3f}")
    else:
        print(f"  {key:<20} {value}")

# ---------------------------------------------------------------------------
# Pattern for Pulse's rules-based fallback
# ---------------------------------------------------------------------------
print()
print("=== With rules-based fallback (Pulse pattern) ===")

CONFIDENCE_THRESHOLD = 0.40  # below this, fall back to rules-based result


def classify_with_fallback(signals: dict, fallback_class: str = "other_commercial") -> dict:
    """Classify using ML; fall back to rules-based result if confidence is low."""
    try:
        kwargs = from_pulse_signals(signals)
        pred = clf.predict(**kwargs)
        if pred.confidence >= CONFIDENCE_THRESHOLD:
            return to_pulse_result(pred, signals=signals)
    except Exception:
        pass  # log in production
    return {
        "asset_class": fallback_class,
        "confidence": 0.0,
        "probabilities": {},
        "model_version": "rules_based",
        "gnaf_pid": signals.get("gnaf_pid"),
    }


output = classify_with_fallback(pulse_signals)
print(f"  asset_class : {output['asset_class']}")
print(f"  confidence  : {output['confidence']:.2%}")
print(f"  model       : {output['model_version']}")
