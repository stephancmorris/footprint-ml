"""Example: basic building footprint classification.

Demonstrates the minimal usage of FootprintClassifier — load the bundled
model and classify a single building polygon.

Run from the repo root after installing the package:

    pip install -e ".[train]"
    python examples/basic_classification.py

Note: This example requires a bundled model artifact in
src/footprint_ml/models/. Run scripts/train_bundled_model.py first,
or pass a model_path to FootprintClassifier.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from footprint_ml import FootprintClassifier

# ---------------------------------------------------------------------------
# 1. Define a building footprint polygon (WGS84 coordinates)
# ---------------------------------------------------------------------------
# This polygon approximates a large rectangular building in western Sydney —
# the kind of geometry typical for a warehouse or distribution centre.

warehouse_polygon = Polygon([
    (150.8720, -33.8650),
    (150.8740, -33.8650),
    (150.8740, -33.8660),
    (150.8720, -33.8660),
])

# ---------------------------------------------------------------------------
# 2. Classify with no optional signals (geometry only)
# ---------------------------------------------------------------------------
clf = FootprintClassifier()

pred = clf.predict(geometry=warehouse_polygon)

print("=== Geometry-only prediction ===")
print(f"Asset class : {pred.asset_class}")
print(f"Confidence  : {pred.confidence:.2%}")
print(f"Model       : {pred.model_version}")
print()

# ---------------------------------------------------------------------------
# 3. Enrich with optional signals for better accuracy
# ---------------------------------------------------------------------------
pred_enriched = clf.predict(
    geometry=warehouse_polygon,
    zone_code="IN1",                         # NSW IN1 General Industrial zone
    osm_tags={"building": "warehouse"},
    anzsic_divisions=["I"],                  # Transport, Postal and Warehousing
)

print("=== Enriched prediction ===")
print(f"Asset class : {pred_enriched.asset_class}")
print(f"Confidence  : {pred_enriched.confidence:.2%}")
print()

# ---------------------------------------------------------------------------
# 4. Inspect the full probability distribution
# ---------------------------------------------------------------------------
print("=== Full probability distribution ===")
for cls, prob in sorted(pred_enriched.probabilities.items(), key=lambda x: -x[1]):
    bar = "█" * int(prob * 30)
    print(f"  {cls:<20} {prob:.3f}  {bar}")
