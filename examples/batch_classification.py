"""Example: batch classification from a DataFrame.

Demonstrates predict_batch() for classifying many buildings at once.

Run from the repo root after installing the package:

    pip install -e ".[train]"
    python examples/batch_classification.py
"""

from __future__ import annotations

import pandas as pd
from shapely.geometry import Polygon

from footprint_ml import FootprintClassifier


def make_polygon(lon: float, lat: float, size: float = 0.002) -> Polygon:
    return Polygon([
        (lon, lat), (lon + size, lat),
        (lon + size, lat + size), (lon, lat + size),
    ])


# ---------------------------------------------------------------------------
# 1. Build a sample DataFrame of buildings
# ---------------------------------------------------------------------------
buildings = pd.DataFrame([
    {
        "building_id": "WH-001",
        "geometry": make_polygon(150.872, -33.865, size=0.003),
        "zone_code": "IN1",
        "osm_tags": {"building": "warehouse"},
        "anzsic_divisions": ["I"],
    },
    {
        "building_id": "RE-042",
        "geometry": make_polygon(151.209, -33.868, size=0.001),
        "zone_code": "B2",
        "osm_tags": {"building": "retail", "shop": "supermarket"},
        "anzsic_divisions": ["G"],
    },
    {
        "building_id": "OF-117",
        "geometry": make_polygon(151.205, -33.870, size=0.0015),
        "zone_code": "B3",
        "osm_tags": {"building": "office"},
        "anzsic_divisions": ["M"],
    },
    {
        "building_id": "MED-008",
        "geometry": make_polygon(151.185, -33.875, size=0.002),
        "zone_code": "SP",
        "osm_tags": {"amenity": "hospital"},
        "anzsic_divisions": ["Q"],
    },
    {
        "building_id": "UNKNOWN",
        "geometry": make_polygon(151.000, -34.000, size=0.002),
        "zone_code": None,   # no zoning data
        "osm_tags": None,    # no OSM tags
        "anzsic_divisions": None,
    },
])

# ---------------------------------------------------------------------------
# 2. Run batch prediction
# ---------------------------------------------------------------------------
clf = FootprintClassifier()
predictions = clf.predict_batch(buildings)

# ---------------------------------------------------------------------------
# 3. Attach results back to the DataFrame
# ---------------------------------------------------------------------------
buildings["asset_class"] = [p.asset_class for p in predictions]
buildings["confidence"] = [p.confidence for p in predictions]

# ---------------------------------------------------------------------------
# 4. Display results
# ---------------------------------------------------------------------------
print("=== Batch classification results ===")
print()
for _, row in buildings.iterrows():
    print(f"  {row['building_id']:<12}  {row['asset_class']:<20}  {row['confidence']:.1%}")

print()
print(f"Classified {len(buildings)} buildings using model: {predictions[0].model_version}")
