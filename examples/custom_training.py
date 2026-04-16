"""Example: training a custom model with FootprintTrainer.

Demonstrates how to fit, evaluate, and save your own model from labelled
building data — for example, training on Pulse DB exports or a custom
dataset with local property types.

Run from the repo root:

    pip install -e ".[train]"
    python examples/custom_training.py
"""

from __future__ import annotations

import pandas as pd
from shapely.geometry import Polygon

from footprint_ml import FootprintClassifier
from footprint_ml.trainer import FootprintTrainer


def make_polygon(lon: float, lat: float, size: float = 0.002) -> Polygon:
    return Polygon([
        (lon, lat), (lon + size, lat),
        (lon + size, lat + size), (lon, lat + size),
    ])


# ---------------------------------------------------------------------------
# 1. Build a minimal labelled training DataFrame
#    In practice this comes from scripts/export_training_data.py or OSM.
# ---------------------------------------------------------------------------
CLASSES = ["warehouse", "industrial", "retail", "office", "medical"]
N_PER_CLASS = 10

rows = []
for i, cls in enumerate(CLASSES):
    for j in range(N_PER_CLASS):
        rows.append({
            "geometry": make_polygon(150.8 + i * 0.05 + j * 0.005, -33.8 + j * 0.003),
            "asset_class": cls,
            "zone_code": "IN1" if cls in ("warehouse", "industrial") else "B2",
            "osm_tags": {"building": cls},
            "anzsic_divisions": ["I"] if cls == "warehouse" else ["G"],
        })

df = pd.DataFrame(rows)
print(f"Training on {len(df)} labelled buildings across {len(CLASSES)} classes.")
print()

# ---------------------------------------------------------------------------
# 2. Fit
# ---------------------------------------------------------------------------
trainer = FootprintTrainer(
    version="my_custom_v1",
    cv_folds=2,                                  # use more folds on real data
    hgbc_params={"max_iter": 100, "max_depth": 5},
)
trainer.fit(df, label_column="asset_class")
print(f"Trained on classes: {trainer.classes}")
print()

# ---------------------------------------------------------------------------
# 3. Evaluate with cross-validation
# ---------------------------------------------------------------------------
metrics = trainer.evaluate(df, label_column="asset_class")
print(f"Cross-validated macro F1: {metrics['macro_f1']:.4f}")
print()
print("Per-class F1:")
for cls, f1 in sorted(metrics["per_class_f1"].items()):
    print(f"  {cls:<20} {f1:.4f}")
print()

# ---------------------------------------------------------------------------
# 4. Save the model artifact
# ---------------------------------------------------------------------------
out_dir = trainer.save("models/my_custom_v1")
print(f"Model saved to: {out_dir}")
print()

# ---------------------------------------------------------------------------
# 5. Load and use the saved model
# ---------------------------------------------------------------------------
clf = FootprintClassifier(model_path=out_dir)

test_polygon = make_polygon(150.872, -33.865, size=0.003)
pred = clf.predict(
    geometry=test_polygon,
    zone_code="IN1",
    osm_tags={"building": "warehouse"},
)

print("=== Test prediction with saved model ===")
print(f"Asset class : {pred.asset_class}")
print(f"Confidence  : {pred.confidence:.2%}")
print(f"Model       : {pred.model_version}")
