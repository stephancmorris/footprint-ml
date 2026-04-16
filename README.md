# footprint-ml

ML classifier for building footprint polygons → commercial property asset classes.

[![CI](https://github.com/stephancmorris/footprint-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/stephancmorris/footprint-ml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/footprint-ml)](https://pypi.org/project/footprint-ml/)
[![Python](https://img.shields.io/pypi/pyversions/footprint-ml)](https://pypi.org/project/footprint-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What it does

Given a building footprint polygon (plus optional enrichment signals), footprint-ml
predicts which commercial property asset class the building belongs to:

`industrial` · `warehouse` · `retail` · `office` · `medical` · `hospitality` ·
`education` · `childcare` · `mixed_use` · `other_commercial`

The bundled model (`au_commercial_v1`) was trained on Australian commercial buildings
using OSM tags and is suitable for zero-config usage. Custom models can be trained
and swapped in with one line.

## Installation

```bash
pip install footprint-ml
```

For training your own models:

```bash
pip install "footprint-ml[train]"
```

## Quick start

```python
from shapely.geometry import Polygon
from footprint_ml import FootprintClassifier

polygon = Polygon([
    (150.872, -33.865), (150.874, -33.865),
    (150.874, -33.866), (150.872, -33.866),
])

clf = FootprintClassifier()
pred = clf.predict(geometry=polygon)

print(pred.asset_class)    # "warehouse"
print(pred.confidence)     # 0.78
print(pred.model_version)  # "au_commercial_v1"
```

### Enriched prediction

Pass optional signals to improve accuracy:

```python
pred = clf.predict(
    geometry=polygon,
    zone_code="IN1",                        # planning zone code
    osm_tags={"building": "warehouse"},     # OpenStreetMap tags
    anzsic_divisions=["I"],                 # ABR/ANZSIC primary divisions
)
```

### Batch prediction

```python
import pandas as pd

# DataFrame must have a 'geometry' column (Shapely polygons)
# Optional columns: zone_code, osm_tags, anzsic_divisions
predictions = clf.predict_batch(df)

df["asset_class"] = [p.asset_class for p in predictions]
df["confidence"]  = [p.confidence  for p in predictions]
```

## Training a custom model

```python
from footprint_ml.trainer import FootprintTrainer

trainer = FootprintTrainer(version="my_model_v1")
trainer.fit(df, label_column="asset_class")   # df has 'geometry' + label column

metrics = trainer.evaluate(df, label_column="asset_class")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

trainer.save("models/my_model_v1")

# Load it back
clf = FootprintClassifier(model_path="models/my_model_v1")
```

### Exporting training data from Pulse

```bash
PULSE_DB_URL=postgresql://user:pass@host/pulse \
  python scripts/export_training_data.py --output data/training.parquet
```

### Training the bundled model

```bash
# From OSM Australia bootstrap (downloads ~700MB PBF on first run)
python scripts/train_bundled_model.py

# From a pre-exported Pulse dataset
python scripts/train_bundled_model.py --training-data data/training.parquet
```

## Pulse integration

footprint-ml is designed to replace the rules-based classifier in Pulse's
data-engine. Use the `_compat` helpers to keep the integration clean:

```python
from footprint_ml import FootprintClassifier
from footprint_ml._compat import from_pulse_signals, to_pulse_result

clf = FootprintClassifier()

def classify(signals: dict) -> dict:
    pred = clf.predict(**from_pulse_signals(signals))
    return to_pulse_result(pred, signals=signals)
```

See [`examples/pulse_integration.py`](examples/pulse_integration.py) for a
complete example including a rules-based fallback pattern.

## API reference

### `FootprintClassifier`

| Method | Description |
|---|---|
| `FootprintClassifier(model_path=None, model_version=None)` | Load bundled model, or from path/GitHub release |
| `predict(geometry, *, zone_code, osm_tags, anzsic_divisions, src_crs)` | Classify a single polygon |
| `predict_batch(dataframe)` | Classify a DataFrame of polygons |
| `model_version` | Version string of the loaded model |
| `asset_classes` | Ordered list of class labels the model knows |

### `Prediction`

```python
@dataclass(frozen=True)
class Prediction:
    asset_class: str            # e.g. "warehouse"
    confidence: float           # probability of the top class
    probabilities: dict[str, float]  # full distribution over all classes
    model_version: str
```

### `FootprintTrainer`

| Method | Description |
|---|---|
| `fit(df, label_column, geometry_column)` | Train on a labelled DataFrame |
| `evaluate(df, label_column, cv_folds)` | Stratified k-fold CV evaluation |
| `save(output_dir, training_date)` | Write `model.joblib` + `meta.json` |

### `AssetClass`

```python
from footprint_ml.types import AssetClass

AssetClass.WAREHOUSE   # "warehouse"
AssetClass("retail")   # AssetClass.RETAIL
```

## Loading models

```python
from footprint_ml import FootprintClassifier

# Bundled model (default)
clf = FootprintClassifier()

# From a local directory
clf = FootprintClassifier(model_path="models/my_model_v1")

# From a GitHub Release
clf = FootprintClassifier(model_version="au_commercial_v2")
```

## Feature set

| Feature | Source | Always present |
|---|---|---|
| `building_area_m2` | Geometry | ✓ |
| `building_perimeter_m` | Geometry | ✓ |
| `building_compactness` | Geometry | ✓ |
| `aspect_ratio` | Geometry | ✓ |
| `bbox_length_m` | Geometry | ✓ |
| `bbox_width_m` | Geometry | ✓ |
| `edge_count` | Geometry | ✓ |
| `log_area` | Geometry | ✓ |
| `elongation` | Geometry | ✓ |
| `zone_code_encoded` | Planning zone | Optional |
| `osm_amenity_encoded` | OSM tags | Optional |
| `osm_building_use_encoded` | OSM tags | Optional |
| `has_osm_shop` | OSM tags | Optional |
| `has_osm_office` | OSM tags | Optional |
| `anzsic_primary_division_encoded` | ABR/ANZSIC | Optional |
| `anzsic_count` | ABR/ANZSIC | Optional |

Geometry is computed in the UTM zone of the polygon centroid (auto-detected).
Missing optional features are `NaN` — the underlying `HistGradientBoostingClassifier`
handles them natively without imputation.

## Requirements

- Python 3.10+
- scikit-learn ≥ 1.3
- shapely ≥ 2.0
- pyproj ≥ 3.6
- joblib ≥ 1.3
- numpy ≥ 1.24

Training extras (`pip install footprint-ml[train]`): pandas ≥ 2.0, matplotlib ≥ 3.7

## Contributing

```bash
git clone https://github.com/stephancmorris/footprint-ml
cd footprint-ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pytest tests/
```

Lint and type-check:

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/footprint_ml/
```

## License

MIT — see [LICENSE](LICENSE).
