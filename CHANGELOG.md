# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-16

### Added
- `FootprintClassifier` — classify building footprint polygons into 10 commercial
  property asset classes using a bundled `HistGradientBoostingClassifier` with
  calibrated probabilities
- `FootprintTrainer` — fit, evaluate (stratified k-fold CV), and save custom models
- `AssetClass` enum and `Prediction` dataclass matching Pulse's DB enum
- `geometry.compute_metrics()` — 9 polygon metrics (area, perimeter, compactness,
  aspect ratio, bounding box dims, edge count, log area, elongation) with
  auto-detected UTM projection
- `features.extract_features()` — 16-feature flat dict from polygon + optional
  zone/OSM/ANZSIC signals; missing signals encoded as NaN (HGBC handles natively)
- `ZoneEncoder` — ordinal encoder for zone codes with unknown-category handling
- `model_registry.load()` — unified loader supporting bundled model, local path,
  and GitHub Releases download with `~/.cache` persistence
- `_compat.from_pulse_signals()` / `to_pulse_result()` — Pulse data-engine integration
- `scripts/export_training_data.py` — SQL export from Pulse DB to CSV/Parquet
- `scripts/train_bundled_model.py` — bootstrap training from OSM PBF or Pulse export,
  with size check against 5MB wheel limit
- CI workflow: ruff + mypy + pytest matrix across Python 3.10/3.11/3.12
- Release workflow: OIDC trusted publisher to PyPI + GitHub Release with changelog notes

[Unreleased]: https://github.com/stephancmorris/footprint-ml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/stephancmorris/footprint-ml/releases/tag/v0.1.0
