"""Train the bundled au_commercial_v1 model and write to src/footprint_ml/models/.

This script bootstraps training data from OSM Australia (Geofabrik PBF extract)
using buildings with unambiguous ``building=*`` tags, then trains and saves the
bundled model that ships in the PyPI wheel.

Usage (from repo root):

    python scripts/train_bundled_model.py

    # Use a pre-exported Pulse CSV/Parquet instead of OSM bootstrap:
    python scripts/train_bundled_model.py --training-data data/training_data.parquet

    # Dry-run: extract features and evaluate without saving
    python scripts/train_bundled_model.py --dry-run

Dependencies (beyond footprint-ml[train]):
    pip install osmium requests tqdm

OSM bootstrap label map (building=* → asset_class):
    warehouse       → warehouse
    industrial      → industrial
    retail          → retail
    supermarket     → retail
    shop            → retail
    office          → office
    hospital        → medical
    clinic          → medical
    hotel / motel   → hospitality
    school          → education
    university      → education
    college         → education
    kindergarten    → childcare
    commercial      → other_commercial
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Root of the repo (two levels up from this script)
_REPO_ROOT = Path(__file__).parent.parent
_BUNDLED_MODELS_DIR = _REPO_ROOT / "src" / "footprint_ml" / "models"

# OSM building=* → AssetClass mapping used for bootstrap labels
_OSM_BUILDING_LABEL_MAP: dict[str, str] = {
    "warehouse": "warehouse",
    "industrial": "industrial",
    "retail": "retail",
    "supermarket": "retail",
    "shop": "retail",
    "office": "office",
    "hospital": "medical",
    "clinic": "medical",
    "hotel": "hospitality",
    "motel": "hospitality",
    "school": "education",
    "university": "education",
    "college": "education",
    "kindergarten": "childcare",
    "commercial": "other_commercial",
}

# Minimum polygon area in m² — filters out mapping errors and tiny features
_MIN_AREA_M2 = 200.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bundled footprint-ml model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training-data",
        default=None,
        help=(
            "Path to a pre-exported CSV or Parquet training file. "
            "If omitted, uses OSM Australia bootstrap data."
        ),
    )
    parser.add_argument(
        "--osm-pbf",
        default=None,
        help=(
            "Path to an Australian OSM PBF extract (e.g. australia-oceania-latest.osm.pbf). "
            "Downloaded from Geofabrik if not provided and --training-data is not set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_BUNDLED_MODELS_DIR),
        help="Directory to write the trained model artifact.",
    )
    parser.add_argument(
        "--version",
        default="au_commercial_v1",
        help="Version string written to meta.json.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds for evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract features and evaluate but do not write model files.",
    )
    return parser.parse_args(argv)


def load_training_data(path: str) -> "Any":
    """Load a training CSV or Parquet into a DataFrame with Shapely geometry."""
    import json as _json

    import pandas as pd
    from shapely import wkt as shapely_wkt

    p = Path(path)
    logger.info("Loading training data from %s…", p)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    df["geometry"] = df["geometry"].apply(shapely_wkt.loads)

    if "osm_tags" in df.columns:
        df["osm_tags"] = df["osm_tags"].apply(
            lambda v: _json.loads(v) if isinstance(v, str) else None
        )
    if "anzsic_divisions" in df.columns:
        df["anzsic_divisions"] = df["anzsic_divisions"].apply(
            lambda v: _json.loads(v) if isinstance(v, str) else None
        )

    logger.info("Loaded %d rows.", len(df))
    return df


def bootstrap_from_osm(pbf_path: str | None) -> "Any":
    """Extract labelled buildings from an OSM PBF file.

    If *pbf_path* is None, attempts to download the Australia extract from
    Geofabrik (~700MB).  Requires ``osmium`` and optionally ``requests``.
    """
    try:
        import osmium  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "osmium is required for OSM bootstrap. Install with: pip install osmium"
        ) from exc

    import pandas as pd
    from shapely.geometry import Polygon

    if pbf_path is None:
        pbf_path = _download_au_pbf()

    logger.info("Parsing OSM buildings from %s…", pbf_path)

    class BuildingHandler(osmium.SimpleHandler):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.records: list[dict] = []

        def area(self, a: Any) -> None:  # type: ignore[override]
            building = a.tags.get("building")
            if building not in _OSM_BUILDING_LABEL_MAP:
                return
            try:
                wkb = osmium.geom.WKBFactory()
                poly = wkb.create_multipolygon(a)
                from shapely import wkb as shapely_wkb
                geom = shapely_wkb.loads(poly, hex=True)
                if geom.area == 0:
                    return
                # Take the largest polygon if MultiPolygon
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: g.area)
            except Exception:
                return

            self.records.append({
                "geometry": geom,
                "asset_class": _OSM_BUILDING_LABEL_MAP[building],
                "osm_tags": {"building": building},
            })

    handler = BuildingHandler()
    handler.apply_file(pbf_path, locations=True)

    df = pd.DataFrame(handler.records)
    logger.info("Extracted %d buildings from OSM.", len(df))
    return df


def _download_au_pbf() -> str:
    """Download the Geofabrik Australia PBF extract to a local cache."""
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "requests is required to download OSM data. Install with: pip install requests"
        ) from exc

    url = "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf"
    cache_dir = Path.home() / ".cache" / "footprint_ml" / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "australia-latest.osm.pbf"

    if dest.exists():
        logger.info("Using cached OSM PBF: %s", dest)
        return str(dest)

    logger.info("Downloading Australian OSM extract (~700MB) from Geofabrik…")
    logger.info("URL: %s", url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
    logger.info("Saved to %s", dest)
    return str(dest)


def filter_small_buildings(df: "Any") -> "Any":
    """Drop buildings smaller than _MIN_AREA_M2 in metric projection."""
    from footprint_ml.geometry import building_area_m2

    before = len(df)
    mask = df["geometry"].apply(lambda g: building_area_m2(g) >= _MIN_AREA_M2)
    df = df[mask].reset_index(drop=True)
    logger.info(
        "Filtered %d small buildings (< %.0f m²); %d remain.",
        before - len(df),
        _MIN_AREA_M2,
        len(df),
    )
    return df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import pandas as pd
    except ImportError as exc:
        logger.error("pandas is required: pip install footprint-ml[train]")
        return 1

    # ------------------------------------------------------------------
    # 1. Load or bootstrap training data
    # ------------------------------------------------------------------
    if args.training_data:
        df = load_training_data(args.training_data)
    else:
        logger.info("No --training-data supplied; bootstrapping from OSM…")
        df = bootstrap_from_osm(args.osm_pbf)

    df = filter_small_buildings(df)

    if len(df) < 100:
        logger.error("Too few training samples (%d). Need at least 100.", len(df))
        return 1

    # Class distribution summary
    logger.info("Class distribution:\n%s", df["asset_class"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 2. Fit
    # ------------------------------------------------------------------
    from footprint_ml.trainer import FootprintTrainer

    trainer = FootprintTrainer(version=args.version, cv_folds=args.cv_folds)

    logger.info("Fitting model…")
    trainer.fit(df, label_column="asset_class")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    logger.info("Evaluating with %d-fold CV…", args.cv_folds)
    metrics = trainer.evaluate(df, label_column="asset_class", cv_folds=args.cv_folds)

    logger.info("Macro F1 (CV): %.4f", metrics["macro_f1"])
    logger.info("Per-class F1:")
    for cls, f1 in sorted(metrics["per_class_f1"].items()):
        logger.info("  %-20s %.4f", cls, f1)

    # ------------------------------------------------------------------
    # 4. Save (unless dry-run)
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("--dry-run: skipping save.")
        return 0

    out_dir = trainer.save(args.output_dir)
    logger.info("Model artifact written to %s", out_dir)

    # Quick size check
    model_file = out_dir / "model.joblib"
    size_mb = model_file.stat().st_size / (1024 * 1024)
    logger.info("model.joblib size: %.2f MB", size_mb)
    if size_mb > 5.0:
        logger.warning(
            "Model exceeds 5MB target (%.2f MB). "
            "Consider reducing max_iter or max_depth.",
            size_mb,
        )

    return 0


# ---------------------------------------------------------------------------
# Type alias for annotations inside functions (avoids pandas import at module level)
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
