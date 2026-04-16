"""Export labelled training data from Pulse DB to CSV/Parquet.

Usage (from repo root, with Pulse DB credentials in environment):

    python scripts/export_training_data.py \\
        --output data/training_data.parquet \\
        --min-area 200 \\
        --limit 100000

The exported file has the following columns:
    geometry          Shapely polygon (WGS84) — stored as WKT in CSV, native in Parquet
    asset_class       Target label (must be a valid AssetClass value)
    zone_code         Optional zoning code string
    osm_tags          Optional dict of OSM key→value tags (JSON in CSV)
    anzsic_divisions  Optional list of ANZSIC division letters (JSON in CSV)
    source            Origin of the label: "manual", "osm_tag", "pulse_rules"

Requires:
    pip install footprint-ml[train] psycopg2-binary sqlalchemy geopandas

Environment variables:
    PULSE_DB_URL  SQLAlchemy connection string, e.g.
                  postgresql://user:pass@host:5432/pulse
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# SQL to extract labelled buildings from Pulse's property_footprint table.
# Adjust schema/table names to match your Pulse DB.
_EXTRACT_SQL = """
SELECT
    ST_AsText(ST_Transform(f.geometry, 4326))           AS geometry_wkt,
    p.asset_class,
    p.zone_code,
    p.osm_tags::text                                    AS osm_tags_json,
    p.anzsic_divisions::text                            AS anzsic_divisions_json,
    p.label_source                                      AS source
FROM property_footprint f
JOIN property p ON f.property_id = p.id
WHERE
    p.asset_class IS NOT NULL
    AND ST_Area(ST_Transform(f.geometry, 3857)) >= :min_area
    AND f.geometry IS NOT NULL
ORDER BY p.id
{limit_clause}
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export labelled training data from Pulse DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="data/training_data.parquet",
        help="Output file path (.parquet or .csv).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=200.0,
        help="Minimum building area in m² (filters out slivers).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum rows to export (no limit if omitted).",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("PULSE_DB_URL"),
        help="SQLAlchemy DB URL (defaults to $PULSE_DB_URL).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.db_url:
        logger.error(
            "No DB URL provided. Set $PULSE_DB_URL or pass --db-url."
        )
        return 1

    try:
        import geopandas as gpd
        import pandas as pd
        import sqlalchemy as sa
        from shapely import wkt as shapely_wkt
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Install with: pip install geopandas sqlalchemy psycopg2-binary")
        return 1

    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    sql = _EXTRACT_SQL.format(limit_clause=limit_clause)

    logger.info("Connecting to Pulse DB…")
    engine = sa.create_engine(args.db_url)

    logger.info("Executing export query (min_area=%.0f m²)…", args.min_area)
    with engine.connect() as conn:
        df = pd.read_sql(
            sa.text(sql),
            conn,
            params={"min_area": args.min_area},
        )

    logger.info("Fetched %d rows.", len(df))

    # Parse geometry WKT → Shapely polygons
    df["geometry"] = df["geometry_wkt"].apply(shapely_wkt.loads)
    df = df.drop(columns=["geometry_wkt"])

    # Parse JSON columns
    df["osm_tags"] = df["osm_tags_json"].apply(
        lambda v: json.loads(v) if v else None
    )
    df["anzsic_divisions"] = df["anzsic_divisions_json"].apply(
        lambda v: json.loads(v) if v else None
    )
    df = df.drop(columns=["osm_tags_json", "anzsic_divisions_json"])

    # Validate asset class labels
    from footprint_ml.types import AssetClass
    valid = set(AssetClass._value2member_map_)
    bad = df[~df["asset_class"].isin(valid)]
    if len(bad):
        logger.warning(
            "Dropping %d rows with unknown asset_class values: %s",
            len(bad),
            bad["asset_class"].unique().tolist(),
        )
        df = df[df["asset_class"].isin(valid)]

    logger.info("Writing %d labelled buildings to %s…", len(df), args.output)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".parquet":
        # Shapely geometry can't go directly into parquet — serialise as WKT
        df_out = df.copy()
        df_out["geometry"] = df_out["geometry"].apply(lambda g: g.wkt)
        df_out["osm_tags"] = df_out["osm_tags"].apply(
            lambda v: json.dumps(v) if v is not None else None
        )
        df_out["anzsic_divisions"] = df_out["anzsic_divisions"].apply(
            lambda v: json.dumps(v) if v is not None else None
        )
        df_out.to_parquet(out, index=False)
    else:
        df["osm_tags"] = df["osm_tags"].apply(
            lambda v: json.dumps(v) if v is not None else None
        )
        df["anzsic_divisions"] = df["anzsic_divisions"].apply(
            lambda v: json.dumps(v) if v is not None else None
        )
        df["geometry"] = df["geometry"].apply(lambda g: g.wkt)
        df.to_csv(out, index=False)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
