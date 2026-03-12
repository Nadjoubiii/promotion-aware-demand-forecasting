"""Build a processed training table from Favorita + optional external sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.data.external_sources import (
    build_weather_features_for_stores,
    load_local_events,
    load_store_coordinates,
    merge_external_sources,
)
from src.data.ingest_favorita import build_base_daily_sales, read_favorita_tables
from src.data.ingest_favorita import download_favorita_from_kaggle
from src.data.validation import (
    find_missing_dates_per_segment,
    find_negative_units,
    find_null_ids,
    find_stockout_candidates,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=settings.favorita_raw_dir, help="Directory containing Favorita CSV files.")
    parser.add_argument(
        "--external-dir",
        default=settings.external_data_dir,
        help="Directory containing optional external sources files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path(settings.processed_data_dir) / "training_table.parquet"),
        help="Output parquet path for merged training table.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download Favorita data from Kaggle if required files are missing.",
    )
    return parser


def run_pipeline(
    raw_dir: Path,
    external_dir: Path,
    output_path: Path,
    download_if_missing: bool = False,
) -> dict[str, int]:
    """Run ingestion + enrichment pipeline and return validation summary counts."""
    if download_if_missing:
        download_favorita_from_kaggle(
            raw_dir=raw_dir,
            competition_slug=settings.favorita_competition_slug,
            force=False,
        )

    tables = read_favorita_tables(raw_dir)
    base_df = build_base_daily_sales(tables)

    coords_path = external_dir / "store_city_coords.csv"
    events_path = external_dir / "local_events.csv"

    weather_df = pd.DataFrame()
    coords_df = load_store_coordinates(coords_path)
    if not coords_df.empty:
        start_date = base_df["date"].min().strftime("%Y-%m-%d")
        end_date = base_df["date"].max().strftime("%Y-%m-%d")
        weather_df = build_weather_features_for_stores(
            coords_df,
            start_date=start_date,
            end_date=end_date,
            timezone=settings.weather_timezone,
            timeout_seconds=settings.weather_request_timeout_seconds,
        )

    events_df = load_local_events(events_path)
    training_df = merge_external_sources(base_df, weather_df, events_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(output_path, index=False)

    validation_counts = {
        "rows": int(len(training_df)),
        "negative_units": int(len(find_negative_units(training_df))),
        "null_ids": int(len(find_null_ids(training_df))),
        "stockout_candidates": int(len(find_stockout_candidates(training_df))),
        "missing_dates": int(len(find_missing_dates_per_segment(training_df))),
    }

    return validation_counts


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = run_pipeline(
        raw_dir=Path(args.raw_dir),
        external_dir=Path(args.external_dir),
        output_path=Path(args.output_path),
        download_if_missing=args.download_if_missing,
    )

    print("Pipeline finished. Validation summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
