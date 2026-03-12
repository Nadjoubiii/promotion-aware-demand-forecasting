"""Load and combine core Favorita tables into a base daily sales table."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import pandas as pd

REQUIRED_FILES = {
    "sales": "train.csv",
    "stores": "stores.csv",
    "products": "items.csv",
    "holidays": "holidays_events.csv",
    "oil": "oil.csv",
    "transactions": "transactions.csv",
}
DEFAULT_COMPETITION_SLUG = "favorita-grocery-sales-forecasting"


def _missing_required_files(raw_dir: Path) -> list[str]:
    """Return missing required CSV filenames in the raw directory."""
    return [name for name in REQUIRED_FILES.values() if not (raw_dir / name).exists()]


def download_favorita_from_kaggle(
    raw_dir: Path,
    competition_slug: str = DEFAULT_COMPETITION_SLUG,
    force: bool = False,
) -> list[str]:
    """Download and unzip Favorita files into ``raw_dir`` using Kaggle CLI.

    Requires:
    - Kaggle CLI installed (``pip install kaggle``)
    - Kaggle credentials configured (``~/.kaggle/kaggle.json``)
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing_before = _missing_required_files(raw_dir)
    if not missing_before and not force:
        return []

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not in PATH. Install with: pip install kaggle"
        )

    archive_path = raw_dir / f"{competition_slug}.zip"
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition_slug,
        "-p",
        str(raw_dir),
        "-f",
        "*.csv.zip",
    ]

    # Some Kaggle competitions package all files in the main zip, so fall back when needed.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        fallback_cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_slug,
            "-p",
            str(raw_dir),
        ]
        fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
        if fallback_result.returncode != 0:
            message = fallback_result.stderr.strip() or fallback_result.stdout.strip() or "Unknown Kaggle error"
            raise RuntimeError(
                "Failed to download Favorita data from Kaggle. "
                "Make sure you accepted competition rules and configured Kaggle API credentials. "
                f"Details: {message}"
            )

    # Unzip any zip files dumped by Kaggle into raw_dir.
    for zip_path in raw_dir.glob("*.zip"):
        shutil.unpack_archive(str(zip_path), str(raw_dir))

    # Some dumps include nested zip files; unpack one more level if present.
    for nested_zip in raw_dir.glob("*.csv.zip"):
        shutil.unpack_archive(str(nested_zip), str(raw_dir))

    # Remove top-level archive if present and force was requested, to avoid stale duplicates.
    if force and archive_path.exists():
        archive_path.unlink(missing_ok=True)

    missing_after = _missing_required_files(raw_dir)
    if missing_after:
        raise FileNotFoundError(
            "Downloaded dataset is incomplete. Missing required files: " + ", ".join(sorted(missing_after))
        )

    return missing_before


def read_favorita_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read required Favorita files from a raw directory."""
    missing = _missing_required_files(raw_dir)
    if missing:
        raise FileNotFoundError(
            "Missing required Favorita files in raw directory: " + ", ".join(sorted(missing))
        )

    tables = {
        "sales": pd.read_csv(
            raw_dir / REQUIRED_FILES["sales"],
            usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
            parse_dates=["date"],
            dtype={
                "store_nbr": "int16",
                "item_nbr": "int32",
                "unit_sales": "float32",
                "onpromotion": "string",
            },
            low_memory=False,
        ),
        "stores": pd.read_csv(
            raw_dir / REQUIRED_FILES["stores"],
            usecols=["store_nbr", "cluster"],
            dtype={"store_nbr": "int16", "cluster": "int16"},
        ),
        "products": pd.read_csv(
            raw_dir / REQUIRED_FILES["products"],
            usecols=["item_nbr", "class", "perishable"],
            dtype={"item_nbr": "int32", "class": "int32", "perishable": "int8"},
        ),
        "holidays": pd.read_csv(
            raw_dir / REQUIRED_FILES["holidays"],
            usecols=["date", "type", "transferred"],
            parse_dates=["date"],
        ),
        "oil": pd.read_csv(
            raw_dir / REQUIRED_FILES["oil"],
            usecols=["date", "dcoilwtico"],
            parse_dates=["date"],
        ),
        "transactions": pd.read_csv(
            raw_dir / REQUIRED_FILES["transactions"],
            usecols=["date", "store_nbr", "transactions"],
            parse_dates=["date"],
            dtype={"store_nbr": "int16", "transactions": "float32"},
        ),
    }

    # Favorita `onpromotion` may contain mixed values across file chunks.
    tables["sales"]["onpromotion"] = (
        tables["sales"]["onpromotion"].str.lower().map({"true": True, "false": False}).fillna(False)
    )
    return tables


def _build_holiday_daily_features(holidays: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-date holiday/event indicators."""
    work = holidays.copy()
    work["transferred"] = work["transferred"].fillna(False)
    work["is_holiday_event"] = work["type"].isin(["Holiday", "Event"]) & (~work["transferred"])

    daily = (
        work.groupby("date", as_index=False)
        .agg(
            is_holiday_event=("is_holiday_event", "max"),
        )
    )
    return daily


def build_base_daily_sales(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge core tables into a store-product-date level dataset."""
    sales = tables["sales"].copy()
    sales = sales.rename(
        columns={
            "store_nbr": "store_id",
            "item_nbr": "product_id",
            "unit_sales": "units",
            "onpromotion": "on_promotion",
        }
    )

    stores = tables["stores"].rename(columns={"store_nbr": "store_id"})
    products = tables["products"].rename(columns={"item_nbr": "product_id", "class": "product_class"})

    holiday_daily = _build_holiday_daily_features(tables["holidays"])

    oil = tables["oil"].rename(columns={"dcoilwtico": "oil_price"}).copy()
    oil["oil_price"] = pd.to_numeric(oil["oil_price"], errors="coerce").ffill()

    transactions = tables["transactions"].rename(
        columns={"store_nbr": "store_id", "transactions": "store_transactions"}
    )

    merged = (
        sales.merge(stores, on="store_id", how="left", copy=False)
        .merge(products, on="product_id", how="left", copy=False)
        .merge(
            transactions[["date", "store_id", "store_transactions"]],
            on=["date", "store_id"],
            how="left",
            copy=False,
        )
        .merge(holiday_daily, on="date", how="left", copy=False)
        .merge(oil[["date", "oil_price"]], on="date", how="left", copy=False)
    )

    merged["on_promotion"] = merged["on_promotion"].fillna(False)
    merged["is_holiday_event"] = merged["is_holiday_event"].fillna(False)
    merged = merged.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return merged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Favorita data via Kaggle and validate required files.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/favorita",
        help="Directory to store Favorita CSV files.",
    )
    parser.add_argument(
        "--competition-slug",
        default=DEFAULT_COMPETITION_SLUG,
        help="Kaggle competition slug.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if required files already exist.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    missing_before = download_favorita_from_kaggle(
        raw_dir=raw_dir,
        competition_slug=args.competition_slug,
        force=args.force,
    )
    if missing_before:
        print(f"Downloaded Favorita files into {raw_dir}.")
    else:
        print(f"Required files already present in {raw_dir}; no download needed.")


if __name__ == "__main__":
    main()
