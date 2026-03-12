"""Data validation checks for ingestion quality."""

from __future__ import annotations

import pandas as pd


def find_negative_units(df: pd.DataFrame, units_col: str = "units") -> pd.DataFrame:
    """Return rows with invalid negative demand values."""
    if units_col not in df.columns:
        raise ValueError(f"Missing units column: {units_col}")
    return df[df[units_col] < 0]


def find_null_ids(df: pd.DataFrame, id_cols: tuple[str, ...] = ("store_id", "product_id")) -> pd.DataFrame:
    """Return rows where one or more entity IDs are missing."""
    missing_cols = [col for col in id_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing ID columns: {missing_cols}")
    return df[df[list(id_cols)].isna().any(axis=1)]


def find_stockout_candidates(
    df: pd.DataFrame,
    units_col: str = "units",
    promo_col: str = "on_promotion",
) -> pd.DataFrame:
    """Flag rows that may represent stockout days (zero sales with no promotion)."""
    for col in (units_col, promo_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df[(df[units_col] == 0) & (~df[promo_col].fillna(False))]


def find_missing_dates_per_segment(
    df: pd.DataFrame,
    date_col: str = "date",
    segment_cols: tuple[str, ...] = ("store_id", "product_id"),
) -> pd.DataFrame:
    """Return missing daily timestamps between min and max dates per segment."""
    required_cols = [date_col, *segment_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = df[required_cols].dropna().copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])

    rows: list[dict[str, object]] = []
    grouped = work_df.groupby(list(segment_cols), dropna=False)
    for segment_values, group in grouped:
        unique_dates = set(group[date_col].dt.normalize())
        all_dates = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")
        missing_dates = [d for d in all_dates if d not in unique_dates]
        for missing_date in missing_dates:
            row = {segment_cols[idx]: segment_values[idx] for idx in range(len(segment_cols))}
            row[date_col] = missing_date
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[*segment_cols, date_col])

    return pd.DataFrame(rows)
