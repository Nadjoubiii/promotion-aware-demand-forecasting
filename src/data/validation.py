"""Data validation checks for ingestion quality."""

from __future__ import annotations

import pandas as pd


def find_negative_units(df: pd.DataFrame, units_col: str = "units") -> pd.DataFrame:
    """Return rows with invalid negative demand values."""
    if units_col not in df.columns:
        raise ValueError(f"Missing units column: {units_col}")
    return df[df[units_col] < 0]
