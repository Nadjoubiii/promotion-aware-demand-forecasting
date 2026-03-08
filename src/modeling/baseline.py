"""Baseline models for quick benchmarking."""

from __future__ import annotations

import pandas as pd


def seasonal_naive_last_week(df: pd.DataFrame, target_col: str = "units", season_lag: int = 7) -> pd.Series:
    """Return seasonal naive predictions using a fixed lag.

    The input frame is expected to be time-ordered per segment.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    return df[target_col].shift(season_lag)
