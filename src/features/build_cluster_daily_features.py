"""Feature builder for cluster-day demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    target_col: str = "units"
    date_col: str = "date"
    segment_col: str = "cluster"
    promo_col: str = "on_promotion"
    holiday_col: str = "is_holiday_event"
    oil_col: str = "oil_price"
    transactions_col: str = "store_transactions"


LAG_COLS = [7, 14, 28]


def build_cluster_daily_base(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Aggregate row-level data to cluster-day level with exogenous summaries."""
    required = [
        cfg.date_col,
        cfg.segment_col,
        cfg.target_col,
        cfg.promo_col,
        cfg.holiday_col,
        cfg.oil_col,
        cfg.transactions_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Avoid an additional full-frame copy to keep memory usage manageable.
    work = df.dropna(subset=[cfg.date_col, cfg.segment_col, cfg.target_col]).copy()
    work[cfg.date_col] = pd.to_datetime(work[cfg.date_col])
    work[cfg.promo_col] = work[cfg.promo_col].astype("int8")
    work[cfg.holiday_col] = work[cfg.holiday_col].astype("int8")
    work[cfg.target_col] = pd.to_numeric(work[cfg.target_col], errors="coerce").astype("float32")
    work[cfg.oil_col] = pd.to_numeric(work[cfg.oil_col], errors="coerce").astype("float32")
    work[cfg.transactions_col] = pd.to_numeric(work[cfg.transactions_col], errors="coerce").astype("float32")

    agg = (
        work.groupby([cfg.segment_col, cfg.date_col], as_index=False)
        .agg(
            units=(cfg.target_col, "sum"),
            promo_rate=(cfg.promo_col, "mean"),
            holiday_rate=(cfg.holiday_col, "mean"),
            oil_price=(cfg.oil_col, "mean"),
            avg_store_transactions=(cfg.transactions_col, "mean"),
        )
        .sort_values([cfg.segment_col, cfg.date_col])
        .reset_index(drop=True)
    )
    return agg


def add_time_features(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Add deterministic calendar features."""
    out = df.copy()
    date = out[cfg.date_col]
    out["day_of_week"] = date.dt.dayofweek
    out["day_of_month"] = date.dt.day
    out["month"] = date.dt.month
    out["week_of_year"] = date.dt.isocalendar().week.astype("int16")
    out["is_month_start"] = date.dt.is_month_start.astype("int8")
    out["is_month_end"] = date.dt.is_month_end.astype("int8")
    return out


def add_lag_rolling_features(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Add autoregressive lag and rolling features per segment."""
    out = df.sort_values([cfg.segment_col, cfg.date_col]).copy()
    g = out.groupby(cfg.segment_col)[cfg.target_col]

    for lag in LAG_COLS:
        out[f"lag_{lag}"] = g.shift(lag)

    out["rolling_mean_7"] = g.shift(1).rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
    out["rolling_mean_28"] = g.shift(1).rolling(28, min_periods=7).mean().reset_index(level=0, drop=True)
    out["rolling_std_28"] = g.shift(1).rolling(28, min_periods=7).std().reset_index(level=0, drop=True)
    return out


def build_model_frame(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Build final supervised frame for modeling."""
    out = build_cluster_daily_base(df=df, cfg=cfg)
    out = add_time_features(out, cfg=cfg)
    out = add_lag_rolling_features(out, cfg=cfg)
    out = out.dropna().reset_index(drop=True)
    return out
