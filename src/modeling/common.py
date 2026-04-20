"""Shared utilities for cluster-day forecasting benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_cluster_daily_features import FeatureConfig, build_model_frame


@dataclass(frozen=True)
class ModelingDataset:
    cfg: FeatureConfig
    model_df: pd.DataFrame
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    feature_cols: list[str]


def load_cluster_daily_model_frame(data_path: str | Path) -> tuple[FeatureConfig, pd.DataFrame]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    cfg = FeatureConfig(target_col="units", date_col="date", segment_col="cluster")
    required_columns = [
        cfg.date_col,
        cfg.segment_col,
        cfg.target_col,
        cfg.promo_col,
        cfg.holiday_col,
        cfg.oil_col,
        cfg.transactions_col,
    ]
    raw = pd.read_parquet(data_path, columns=required_columns)
    model_df = build_model_frame(raw, cfg=cfg)
    return cfg, model_df


def load_cluster_daily_dataset(data_path: str | Path, cutoff_date: str) -> ModelingDataset:
    cfg, model_df = load_cluster_daily_model_frame(data_path)

    cutoff = pd.Timestamp(cutoff_date)
    train_df = model_df[model_df[cfg.date_col] <= cutoff].copy()
    valid_df = model_df[model_df[cfg.date_col] > cutoff].copy()
    feature_cols = [column for column in model_df.columns if column not in {cfg.target_col, cfg.date_col}]

    return ModelingDataset(
        cfg=cfg,
        model_df=model_df,
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
    )


def safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100)


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape_pct": safe_mape(y_true, y_pred),
    }


def baseline_predictions(valid_df: pd.DataFrame) -> pd.Series:
    return valid_df["lag_7"]


def save_prediction_frame(
    output_dir: str | Path,
    valid_df: pd.DataFrame,
    cfg: FeatureConfig,
    prediction_columns: dict[str, pd.Series],
) -> None:
    output_dir = Path(output_dir)
    pred_frame = valid_df[[cfg.date_col, cfg.segment_col, cfg.target_col, "lag_7"]].copy()
    pred_frame = pred_frame.rename(columns={"lag_7": "pred_seasonal_naive_7d"})

    for column_name, values in prediction_columns.items():
        pred_frame[column_name] = values.values if hasattr(values, "values") else values

    pred_frame.tail(2000).to_csv(output_dir / "validation_predictions_sample.csv", index=False)


def save_metrics(output_dir: str | Path, rows: list[dict[str, float | str]]) -> pd.DataFrame:
    output_dir = Path(output_dir)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False)
    return metrics_df