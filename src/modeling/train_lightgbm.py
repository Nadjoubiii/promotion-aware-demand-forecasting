"""Train a LightGBM model on cluster-day features and compare against baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_cluster_daily_features import FeatureConfig, build_model_frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100)


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape_pct": _safe_mape(y_true, y_pred),
    }


def main() -> None:
    args = _build_parser().parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(data_path)
    cfg = FeatureConfig(target_col="units", date_col="date", segment_col="cluster")
    model_df = build_model_frame(raw, cfg=cfg)

    cutoff = pd.Timestamp(args.cutoff_date)
    train_df = model_df[model_df[cfg.date_col] <= cutoff].copy()
    valid_df = model_df[model_df[cfg.date_col] > cutoff].copy()

    feature_cols = [
        c
        for c in model_df.columns
        if c not in {cfg.target_col, cfg.date_col}
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[cfg.target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[cfg.target_col]

    # A compact tabular model to establish a first strong non-baseline benchmark.
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        objective="regression",
    )
    model.fit(X_train, y_train)

    pred_lgbm = pd.Series(model.predict(X_valid), index=valid_df.index)
    pred_baseline = valid_df["lag_7"]

    metrics_lgbm = _metrics(y_valid, pred_lgbm)
    metrics_baseline = _metrics(y_valid, pred_baseline)

    metrics_df = pd.DataFrame(
        [
            {"model": "lightgbm_cluster_daily", **metrics_lgbm},
            {"model": "seasonal_naive_7d", **metrics_baseline},
        ]
    )
    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False)

    pred_frame = valid_df[[cfg.date_col, cfg.segment_col, cfg.target_col, "lag_7"]].copy()
    pred_frame = pred_frame.rename(columns={"lag_7": "pred_seasonal_naive_7d"})
    pred_frame["pred_lightgbm"] = pred_lgbm.values
    pred_frame.tail(2000).to_csv(output_dir / "validation_predictions_sample.csv", index=False)

    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    fi.to_csv(output_dir / "feature_importance.csv", index=False)

    model.booster_.save_model(str(output_dir / "lightgbm_model.txt"))

    print("LightGBM training complete.")
    print(f"Saved outputs in: {output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
