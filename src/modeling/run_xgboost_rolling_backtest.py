"""Run rolling-origin backtesting for XGBoost on cluster-day features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import baseline_predictions, load_cluster_daily_model_frame, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 28])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=28)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--output-dir", default="reports/modeling/xgboost_backtest")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _build_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )


def _compute_fold_boundaries(
    unique_dates: list[pd.Timestamp],
    n_folds: int,
    step_days: int,
    max_horizon: int,
    min_train_days: int,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    boundaries: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    n_dates = len(unique_dates)

    for fold_idx in range(n_folds):
        eval_end_idx = n_dates - 1 - (n_folds - 1 - fold_idx) * step_days
        train_end_idx = eval_end_idx - max_horizon
        if train_end_idx < min_train_days:
            continue
        if eval_end_idx >= n_dates or train_end_idx < 0:
            continue

        train_end_date = unique_dates[train_end_idx]
        eval_end_date = unique_dates[eval_end_idx]
        boundaries.append((fold_idx + 1, train_end_date, eval_end_date))

    if not boundaries:
        raise ValueError("Not enough history for requested folds/horizons/min-train-days configuration.")

    return boundaries


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = sorted(set(args.horizons))
    max_horizon = max(horizons)

    cfg, model_df = load_cluster_daily_model_frame(args.data_path)
    model_df = model_df.sort_values([cfg.segment_col, cfg.date_col]).reset_index(drop=True)

    unique_dates = sorted(model_df[cfg.date_col].drop_duplicates())
    boundaries = _compute_fold_boundaries(
        unique_dates=unique_dates,
        n_folds=args.n_folds,
        step_days=args.step_days,
        max_horizon=max_horizon,
        min_train_days=args.min_train_days,
    )

    feature_cols = [column for column in model_df.columns if column not in {cfg.target_col, cfg.date_col}]

    fold_rows: list[dict[str, float | int | str]] = []
    diagnostics_rows: list[dict[str, float | int | str]] = []

    for fold_id, train_end_date, eval_end_date in boundaries:
        train_df = model_df[model_df[cfg.date_col] <= train_end_date].copy()
        eval_df = model_df[
            (model_df[cfg.date_col] > train_end_date) & (model_df[cfg.date_col] <= eval_end_date)
        ].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[cfg.target_col]
        X_eval = eval_df[feature_cols]
        y_eval = eval_df[cfg.target_col]

        model = _build_model(args.seed)
        model.fit(X_train, y_train)

        pred_train = pd.Series(model.predict(X_train), index=train_df.index)
        pred_eval = pd.Series(model.predict(X_eval), index=eval_df.index)

        train_metrics = metrics(y_train, pred_train)
        eval_metrics_max_h = metrics(y_eval, pred_eval)
        diagnostics_rows.append(
            {
                "fold": fold_id,
                "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                "eval_end_date": eval_end_date.strftime("%Y-%m-%d"),
                "train_mae": train_metrics["mae"],
                "eval_mae_max_horizon": eval_metrics_max_h["mae"],
                "mae_gap": eval_metrics_max_h["mae"] - train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "eval_rmse_max_horizon": eval_metrics_max_h["rmse"],
                "rmse_gap": eval_metrics_max_h["rmse"] - train_metrics["rmse"],
            }
        )

        for horizon in horizons:
            horizon_end_date = train_end_date + pd.Timedelta(days=horizon)
            eval_h = eval_df[eval_df[cfg.date_col] <= horizon_end_date].copy()
            pred_h = pred_eval.loc[eval_h.index]
            baseline_h = baseline_predictions(eval_h)
            y_h = eval_h[cfg.target_col]

            fold_rows.append(
                {
                    "fold": fold_id,
                    "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                    "horizon_days": horizon,
                    "model": "xgboost_cluster_daily",
                    **metrics(y_h, pred_h),
                }
            )
            fold_rows.append(
                {
                    "fold": fold_id,
                    "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                    "horizon_days": horizon,
                    "model": "seasonal_naive_7d",
                    **metrics(y_h, baseline_h),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    summary = (
        fold_metrics.groupby(["model", "horizon_days"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape_pct", "mean"),
            mape_std=("mape_pct", "std"),
        )
        .sort_values(["horizon_days", "mae_mean"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary_by_horizon.csv", index=False)

    pivot = summary.pivot(index="horizon_days", columns="model", values="mae_mean")
    if {"xgboost_cluster_daily", "seasonal_naive_7d"}.issubset(pivot.columns):
        improvement = (
            (pivot["seasonal_naive_7d"] - pivot["xgboost_cluster_daily"]) / pivot["seasonal_naive_7d"] * 100
        )
        improvement_df = improvement.rename("xgboost_mae_improvement_vs_naive_pct").reset_index()
        improvement_df.to_csv(output_dir / "xgboost_vs_naive_improvement.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(output_dir / "overfitting_diagnostics.csv", index=False)

    print("Rolling backtest complete.")
    print(f"Saved outputs in: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
