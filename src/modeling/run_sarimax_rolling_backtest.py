"""Run rolling-origin backtesting for SARIMAX on cluster-day features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import baseline_predictions, load_cluster_daily_model_frame, metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

EXOGENOUS_COLS = [
    "promo_rate",
    "holiday_rate",
    "oil_price",
    "avg_store_transactions",
    "day_of_week",
    "month",
    "is_month_start",
    "is_month_end",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
        + """
Overfitting detection for SARIMAX is different from tree models:
1. Per-cluster convergence: tracks if model fitting converges (divergence under overfitting).
2. Residual autocorrelation: residuals should be white noise; high ACF suggests underfit or overfitting to trend.
3. Horizon performance drift: if 28-day MAE grows much more than 7-day, it may indicate short-term overfitting.
4. Fold-to-fold stability: consistent metrics across folds suggest generalization; high variance suggests brittle fit.
"""
    )
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 28])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=28)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--output-dir", default="reports/modeling/sarimax_backtest")
    return parser


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


def _fit_predict_one_cluster(
    train_cluster: pd.DataFrame, eval_cluster: pd.DataFrame
) -> tuple[pd.Series, dict[str, float | int]]:
    try:
        model = SARIMAX(
            endog=train_cluster["units"],
            exog=train_cluster[EXOGENOUS_COLS],
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

        preds = fitted.forecast(steps=len(eval_cluster), exog=eval_cluster[EXOGENOUS_COLS])
        residuals = fitted.resid

        convergence_info = {
            "converged": int(fitted.mle_retvals is not None and fitted.mle_retvals.get("converged", 0)),
            "residuals_mean_abs": float(abs(residuals).mean()),
            "residuals_std": float(residuals.std()),
        }
        return pd.Series(preds, index=eval_cluster.index), convergence_info
    except Exception:
        return pd.Series(index=eval_cluster.index), {"converged": 0, "residuals_mean_abs": float("nan"), "residuals_std": float("nan")}


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

    fold_rows: list[dict[str, float | int | str]] = []
    diagnostics_rows: list[dict[str, float | int | str]] = []

    for fold_id, train_end_date, eval_end_date in boundaries:
        train_df = model_df[model_df[cfg.date_col] <= train_end_date].copy()
        eval_df = model_df[
            (model_df[cfg.date_col] > train_end_date) & (model_df[cfg.date_col] <= eval_end_date)
        ].copy()

        pred_parts: list[pd.Series] = []
        convergence_logs: list[dict[str, float | int]] = []

        for cluster_value, train_cluster in train_df.groupby(cfg.segment_col):
            eval_cluster = eval_df[eval_df[cfg.segment_col] == cluster_value].copy()
            if eval_cluster.empty:
                continue

            pred_cluster, conv_info = _fit_predict_one_cluster(train_cluster, eval_cluster)
            if not pred_cluster.empty:
                pred_parts.append(pred_cluster)
                convergence_logs.append(
                    {
                        "fold": fold_id,
                        "cluster": cluster_value,
                        "converged": conv_info["converged"],
                        "residuals_mean_abs": conv_info["residuals_mean_abs"],
                        "residuals_std": conv_info["residuals_std"],
                    }
                )

        pred_all = pd.concat(pred_parts).sort_index()
        y_eval = eval_df.loc[pred_all.index, cfg.target_col]
        eval_subset = eval_df.loc[pred_all.index]
        pred_baseline = baseline_predictions(eval_subset)

        eval_metrics_max_h = metrics(y_eval, pred_all)
        baseline_metrics_max_h = metrics(y_eval, pred_baseline)
        convergence_rate = (
            sum(log["converged"] for log in convergence_logs) / len(convergence_logs)
            if convergence_logs
            else 0.0
        )

        diagnostics_rows.append(
            {
                "fold": fold_id,
                "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                "eval_end_date": eval_end_date.strftime("%Y-%m-%d"),
                "convergence_rate": convergence_rate,
                "sarimax_mae_max_horizon": eval_metrics_max_h["mae"],
                "baseline_mae_max_horizon": baseline_metrics_max_h["mae"],
                "mae_improvement_vs_baseline": baseline_metrics_max_h["mae"] - eval_metrics_max_h["mae"],
            }
        )

        for horizon in horizons:
            horizon_end_date = train_end_date + pd.Timedelta(days=horizon)
            eval_h = eval_subset[eval_subset[cfg.date_col] <= horizon_end_date].copy()
            pred_h = pred_all.loc[eval_h.index]
            baseline_h = baseline_predictions(eval_h)
            y_h = eval_h[cfg.target_col]

            fold_rows.append(
                {
                    "fold": fold_id,
                    "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                    "horizon_days": horizon,
                    "model": "sarimax_cluster_daily",
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
    if {"sarimax_cluster_daily", "seasonal_naive_7d"}.issubset(pivot.columns):
        improvement = (
            (pivot["seasonal_naive_7d"] - pivot["sarimax_cluster_daily"]) / pivot["seasonal_naive_7d"] * 100
        )
        improvement_df = improvement.rename("sarimax_mae_improvement_vs_naive_pct").reset_index()
        improvement_df.to_csv(output_dir / "sarimax_vs_naive_improvement.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(output_dir / "convergence_and_performance_diagnostics.csv", index=False)

    print("Rolling backtest complete.")
    print(f"Saved outputs in: {output_dir}")
    print(summary.to_string(index=False))
    print("\n=== SARIMAX Overfitting Diagnostics ===")
    if diagnostics_df.shape[0] > 0:
        print("Convergence rate by fold:", diagnostics_df["convergence_rate"].mean())
        print("MAE improvement vs baseline:", diagnostics_df["mae_improvement_vs_baseline"].mean())


if __name__ == "__main__":
    main()
