"""Train per-cluster SARIMAX models and compare against baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import baseline_predictions, load_cluster_daily_dataset, metrics, save_metrics, save_prediction_frame

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling/sarimax")
    return parser


def fit_predict_one_cluster(train_cluster: pd.DataFrame, valid_cluster: pd.DataFrame) -> pd.Series:
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
    preds = fitted.forecast(steps=len(valid_cluster), exog=valid_cluster[EXOGENOUS_COLS])
    return pd.Series(preds, index=valid_cluster.index)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cluster_daily_dataset(args.data_path, args.cutoff_date)
    pred_parts: list[pd.Series] = []

    for cluster_value, train_cluster in dataset.train_df.groupby(dataset.cfg.segment_col):
        valid_cluster = dataset.valid_df[dataset.valid_df[dataset.cfg.segment_col] == cluster_value].copy()
        if valid_cluster.empty:
            continue
        pred_parts.append(fit_predict_one_cluster(train_cluster, valid_cluster))

    pred_sarimax = pd.concat(pred_parts).sort_index()
    y_valid = dataset.valid_df.loc[pred_sarimax.index, dataset.cfg.target_col]
    valid_eval = dataset.valid_df.loc[pred_sarimax.index]
    pred_baseline = baseline_predictions(valid_eval)

    metrics_df = save_metrics(
        output_dir,
        [
            {"model": "sarimax_cluster_daily", **metrics(y_valid, pred_sarimax)},
            {"model": "seasonal_naive_7d", **metrics(y_valid, pred_baseline)},
        ],
    )
    save_prediction_frame(
        output_dir,
        valid_eval,
        dataset.cfg,
        {"pred_sarimax": pred_sarimax},
    )

    print("SARIMAX training complete.")
    print(f"Saved outputs in: {output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
