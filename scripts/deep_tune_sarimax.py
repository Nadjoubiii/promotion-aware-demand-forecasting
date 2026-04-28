"""Per-cluster SARIMAX order search — uses IDENTICAL data pipeline as train_sarimax.py.

Uses load_cluster_daily_dataset from src.modeling.common so train/valid splits,
dropna filtering, and exog feature values are byte-for-byte identical to the
baseline evaluation. This ensures MAE comparisons are apples-to-apples.

Usage:
  python scripts/deep_tune_sarimax.py --data-path data/processed/training_table.parquet \\
      --cutoff-date 2017-06-30 --max-clusters 0

Outputs (default folder `reports/modeling/sarimax_deep`):
- `per_cluster_sarimax_best.csv` — best order per cluster
- `sarimax_all_trials.csv`       — all tried (cluster, order) combos
- `sarimax_deep_overall_metrics.csv` — overall MAE/RMSE/MAPE vs baseline
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import load_cluster_daily_dataset

# Exactly matches train_sarimax.py
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

# Grid: (order, seasonal_order) pairs to try per cluster.
# Kept small on purpose — SARIMAX fitting on daily data with 8 exog vars
# is expensive (~60-120s per fit). The original (1,0,1)+(1,0,1,7) is first.
ORDER_GRID = [
    ((1, 0, 1), (1, 0, 1, 7)),  # original baseline — must always be in grid
    ((1, 1, 1), (1, 0, 1, 7)),
    ((0, 1, 1), (0, 1, 1, 7)),
    ((1, 0, 1), (0, 1, 1, 7)),
    ((2, 0, 1), (1, 0, 1, 7)),
    ((1, 0, 2), (1, 0, 1, 7)),
]


def load_sarimax_data(data_path: str, cutoff_date: str):
    """Load data using the IDENTICAL pipeline as train_sarimax.py (via load_cluster_daily_dataset).
    This ensures the train/valid split and dropna filtering are exactly the same,
    so our MAE is comparable to the baseline 2311.41."""
    print("Loading data ...", flush=True)
    dataset = load_cluster_daily_dataset(data_path, cutoff_date)
    train_df = dataset.train_df
    valid_df = dataset.valid_df
    print(f"Loaded. train={train_df.shape}, valid={valid_df.shape}, "
          f"clusters={train_df[dataset.cfg.segment_col].nunique()}", flush=True)
    return train_df, valid_df, dataset.cfg.segment_col


def safe_mape(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100)


def compute_metrics(y_true, y_pred):
    err = y_true - y_pred
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape_pct": safe_mape(y_true, y_pred),
    }


def fit_one(train_cluster, valid_cluster, order, seasonal_order):
    """Fit SARIMAX with exog and forecast. Returns pd.Series or (None, error_str).
    Data is pre-cleaned by load_cluster_daily_dataset (same as train_sarimax.py),
    so no additional NaN filling is needed."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            train_exog = train_cluster[EXOGENOUS_COLS].astype(float)
            valid_exog = valid_cluster[EXOGENOUS_COLS].astype(float)

            model = SARIMAX(
                endog=train_cluster["units"],
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            preds = fitted.forecast(
                steps=len(valid_cluster),
                exog=valid_exog,
            )
            return pd.Series(preds.values, index=valid_cluster.index)
        except Exception as e:
            return None, str(e)


def tune_cluster(cluster_id, train_cluster, valid_cluster, trial_rows):
    """Grid search over ORDER_GRID for one cluster. Returns best preds Series."""
    y_valid = valid_cluster["units"]
    best_mae = float("inf")
    best_preds = None
    best_order = None
    best_seas = None

    for order, seasonal_order in ORDER_GRID:
        result = fit_one(train_cluster, valid_cluster, order, seasonal_order)
        if result is None or isinstance(result, tuple):
            err_msg = result[1] if isinstance(result, tuple) else "unknown"
            trial_rows.append({
                "cluster": cluster_id, "order": str(order),
                "seasonal_order": str(seasonal_order), "mae": None, "error": err_msg,
            })
            continue
        preds = result.clip(lower=0)
        mae = float(np.abs(y_valid - preds).mean())
        trial_rows.append({
            "cluster": cluster_id, "order": str(order),
            "seasonal_order": str(seasonal_order), "mae": mae, "error": None,
        })
        print(f"    order={order} seasonal={seasonal_order} MAE={mae:.1f}", flush=True)
        if mae < best_mae:
            best_mae = mae
            best_preds = preds
            best_order = order
            best_seas = seasonal_order

    print(f"  [cluster={cluster_id}] best order={best_order} seasonal={best_seas} MAE={best_mae:.1f}", flush=True)
    return best_preds, best_order, best_seas, best_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling/sarimax_deep")
    parser.add_argument("--max-clusters", type=int, default=0,
                        help="Limit number of clusters to tune (0 = all). Use for quick tests.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, segment_col = load_sarimax_data(args.data_path, args.cutoff_date)

    clusters = sorted(train_df[segment_col].unique())
    if args.max_clusters and args.max_clusters > 0:
        clusters = clusters[:args.max_clusters]
    print(f"Tuning SARIMAX for {len(clusters)} clusters, {len(ORDER_GRID)} order combos each.", flush=True)

    pred_parts = []
    cluster_summary = []
    trial_rows = []

    for cluster_id in clusters:
        print(f"\nCluster {cluster_id} ...", flush=True)
        train_cluster = train_df[train_df[segment_col] == cluster_id].copy()
        valid_cluster = valid_df[valid_df[segment_col] == cluster_id].copy()
        if valid_cluster.empty:
            print(f"  [cluster={cluster_id}] no validation data, skipping")
            continue

        best_preds, best_order, best_seas, best_mae = tune_cluster(
            cluster_id, train_cluster, valid_cluster, trial_rows
        )
        if best_preds is not None:
            pred_parts.append(best_preds)
            cluster_summary.append({
                "cluster": cluster_id,
                "best_order": str(best_order),
                "best_seasonal_order": str(best_seas),
                "mae": best_mae,
            })

    # Evaluate overall — same method as train_sarimax.py
    if pred_parts:
        pred_sarimax = pd.concat(pred_parts).sort_index()
        y_valid = valid_df.loc[pred_sarimax.index, "units"]
        m = compute_metrics(y_valid, pred_sarimax)
        print(f"\n=== OVERALL TUNED SARIMAX ===", flush=True)
        print(f"MAE  = {m['mae']:.4f}", flush=True)
        print(f"RMSE = {m['rmse']:.4f}", flush=True)
        print(f"MAPE = {m['mape_pct']:.4f}%", flush=True)
        print(f"\nBaseline (corrected 17-cluster train_sarimax.py): MAE=4305.61, RMSE=6686.01, MAPE=9.07%", flush=True)
        delta = m['mae'] - 4305.610265062486
        print(f"MAE delta vs baseline: {delta:+.2f}  ({'better' if delta < 0 else 'worse'})", flush=True)

        pd.DataFrame([{"model": "sarimax_deep_tuned", **m}]).to_csv(
            out / "sarimax_deep_overall_metrics.csv", index=False
        )
    else:
        print("No predictions produced.", flush=True)

    pd.DataFrame(cluster_summary).to_csv(out / "per_cluster_sarimax_best.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(out / "sarimax_all_trials.csv", index=False)
    print(f"\nOutputs saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
