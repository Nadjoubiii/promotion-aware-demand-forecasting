"""Run baseline forecasting benchmarks on segment-level daily demand."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="data/processed/training_table.parquet",
        help="Path to the processed training table parquet.",
    )
    parser.add_argument(
        "--segment-col",
        default="cluster",
        help="Segment column for grouped baselines (e.g., cluster, store_id).",
    )
    parser.add_argument(
        "--target-col",
        default="units",
        help="Target column to forecast.",
    )
    parser.add_argument(
        "--date-col",
        default="date",
        help="Date column name.",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2017-06-30",
        help="Validation starts after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/baseline",
        help="Directory to save benchmark artifacts.",
    )
    return parser


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100)


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape_pct": _safe_mape(y_true, y_pred),
    }


def _build_segment_daily(
    df: pd.DataFrame,
    date_col: str,
    segment_col: str,
    target_col: str,
) -> pd.DataFrame:
    out = (
        df[[date_col, segment_col, target_col]]
        .dropna(subset=[date_col, segment_col, target_col])
        .groupby([segment_col, date_col], as_index=False)[target_col]
        .sum()
        .sort_values([segment_col, date_col])
        .reset_index(drop=True)
    )
    return out


def _add_baselines(
    segment_daily: pd.DataFrame,
    segment_col: str,
    target_col: str,
) -> pd.DataFrame:
    out = segment_daily.copy()

    grouped = out.groupby(segment_col)[target_col]
    out["pred_seasonal_naive_7d"] = grouped.shift(7)
    out["pred_moving_avg_7d"] = grouped.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

    return out


def _evaluate(
    scored_df: pd.DataFrame,
    segment_col: str,
    target_col: str,
    pred_col: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    work = scored_df[[segment_col, target_col, pred_col]].dropna(subset=[pred_col])

    overall = _compute_metrics(work[target_col], work[pred_col])

    rows: list[dict[str, float | int | str]] = []
    for seg_value, g in work.groupby(segment_col, sort=False):
        rows.append({segment_col: seg_value, **_compute_metrics(g[target_col], g[pred_col])})
    per_segment = pd.DataFrame(rows).sort_values("mae")
    return overall, per_segment


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_parquet(data_path, columns=[args.date_col, args.segment_col, args.target_col])
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    segment_daily = _build_segment_daily(
        df=df,
        date_col=args.date_col,
        segment_col=args.segment_col,
        target_col=args.target_col,
    )

    scored = _add_baselines(
        segment_daily=segment_daily,
        segment_col=args.segment_col,
        target_col=args.target_col,
    )

    cutoff = pd.Timestamp(args.cutoff_date)
    valid = scored[scored[args.date_col] > cutoff].copy()

    metrics_rows: list[dict[str, float | str]] = []
    for pred_col, model_name in [
        ("pred_seasonal_naive_7d", "seasonal_naive_7d"),
        ("pred_moving_avg_7d", "moving_avg_7d"),
    ]:
        overall, per_segment = _evaluate(
            scored_df=valid,
            segment_col=args.segment_col,
            target_col=args.target_col,
            pred_col=pred_col,
        )

        metrics_rows.append({"model": model_name, **overall})
        per_segment.insert(1, "model", model_name)
        per_segment.to_csv(output_dir / f"metrics_by_{args.segment_col}_{model_name}.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics_overall.csv", index=False)

    sample_cols = [args.date_col, args.segment_col, args.target_col, "pred_seasonal_naive_7d", "pred_moving_avg_7d"]
    valid[sample_cols].tail(500).to_csv(output_dir / "validation_predictions_sample.csv", index=False)

    print("Baseline benchmark complete.")
    print(f"Output directory: {output_dir}")
    print("Saved files:")
    print("- metrics_overall.csv")
    print(f"- metrics_by_{args.segment_col}_seasonal_naive_7d.csv")
    print(f"- metrics_by_{args.segment_col}_moving_avg_7d.csv")
    print("- validation_predictions_sample.csv")


if __name__ == "__main__":
    main()
