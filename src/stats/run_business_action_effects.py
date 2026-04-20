"""Estimate effect sizes and confidence intervals for business actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


EVENTS = {
    "promotion": "on_promotion",
    "holiday": "is_holiday_event",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--output-dir", default="reports/stats")
    parser.add_argument("--segment-col", default="cluster")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--min-days-per-group", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    pooled_den = (len(x) - 1) + (len(y) - 1)
    if pooled_den <= 0:
        return float("nan")
    pooled_std = np.sqrt(((len(x) - 1) * x_var + (len(y) - 1) * y_var) / pooled_den)
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xb.mean() - yb.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p.values)
    ranked = p.values[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return pd.Series(out, index=p.index)


def _build_segment_day(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    work = df[["date", segment_col, "units", "on_promotion", "is_holiday_event"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work["on_promotion"] = work["on_promotion"].astype(bool)
    work["is_holiday_event"] = work["is_holiday_event"].astype(bool)
    out = (
        work.groupby([segment_col, "date"], as_index=False)
        .agg(
            units=("units", "sum"),
            on_promotion=("on_promotion", "any"),
            is_holiday_event=("is_holiday_event", "any"),
        )
        .sort_values([segment_col, "date"])
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    args = _build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        data_path,
        columns=["date", args.segment_col, "units", "on_promotion", "is_holiday_event"],
    )
    segment_day = _build_segment_day(df, args.segment_col)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, float | int | str | bool]] = []

    for event_name, event_col in EVENTS.items():
        for segment_value, g in segment_day.groupby(args.segment_col):
            event_vals = g[g[event_col]]["units"].to_numpy(dtype=float)
            base_vals = g[~g[event_col]]["units"].to_numpy(dtype=float)

            if len(event_vals) < args.min_days_per_group or len(base_vals) < args.min_days_per_group:
                continue

            mean_event = float(np.mean(event_vals))
            mean_base = float(np.mean(base_vals))
            uplift_abs = mean_event - mean_base
            uplift_pct = float(100 * uplift_abs / mean_base) if mean_base != 0 else float("nan")
            pvalue = float(ttest_ind(event_vals, base_vals, equal_var=False, nan_policy="omit").pvalue)
            ci_low, ci_high = _bootstrap_ci(event_vals, base_vals, args.bootstrap_iterations, rng)

            rows.append(
                {
                    "event_type": event_name,
                    args.segment_col: segment_value,
                    "n_event_days": int(len(event_vals)),
                    "n_non_event_days": int(len(base_vals)),
                    "mean_event": mean_event,
                    "mean_non_event": mean_base,
                    "uplift_abs": uplift_abs,
                    "uplift_pct": uplift_pct,
                    "cohens_d": _cohens_d(event_vals, base_vals),
                    "welch_t_pvalue": pvalue,
                    "mean_diff_ci95_low": ci_low,
                    "mean_diff_ci95_high": ci_high,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        print("No segments passed minimum group-size threshold.")
        return

    result["fdr_qvalue"] = result.groupby("event_type")["welch_t_pvalue"].transform(_benjamini_hochberg)
    result["significant_fdr_0_05"] = result["fdr_qvalue"] < 0.05
    result["actionable_positive_flag"] = (
        result["significant_fdr_0_05"]
        & (result["mean_diff_ci95_low"] > 0)
        & (result["uplift_pct"] >= 5)
    )
    result = result.sort_values(["event_type", "uplift_pct"], ascending=[True, False]).reset_index(drop=True)

    out_file = output_dir / "business_action_effects.csv"
    result.to_csv(out_file, index=False)

    print("Business action effect analysis complete.")
    print(f"Saved: {out_file}")
    print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
