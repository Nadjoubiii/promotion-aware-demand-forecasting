"""Run promotion significance analysis by segment.

This script compares promo vs non-promo demand distributions at a segment-day level and
reports effect size, confidence intervals, and statistical significance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--output-dir", default="reports/stats")
    parser.add_argument("--segment-col", default="cluster")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-col", default="units")
    parser.add_argument("--promo-col", default="on_promotion")
    parser.add_argument(
        "--min-days-per-group",
        type=int,
        default=30,
        help="Minimum number of segment-days in both promo and non-promo groups.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap draws for mean-difference confidence interval.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d using pooled standard deviation."""
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


def _bootstrap_mean_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for mean(x) - mean(y)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")

    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xb.mean() - yb.mean()

    low = float(np.quantile(diffs, alpha / 2))
    high = float(np.quantile(diffs, 1 - alpha / 2))
    return low, high


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
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


def _build_segment_day_table(
    df: pd.DataFrame,
    segment_col: str,
    date_col: str,
    target_col: str,
    promo_col: str,
) -> pd.DataFrame:
    """Aggregate to segment-day-promo level to reduce row-level dependence.

    Each row in the output represents one segment-day and promo status, using mean units.
    """
    out = (
        df[[segment_col, date_col, target_col, promo_col]]
        .dropna(subset=[segment_col, date_col, target_col, promo_col])
        .assign(**{promo_col: lambda x: x[promo_col].astype(bool), date_col: lambda x: pd.to_datetime(x[date_col])})
        .groupby([segment_col, date_col, promo_col], as_index=False)[target_col]
        .mean()
    )
    return out


def main() -> None:
    args = _build_parser().parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path, columns=[args.segment_col, args.date_col, args.target_col, args.promo_col])
    segment_day = _build_segment_day_table(
        df=df,
        segment_col=args.segment_col,
        date_col=args.date_col,
        target_col=args.target_col,
        promo_col=args.promo_col,
    )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str | bool]] = []

    for seg_value, g in segment_day.groupby(args.segment_col, sort=False):
        promo_vals = g[g[args.promo_col]][args.target_col].to_numpy(dtype=float)
        nonpromo_vals = g[~g[args.promo_col]][args.target_col].to_numpy(dtype=float)

        if len(promo_vals) < args.min_days_per_group or len(nonpromo_vals) < args.min_days_per_group:
            continue

        mean_promo = float(np.mean(promo_vals))
        mean_nonpromo = float(np.mean(nonpromo_vals))
        uplift_abs = mean_promo - mean_nonpromo
        uplift_pct = float(100 * uplift_abs / mean_nonpromo) if mean_nonpromo != 0 else float("nan")

        # Welch's t-test is robust to unequal variances between groups.
        welch = ttest_ind(promo_vals, nonpromo_vals, equal_var=False, nan_policy="omit")

        # Mann-Whitney is a non-parametric check against distributional assumptions.
        mwu = mannwhitneyu(promo_vals, nonpromo_vals, alternative="two-sided")

        ci_low, ci_high = _bootstrap_mean_diff_ci(
            x=promo_vals,
            y=nonpromo_vals,
            n_boot=args.bootstrap_iterations,
            rng=rng,
            alpha=0.05,
        )

        rows.append(
            {
                args.segment_col: seg_value,
                "n_days_promo": int(len(promo_vals)),
                "n_days_nonpromo": int(len(nonpromo_vals)),
                "mean_promo": mean_promo,
                "mean_nonpromo": mean_nonpromo,
                "uplift_abs": uplift_abs,
                "uplift_pct": uplift_pct,
                "cohens_d": _cohens_d(promo_vals, nonpromo_vals),
                "welch_t_pvalue": float(welch.pvalue),
                "mannwhitney_pvalue": float(mwu.pvalue),
                "mean_diff_ci95_low": ci_low,
                "mean_diff_ci95_high": ci_high,
                "significant_0_05": bool(float(welch.pvalue) < 0.05),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        print("No segments passed minimum group-size threshold.")
        return

    result["welch_fdr_qvalue"] = _benjamini_hochberg(result["welch_t_pvalue"])
    result["significant_fdr_0_05"] = result["welch_fdr_qvalue"] < 0.05

    result = result.sort_values("uplift_pct", ascending=False).reset_index(drop=True)

    out_file = output_dir / f"promo_significance_by_{args.segment_col}.csv"
    result.to_csv(out_file, index=False)

    print("Promotion significance analysis complete.")
    print(f"Saved: {out_file}")
    print("Top rows:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
