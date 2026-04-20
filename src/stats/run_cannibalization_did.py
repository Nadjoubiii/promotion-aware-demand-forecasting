"""Run difference-in-differences style cannibalization checks with cluster-matched controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


REQUIRED_COLUMNS = ["date", "store_id", "cluster", "product_class", "units", "on_promotion"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--output-dir", default="reports/stats")
    parser.add_argument("--pre-days", type=int, default=28)
    parser.add_argument("--post-days", type=int, default=28)
    parser.add_argument("--min-promo-days", type=int, default=14)
    parser.add_argument("--max-control-promo-days", type=int, default=3)
    parser.add_argument("--min-stores-per-group", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _bootstrap_ci(treated_changes: np.ndarray, control_changes: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        tb = rng.choice(treated_changes, size=len(treated_changes), replace=True)
        cb = rng.choice(control_changes, size=len(control_changes), replace=True)
        diffs[i] = tb.mean() - cb.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main() -> None:
    args = _build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path, columns=REQUIRED_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["on_promotion"] = df["on_promotion"].astype(bool)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").astype("float32")

    store_class_day = (
        df.groupby(["cluster", "store_id", "date", "product_class"], as_index=False)
        .agg(units=("units", "sum"), promo_active=("on_promotion", "any"))
        .sort_values(["cluster", "store_id", "product_class", "date"])
        .reset_index(drop=True)
    )

    store_day_total = (
        store_class_day.groupby(["cluster", "store_id", "date"], as_index=False)["units"]
        .sum()
        .rename(columns={"units": "store_day_total_units"})
    )
    panel = store_class_day.merge(store_day_total, on=["cluster", "store_id", "date"], how="left")
    panel["other_class_units"] = panel["store_day_total_units"] - panel["units"]

    store_promo_days = (
        panel.groupby(["cluster", "product_class", "store_id"], as_index=False)["promo_active"]
        .sum()
        .rename(columns={"promo_active": "promo_days"})
    )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str | bool]] = []

    for (cluster_value, product_class), group_store_counts in store_promo_days.groupby(["cluster", "product_class"]):
        treated_stores = group_store_counts[group_store_counts["promo_days"] >= args.min_promo_days]["store_id"].tolist()
        control_stores = group_store_counts[group_store_counts["promo_days"] <= args.max_control_promo_days]["store_id"].tolist()
        control_stores = [store_id for store_id in control_stores if store_id not in treated_stores]
        if len(treated_stores) < args.min_stores_per_group or len(control_stores) < args.min_stores_per_group:
            continue

        subgroup = panel[(panel["cluster"] == cluster_value) & (panel["product_class"] == product_class)].copy()
        campaign_dates = subgroup[(subgroup["store_id"].isin(treated_stores)) & (subgroup["promo_active"])]["date"]
        if campaign_dates.empty:
            continue
        campaign_start = campaign_dates.min()
        pre_start = campaign_start - pd.Timedelta(days=args.pre_days)
        post_end = campaign_start + pd.Timedelta(days=args.post_days - 1)

        window = subgroup[(subgroup["date"] >= pre_start) & (subgroup["date"] <= post_end)].copy()
        if window.empty:
            continue

        window["group"] = np.where(window["store_id"].isin(treated_stores), "treated", np.where(window["store_id"].isin(control_stores), "control", "other"))
        window = window[window["group"].isin(["treated", "control"])].copy()
        window["period"] = np.where(window["date"] < campaign_start, "pre", "post")

        store_period = (
            window.groupby(["group", "store_id", "period"], as_index=False)["other_class_units"]
            .mean()
        )
        pivot = store_period.pivot(index=["group", "store_id"], columns="period", values="other_class_units").reset_index()
        pivot = pivot.dropna(subset=["pre", "post"])
        if pivot.empty:
            continue

        pivot["change"] = pivot["post"] - pivot["pre"]
        treated_changes = pivot[pivot["group"] == "treated"]["change"].to_numpy(dtype=float)
        control_changes = pivot[pivot["group"] == "control"]["change"].to_numpy(dtype=float)
        if len(treated_changes) < args.min_stores_per_group or len(control_changes) < args.min_stores_per_group:
            continue

        did_effect = float(np.mean(treated_changes) - np.mean(control_changes))
        control_pre_mean = float(pivot[pivot["group"] == "control"]["pre"].mean())
        did_pct_of_control_pre = float(100 * did_effect / control_pre_mean) if control_pre_mean != 0 else float("nan")
        ci_low, ci_high = _bootstrap_ci(treated_changes, control_changes, args.bootstrap_iterations, rng)
        pvalue = float(ttest_ind(treated_changes, control_changes, equal_var=False, nan_policy="omit").pvalue)

        rows.append(
            {
                "cluster": cluster_value,
                "product_class": product_class,
                "campaign_start": campaign_start.strftime("%Y-%m-%d"),
                "n_treated_stores": int(len(treated_changes)),
                "n_control_stores": int(len(control_changes)),
                "treated_pre_mean_other_class_units": float(pivot[pivot["group"] == "treated"]["pre"].mean()),
                "treated_post_mean_other_class_units": float(pivot[pivot["group"] == "treated"]["post"].mean()),
                "control_pre_mean_other_class_units": control_pre_mean,
                "control_post_mean_other_class_units": float(pivot[pivot["group"] == "control"]["post"].mean()),
                "did_effect_other_class_units": did_effect,
                "did_pct_of_control_pre": did_pct_of_control_pre,
                "did_ci95_low": ci_low,
                "did_ci95_high": ci_high,
                "welch_t_pvalue": pvalue,
                "cannibalization_flag": bool((did_effect < 0) and (ci_high < 0) and (pvalue < 0.05)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        print("No cluster/product_class combinations passed the DiD thresholds.")
        return

    result = result.sort_values("did_effect_other_class_units", ascending=True).reset_index(drop=True)
    out_file = output_dir / "cannibalization_did_by_cluster_product_class.csv"
    result.to_csv(out_file, index=False)

    print("Cannibalization DiD analysis complete.")
    print(f"Saved: {out_file}")
    print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
