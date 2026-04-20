"""Run count-distribution and residual diagnostics for retail demand."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan


REQUIRED_COLUMNS = [
    "date",
    "cluster",
    "units",
    "on_promotion",
    "is_holiday_event",
    "store_transactions",
    "oil_price",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--output-dir", default="reports/stats")
    parser.add_argument("--min-cluster-days", type=int, default=60)
    return parser


def _count_stats(values: pd.Series) -> dict[str, float]:
    arr = values.to_numpy(dtype=float)
    mean_units = float(np.mean(arr))
    variance_units = float(np.var(arr, ddof=1)) if len(arr) > 1 else float("nan")
    observed_zero_rate = float(np.mean(arr == 0))
    expected_zero_poisson = float(np.exp(-mean_units)) if mean_units >= 0 else float("nan")
    dispersion_ratio = float(variance_units / mean_units) if mean_units > 0 else float("nan")
    nb_alpha = float(max((variance_units - mean_units) / (mean_units**2), 0.0)) if mean_units > 0 else float("nan")
    return {
        "n_obs": int(len(arr)),
        "mean_units": mean_units,
        "variance_units": variance_units,
        "dispersion_ratio": dispersion_ratio,
        "observed_zero_rate": observed_zero_rate,
        "poisson_expected_zero_rate": expected_zero_poisson,
        "zero_inflation_gap": observed_zero_rate - expected_zero_poisson,
        "nb_alpha_estimate": nb_alpha,
        "poisson_suitable_flag": bool((dispersion_ratio <= 1.2) and (abs(observed_zero_rate - expected_zero_poisson) <= 0.02)) if np.isfinite(dispersion_ratio) else False,
        "negbin_preferred_flag": bool((dispersion_ratio > 1.2) or (observed_zero_rate - expected_zero_poisson > 0.02)) if np.isfinite(dispersion_ratio) else False,
    }


def _build_cluster_day(df: pd.DataFrame) -> pd.DataFrame:
    work = df[REQUIRED_COLUMNS].dropna(subset=["date", "cluster", "units"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    work["units"] = pd.to_numeric(work["units"], errors="coerce").astype("float32")
    work["on_promotion"] = work["on_promotion"].astype(bool)
    work["is_holiday_event"] = work["is_holiday_event"].astype(bool)
    work["store_transactions"] = pd.to_numeric(work["store_transactions"], errors="coerce").astype("float32")
    work["oil_price"] = pd.to_numeric(work["oil_price"], errors="coerce").astype("float32")

    out = (
        work.groupby(["cluster", "date"], as_index=False)
        .agg(
            units=("units", "sum"),
            promo_rate=("on_promotion", "mean"),
            holiday_rate=("is_holiday_event", "mean"),
            avg_store_transactions=("store_transactions", "mean"),
            oil_price=("oil_price", "mean"),
        )
        .sort_values(["cluster", "date"])
        .reset_index(drop=True)
    )
    out["avg_store_transactions"] = out["avg_store_transactions"].ffill().bfill()
    out["oil_price"] = out["oil_price"].ffill().bfill()
    out["day_of_week"] = out["date"].dt.dayofweek.astype("int8")
    out["month"] = out["date"].dt.month.astype("int8")
    out = out.dropna(subset=["units", "promo_rate", "holiday_rate", "avg_store_transactions", "oil_price"]).reset_index(drop=True)
    return out


def _fit_residual_diagnostics(cluster_day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = cluster_day.copy()
    design = pd.get_dummies(
        overall[["promo_rate", "holiday_rate", "avg_store_transactions", "oil_price", "day_of_week", "month"]],
        columns=["day_of_week", "month"],
        drop_first=True,
        dtype=float,
    )
    design = sm.add_constant(design)
    model = sm.OLS(overall["units"], design).fit()
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    lb = acorr_ljungbox(model.resid, lags=[7, 14], return_df=True)

    overall_row = pd.DataFrame(
        [
            {
                "scope": "overall_cluster_day",
                "n_obs": int(len(overall)),
                "r_squared": float(model.rsquared),
                "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
                "breusch_pagan_f_pvalue": float(bp_f_pvalue),
                "ljung_box_pvalue_lag7": float(lb.loc[7, "lb_pvalue"]),
                "ljung_box_pvalue_lag14": float(lb.loc[14, "lb_pvalue"]),
                "residual_std": float(np.std(model.resid, ddof=1)),
                "heteroskedasticity_flag": bool(bp_lm_pvalue < 0.05),
                "autocorrelation_flag_lag7": bool(lb.loc[7, "lb_pvalue"] < 0.05),
                "autocorrelation_flag_lag14": bool(lb.loc[14, "lb_pvalue"] < 0.05),
            }
        ]
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for cluster_value, g in cluster_day.groupby("cluster"):
        if len(g) < 30:
            continue
        X = pd.get_dummies(
            g[["promo_rate", "holiday_rate", "avg_store_transactions", "oil_price", "day_of_week", "month"]],
            columns=["day_of_week", "month"],
            drop_first=True,
            dtype=float,
        )
        X = sm.add_constant(X)
        fitted = sm.OLS(g["units"], X).fit()
        bp = het_breuschpagan(fitted.resid, fitted.model.exog)
        lb_cluster = acorr_ljungbox(fitted.resid, lags=[7, 14], return_df=True)
        rows.append(
            {
                "cluster": cluster_value,
                "n_obs": int(len(g)),
                "r_squared": float(fitted.rsquared),
                "breusch_pagan_lm_pvalue": float(bp[1]),
                "breusch_pagan_f_pvalue": float(bp[3]),
                "ljung_box_pvalue_lag7": float(lb_cluster.loc[7, "lb_pvalue"]),
                "ljung_box_pvalue_lag14": float(lb_cluster.loc[14, "lb_pvalue"]),
                "residual_std": float(np.std(fitted.resid, ddof=1)),
                "heteroskedasticity_flag": bool(bp[1] < 0.05),
                "autocorrelation_flag_lag7": bool(lb_cluster.loc[7, "lb_pvalue"] < 0.05),
                "autocorrelation_flag_lag14": bool(lb_cluster.loc[14, "lb_pvalue"] < 0.05),
            }
        )

    return overall_row, pd.DataFrame(rows)


def main() -> None:
    args = _build_parser().parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path, columns=REQUIRED_COLUMNS)

    overall_count = pd.DataFrame([{"scope": "row_level_overall", **_count_stats(df["units"])}])
    by_cluster_count = pd.DataFrame(
        [
            {"cluster": cluster_value, **_count_stats(g["units"])}
            for cluster_value, g in df.groupby("cluster")
        ]
    ).sort_values("dispersion_ratio", ascending=False)

    cluster_day = _build_cluster_day(df)
    overall_resid, by_cluster_resid = _fit_residual_diagnostics(cluster_day)

    overall_count.to_csv(output_dir / "distribution_count_diagnostics_overall.csv", index=False)
    by_cluster_count.to_csv(output_dir / "distribution_count_diagnostics_by_cluster.csv", index=False)
    overall_resid.to_csv(output_dir / "residual_diagnostics_overall.csv", index=False)
    by_cluster_resid.to_csv(output_dir / "residual_diagnostics_by_cluster.csv", index=False)

    print("Distribution diagnostics complete.")
    print(f"Saved overall count stats to: {output_dir / 'distribution_count_diagnostics_overall.csv'}")
    print(f"Saved cluster count stats to: {output_dir / 'distribution_count_diagnostics_by_cluster.csv'}")
    print(f"Saved residual diagnostics to: {output_dir / 'residual_diagnostics_overall.csv'}")
    print(f"Saved residual diagnostics to: {output_dir / 'residual_diagnostics_by_cluster.csv'}")
    print(overall_count.to_string(index=False))
    print(overall_resid.to_string(index=False))


if __name__ == "__main__":
    main()
