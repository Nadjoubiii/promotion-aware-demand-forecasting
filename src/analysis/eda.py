"""Minimal EDA starter script."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/processed/training_table.parquet")
PLOTS_DIR = Path("reports/eda")
MAX_PLOT_SAMPLE = 2_000_000


def _save_plot(name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=150)
    plt.close()


def build_plots(df: pd.DataFrame) -> None:
    """Create and save a small set of high-value EDA plots."""
    sns.set_theme(style="whitegrid")

    # 1) Daily demand trend + rolling mean
    if {"date", "units"}.issubset(df.columns):
        daily = (
            df[["date", "units"]]
            .dropna()
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .groupby("date", as_index=False)["units"]
            .sum()
            .sort_values("date")
        )
        daily["rolling_28d"] = daily["units"].rolling(28, min_periods=1).mean()

        plt.figure(figsize=(12, 4))
        plt.plot(daily["date"], daily["units"], alpha=0.35, label="daily units")
        plt.plot(daily["date"], daily["rolling_28d"], linewidth=2, label="28d rolling mean")
        plt.title("Total Daily Units Sold")
        plt.xlabel("Date")
        plt.ylabel("Units")
        plt.legend()
        _save_plot("01_daily_units_trend.png")

    # Prepare sample for heavier distribution plots
    plot_df = df
    if len(df) > MAX_PLOT_SAMPLE:
        plot_df = df.sample(MAX_PLOT_SAMPLE, random_state=42)

    # 2) Promotion effect distribution
    if {"units", "on_promotion"}.issubset(plot_df.columns):
        promo = plot_df[["units", "on_promotion"]].dropna().copy()
        promo["on_promotion"] = promo["on_promotion"].astype(bool)
        promo["units_clipped"] = promo["units"].clip(upper=promo["units"].quantile(0.99))

        plt.figure(figsize=(8, 4))
        sns.boxplot(data=promo, x="on_promotion", y="units_clipped")
        plt.title("Units Distribution: Promotion vs Non-Promotion")
        plt.xlabel("On Promotion")
        plt.ylabel("Units (clipped at 99th percentile)")
        _save_plot("02_promo_vs_nonpromo_units.png")

    # 3) Top stores by total units
    if {"store_id", "units"}.issubset(df.columns):
        top_stores = (
            df[["store_id", "units"]]
            .dropna()
            .groupby("store_id", as_index=False)["units"]
            .sum()
            .sort_values("units", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(data=top_stores, x="store_id", y="units", color="#2a9d8f")
        plt.title("Top 15 Stores by Total Units Sold")
        plt.xlabel("Store ID")
        plt.ylabel("Total Units")
        _save_plot("03_top_stores_total_units.png")

    # 4) Missingness profile by column
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_plot = missing_pct[missing_pct > 0].head(20)
    if not missing_plot.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing_plot.values, y=missing_plot.index, color="#e76f51")
        plt.title("Top Missing Columns (%)")
        plt.xlabel("Missing %")
        plt.ylabel("Column")
        _save_plot("04_missingness_top_columns.png")


def main() -> None:
    """Run exploratory data analysis steps."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print("=== Dataset Overview ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column names: {', '.join(df.columns)}")

    print("\n=== Dtypes ===")
    print(df.dtypes)

    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce")
        print("\n=== Date Range ===")
        print(f"Min date: {date_series.min()}")
        print(f"Max date: {date_series.max()}")

    print("\n=== Missing Values ===")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct.round(2)})
    print(missing_table[missing_table["missing_count"] > 0].head(30))

    print("\n=== Key Counts ===")
    if "store_id" in df.columns:
        print(f"Unique stores: {df['store_id'].nunique():,}")
    if "product_id" in df.columns:
        print(f"Unique products: {df['product_id'].nunique():,}")
    if {"store_id", "product_id"}.issubset(df.columns):
        print(f"Unique store-product pairs: {df[['store_id', 'product_id']].drop_duplicates().shape[0]:,}")

    print("\n=== Sample Rows ===")
    print(df.head(5))

    build_plots(df)
    print(f"\nSaved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
