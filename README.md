# Promotion-Aware Multi-Store Demand Forecasting & Inventory Intelligence Platform

End-to-end forecasting and decision-support platform for retail demand planning across stores and products.

## Objectives
- Forecast weekly demand by `store_id` x `product_id`.
- Quantify promotion lift, cannibalization, and uncertainty.
- Provide operational recommendations (restock risk, underperformance risk, low-confidence forecasts).
- Build a reproducible analytics + ML pipeline suitable for production-style deployment.

## Core Capabilities
- Multi-table ingestion (`sales`, `stores`, `products`, `promotions`, `calendar`, optional `weather/events`).
- Data quality checks for missing dates, invalid IDs, and stockout anomalies.
- Feature engineering for lags, rolling windows, seasonality, and promo/event context.
- Statistical analysis layer (hypothesis testing, confidence intervals, distribution checks, effect size).
- Baseline + advanced forecasting models with segment-wise evaluation.
- Prediction intervals and uncertainty flags.
- API + dashboard for operational decision support.

## Proposed Stack
- Python, Pandas/Polars
- PostgreSQL
- dbt-style transformation layer (or clean SQL + Python transformations)
- XGBoost / LightGBM / Statsmodels / Prophet
- FastAPI
- Streamlit
- MLflow
- Docker

## Suggested Deployment
- Frontend: Streamlit Community Cloud
- Backend API: Render or Railway
- Database: Neon Postgres
- Automation: GitHub Actions scheduled workflows

## Repository Layout
```text
.
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ external/
├─ docs/
├─ notebooks/
├─ src/
│  ├─ config/
│  ├─ data/
│  ├─ features/
│  ├─ modeling/
│  ├─ evaluation/
│  ├─ stats/
│  ├─ api/
│  └─ dashboard/
├─ tests/
├─ .github/workflows/
├─ .gitignore
└─ requirements.txt
```

## Quick Start (local)
1. Create and activate virtual environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set DB connection values.
4. Run baseline pipeline scripts from `src/`.

## Current Status

### Completed
- Favorita ingestion pipeline with optional Kaggle API download.
- Processed training table build (`data/processed/training_table.parquet`).
- Validation checks for negative units, null IDs, stockout candidates, and missing segment dates.
- Interactive EDA notebook with:
	- missingness profiling
	- trend and rolling demand plots
	- promo vs non-promo ECDF comparison
	- segment summaries
	- zero/negative sales checks
	- promo uplift by cluster
- Baseline forecasting benchmark script with time split and metrics exports.

### Key Artifacts
- EDA notebook: `notebooks/eda_notebook.ipynb`
- Baseline script: `src/modeling/run_baseline_benchmark.py`
- EDA outputs: `reports/eda_notebook/`
- Baseline outputs: `reports/baseline/`

### Latest Baseline Result (cluster-level, cutoff 2017-06-30)
- `seasonal_naive_7d` outperformed `moving_avg_7d` on MAE/RMSE/MAPE.
- Overall metrics are saved in `reports/baseline/metrics_overall.csv`.

### Next Steps
- Statistical analysis phase: promo significance tests, confidence intervals, and effect sizes.
- Add stronger forecasting models and compare against the seasonal-naive baseline.
