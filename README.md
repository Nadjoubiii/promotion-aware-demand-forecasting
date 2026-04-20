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
- Interactive EDA notebook with missingness profiling, trend plots, promo comparisons, and segment summaries.
- Baseline forecasting benchmark script with time split and metrics exports.
- Statistical analysis layer with:
  - promo significance by cluster and store
  - business-action effect sizes and bootstrap confidence intervals for promotion and holiday actions
  - count-distribution diagnostics for Poisson vs Negative Binomial suitability
  - residual heteroskedasticity and autocorrelation checks
  - cluster-matched difference-in-differences cannibalization checks
- Advanced model benchmarks and rolling backtests for XGBoost, LightGBM, CatBoost, ElasticNet, and SARIMAX.

### Key Artifacts
- EDA notebook: `notebooks/eda_notebook.ipynb`
- Baseline script: `src/modeling/run_baseline_benchmark.py`
- Stats scripts: `src/stats/`
- Modeling scripts: `src/modeling/`
- EDA outputs: `reports/eda_notebook/`
- Baseline outputs: `reports/baseline/`
- Statistical outputs: `reports/stats/`
- Modeling outputs: `reports/modeling/`

### Latest Statistical Findings
- Promotion uplift is strongly positive across all 17 clusters in the action-effects pass, with top clusters showing roughly 97% to 138% uplift and large Cohen's d values.
- Holiday effects are actionable in 12 clusters with positive bootstrap confidence intervals.
- Row-level demand is heavily overdispersed (`dispersion_ratio` about 65 overall), so Negative Binomial assumptions fit better than Poisson.
- Residual diagnostics on cluster-day aggregates still show heteroskedasticity and autocorrelation across all clusters, which justifies moving beyond simple linear count assumptions.
- The current cluster-matched DiD cannibalization pass found no statistically strong negative spillover effects under the available control design.

### Latest Modeling Findings
- Single-split leader: XGBoost.
- Rolling backtests kept XGBoost, LightGBM, and SARIMAX all well ahead of the seasonal-naive baseline.
- SARIMAX was especially strong at 7-day and 14-day horizons, while XGBoost remained the strongest overall ML model.

### Next Steps
- Finalize model choice and package the model-selection story in the portfolio documentation.
- Build the decision layer, API, and dashboard on top of the validated forecasting outputs.
