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

- Summary: Statistical analyses confirm strong, reproducible promotion and holiday uplifts, and identify important distributional features of demand (overdispersion and residual correlation) that informed model choices and uncertainty quantification.

- Promotion & Holiday Effects: Bootstrap-estimated effect sizes show consistently positive, and often large, uplift from promotions (top clusters ~97%–138%). Holiday effects are actionable in 12 clusters. Effect sizes were reported as Cohen's d and supported by bootstrap 95% confidence intervals; significance was adjusted with FDR to control false positives across many tests.

- Count Distribution Diagnostics: Row-level demand exhibits strong overdispersion and modest zero-inflation. The estimated dispersion ratio (~65) and likelihood comparisons favor Negative Binomial over Poisson for count-modeling assumptions. This motivated using dispersion-aware loss functions and model choices that handle heavy-tailed count behavior.

- Residual Diagnostics & Robust Errors: Residuals from cluster-aggregated fits show heteroskedasticity (Breusch–Pagan) and short-range autocorrelation (Ljung–Box), so prediction intervals and inference use robust standard errors, block bootstrap for serial dependence, and conservative CI estimators where appropriate.

- Difference-in-Differences (Cannibalization): A cluster-matched DiD pipeline tested spillovers using matched controls, covariate balance checks, and bootstrap CIs. Under the chosen control design and sample sizes, we found no strong evidence of negative cannibalization effects. Notes: control selection is conservative by design; relaxing matching thresholds increases power but may reduce causal validity.

- Reproducibility & Artifacts: All analyses are reproducible via `src/stats/` scripts and produce CSV outputs in `reports/stats/`. Key scripts:
  - `src/stats/run_distribution_diagnostics.py` — dispersion, zero-inflation, and residual tests
  - `src/stats/run_business_action_effects.py` — bootstrap uplifts, Cohen's d, FDR correction
  - `src/stats/run_cannibalization_did.py` — matched DiD cannibalization checks

- Practical Implications: Use Negative Binomial or dispersion-aware ML losses for forecasting; retain robust/resampled CIs for business decisions; treat promotion uplift estimates as actionable inputs for inventory and pricing decisions; continue monitoring for localized spillovers (re-run DiD periodically as campaigns change).

For reproducible outputs and detailed tables/figures, see `reports/stats/`.

### Latest Modeling Findings
- Single-split leader: XGBoost.
- Rolling backtests kept XGBoost, LightGBM, and SARIMAX all well ahead of the seasonal-naive baseline.
- SARIMAX was especially strong at 7-day and 14-day horizons, while XGBoost remained the strongest overall ML model.

### Next Steps
- Finalize model choice and package the model-selection story in the portfolio documentation.
- Build the decision layer, API, and dashboard on top of the validated forecasting outputs.
