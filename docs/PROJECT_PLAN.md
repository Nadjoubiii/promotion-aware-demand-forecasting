# Project Plan

## Phase 0: Repository and Scope (Day 1-2)
- Finalize problem framing and KPIs.
- Define data schema and assumptions.
- Set up repository, coding standards, and folder structure.

## Phase 1: Data Foundation (Week 1)
- Ingest or simulate realistic multi-store multi-product data.
- Create normalized tables: sales, stores, products, promotions, calendar/events.
- Implement quality checks:
  - Missing date continuity per store-product.
  - Invalid foreign keys (`store_id`, `product_id`).
  - Negative units/prices.
  - Potential stockout periods.

## Phase 2: Statistical Analysis Layer (Week 2, completed)
- Exploratory decomposition by segment (trend/seasonality/event effects).
- Hypothesis testing:
  - Promotion uplift significance by category/store cluster.
  - Holiday impact significance.
  - Mean/variance comparison pre-vs-post campaign windows.
- Distribution diagnostics:
  - Count-data behavior (Poisson/NegBin suitability checks) completed via `src/stats/run_distribution_diagnostics.py` and `reports/stats/distribution_count_diagnostics_*.csv`.
  - Heteroskedasticity and residual structure checks completed via `reports/stats/residual_diagnostics_*.csv`.
- Effect size and confidence intervals for business actions completed via `src/stats/run_business_action_effects.py` and `reports/stats/business_action_effects.csv`.
- Cannibalization analysis:
  - Difference-in-differences style checks across cluster-matched control stores completed via `src/stats/run_cannibalization_did.py` and `reports/stats/cannibalization_did_by_cluster_product_class.csv`.
  - Current result: no statistically strong negative cannibalization flags in the first DiD pass.

## Phase 3: Forecasting Models (Week 3-4)
- Baselines: seasonal naive, moving average, simple regression.
- Statistical models: SARIMAX/ETS by segment.
- ML models: gradient boosting with lag/rolling/event features.
- Backtesting by rolling-origin validation.
- Segment-level model selection (store/product/category clusters).
- Prediction intervals and uncertainty calibration.

## Phase 4: Decision Intelligence (Week 5)
- Rule-based recommendation engine:
  - Restock alerts for expected shortfall.
  - Underperformance warning by confidence-adjusted forecast error risk.
  - Promotion distortion/cannibalization flag.
- Reliability scoring:
  - Stable, medium-risk, high-uncertainty forecast classes.

## Phase 5: Productization (Week 6)
- FastAPI endpoints for forecast/recommendation retrieval.
- Streamlit dashboard for operators and store managers.
- Model/version tracking with MLflow.
- Dockerized local deployment.

## Phase 6: Hardening (Optional Weeks 7-10)
- CI/CD via GitHub Actions.
- Scheduled retraining and batch inference.
- Data drift and forecast drift monitoring.
- Test suite expansion and performance optimization.

## Deliverables
- Reproducible pipeline.
- Statistical analysis report with business interpretation.
- Forecast service API.
- Decision dashboard.
- Public GitHub repo with documentation and demo screenshots.
