# Retail Demand Forecasting System

> End-to-end demand forecasting and promotion analytics platform for a multi-region grocery chain — 125M+ transaction records, 17 store clusters, daily cadence.

**Live dashboard →** run `streamlit run app/dashboard.py` (see Quick-start below).

---

## The Business Problem

A grocery retail chain needed accurate short-horizon demand forecasts to drive replenishment decisions and evaluate promotional effectiveness. The existing weekly naive baseline produced unacceptably high errors and gave planners no insight into *why* demand moved.

**What was built:**

| Deliverable | Detail |
|---|---|
| Production forecasting pipeline | Cluster-level daily models with automated feature engineering |
| Multi-model evaluation | 7 models compared on an identical holdout window |
| Promotional analytics | Per-cluster uplift, effect size, and statistical significance |
| Cannibalization detection | Difference-in-Differences across product classes |
| Planner dashboard | Interactive Streamlit app for daily use |

---

## Results at a Glance

| Model | MAE | MAPE | vs Naive |
|---|---|---|---|
| **XGBoost** *(best)* | **2,281** | **5.2%** | **−60%** |
| LightGBM | 2,322 | 5.5% | −59% |
| CatBoost | 2,507 | 6.0% | −56% |
| SARIMAX (tuned) | 3,073 | 6.6% | −46% |
| Seasonal Naive *(baseline)* | 5,697 | 12.0% | — |

XGBoost and LightGBM were both tuned with deep grid search (1,000-estimator runs). SARIMAX received per-cluster ARIMA order selection.

---

## Statistical Findings

- **Promotions**: significant uplift in 14/17 clusters (FDR-corrected Welch t-test, α = 0.05), median +18% lift.
- **Holidays**: significant uplift in 12/17 clusters, median +11% lift.
- **Cannibalization**: no significant cross-class demand destruction detected (DiD analysis across all campaigns).
- **Demand distribution**: highly overdispersed (variance/mean ratio 65×), Negative Binomial preferred over Poisson in every cluster — motivates gradient boosting over linear models.

---

## Quick-start

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd forecasting-system
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                       # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app/dashboard.py
```

**Docker (one command):**
```bash
docker build -t demand-forecast . && docker run -p 8501:8501 demand-forecast
```

---

## Repository Layout

```text
.
├── app/
│   └── dashboard.py          # Streamlit portfolio dashboard
├── src/
│   ├── modeling/             # Training scripts (XGBoost, LightGBM, SARIMAX, etc.)
│   ├── features/             # Feature engineering pipeline
│   ├── stats/                # Statistical analysis (promo, cannibalization, distribution)
│   └── evaluation/           # Metrics and residual diagnostics
├── scripts/
│   └── deep_tune_*.py        # Hyperparameter tuning scripts
├── reports/
│   ├── modeling/             # Per-model metrics, predictions, feature importance
│   └── stats/                # Statistical analysis outputs
├── tests/                    # Pytest unit tests
├── data/
│   └── processed/            # Parquet feature table (125M+ rows)
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml  # GitHub Actions CI
```

---

## Stack

| Layer | Technology |
|---|---|
| Core ML | XGBoost, LightGBM, CatBoost, Statsmodels (SARIMAX) |
| Feature engineering | Pandas, NumPy — lags, rolling windows, promo/event encodings |
| Statistics | Welch t-test, FDR correction, Cohen's d, DiD, NegBin dispersion |
| Visualisation | Plotly, Streamlit |
| CI | GitHub Actions |
| Deployment | Docker / Streamlit Community Cloud |

---

## License

MIT — see [LICENSE](LICENSE).

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
