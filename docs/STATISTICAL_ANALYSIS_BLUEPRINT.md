# Statistical Analysis Blueprint

## Why this matters
Demand forecasting quality depends on understanding causal and structural demand patterns, not just fitting a black-box model.

## Core statistical tracks
1. Time-series diagnostics
- Stationarity checks (ADF/KPSS where appropriate)
- Seasonal decomposition and changepoint screening
- Autocorrelation and partial autocorrelation assessment

2. Promotion impact inference
- Uplift estimation by campaign type, depth, and duration
- Confidence intervals for uplift effect
- Significance tests with multiple-testing control when needed

3. Event and holiday effects
- Event window analysis
- Pre/post tests and effect size reporting
- Interaction effects (promotion x holiday)

4. Cross-store heterogeneity
- Hierarchical comparisons by region/store format
- Mixed-effects style thinking for pooling where sample sizes are small
- Outlier store behavior detection

5. Stockout-aware demand correction
- Identify censored demand periods
- Estimate latent demand during stockout windows
- Compare corrected vs observed demand series

6. Uncertainty quantification
- Prediction intervals with coverage tracking
- Calibration checks by segment
- Risk-tier labels for operational use

## Recommended outputs per sprint
- Statistical memo (1-2 pages): findings and business interpretation.
- Notebook with diagnostics and test results.
- Artifact table with effect estimates and confidence intervals.
- Segment reliability scoreboard.

## Minimum statistical acceptance criteria
- At least one validated promotion uplift method.
- At least one uncertainty calibration report.
- Segment-wise residual diagnostics documented.
- Business decisions linked to statistically supported evidence.
