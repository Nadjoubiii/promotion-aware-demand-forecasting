# Statistical Findings

Summary

- Statistical analyses confirm strong, reproducible promotion and holiday uplifts, and highlight distributional features of demand that guided model and uncertainty choices.

Key Results

- Promotion & Holiday Effects: Bootstrap-estimated uplifts are consistently positive; top clusters show ~97%–138% promotion uplift. Holiday effects are actionable in 12 clusters. Significance was adjusted using FDR and effect sizes reported as Cohen's d with bootstrap 95% CIs.

- Count Distribution: Row-level demand is heavily overdispersed (dispersion ratio ≈ 65) with modest zero-inflation; likelihood comparisons favor Negative Binomial over Poisson. This motivated dispersion-aware losses and model choices.

- Residual Diagnostics: Cluster-aggregated residuals exhibit heteroskedasticity (Breusch–Pagan) and short-range autocorrelation (Ljung–Box). Inference uses robust standard errors and block bootstrap where appropriate.

- Cannibalization (DiD): Cluster-matched difference-in-differences tests found no strong evidence of negative spillovers under the conservative control design. Relaxing match thresholds increases power but may reduce causal validity.

Reproducibility

- Scripts: `src/stats/run_distribution_diagnostics.py`, `src/stats/run_business_action_effects.py`, `src/stats/run_cannibalization_did.py`.
- Outputs: `reports/stats/` (CSV tables and diagnostic figures).

Practical Implications

- Use Negative Binomial or dispersion-aware ML objectives for forecasting counts.
- Report robust/resampled CIs for business decisions, and re-run DiD checks after major campaign changes.

If you want, I can embed top-table snippets and figures from `reports/stats/` into this document next.
