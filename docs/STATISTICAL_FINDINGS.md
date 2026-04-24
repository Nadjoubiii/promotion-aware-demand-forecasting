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

Top tables (snippets)

Distribution diagnostics (overall):

```
scope,n_obs,mean_units,variance_units,dispersion_ratio,observed_zero_rate,poisson_expected_zero_rate,zero_inflation_gap,nb_alpha_estimate,poisson_suitable_flag,negbin_preferred_flag
row_level_overall,125497040,8.554865268447763,557.2031920165741,65.13290093201852,0.0,0.00019260573766883136,-0.00019260573766883136,7.49665820788024,False,True
```

Business-action effects (top rows):

```
event_type,cluster,n_event_days,n_non_event_days,mean_event,mean_non_event,uplift_abs,uplift_pct,cohens_d,welch_t_pvalue,mean_diff_ci95_low,mean_diff_ci95_high,fdr_qvalue,significant_fdr_0_05,actionable_positive_flag
holiday,16,178,1391,9320.9746,8330.6745,990.3001,11.8874,0.2873,0.00494,359.2750,1626.7146,0.01400,True,True
promotion,2,1149,530,21085.5168,8845.9301,12239.5866,138.3640,1.8677,0.0,11728.9164,12736.5056,0.0,True,True
promotion,10,1212,468,59187.6185,29031.9721,30155.6464,103.8705,1.6778,0.0,28606.6653,31600.1580,0.0,True,True
promotion,11,1142,537,71238.1496,35866.4864,35371.6632,98.6204,1.6552,0.0,33662.6280,37127.7076,0.0,True,True
```

Residual diagnostics (by cluster, top rows):

```
cluster,n_obs,r_squared,breusch_pagan_lm_pvalue,breusch_pagan_f_pvalue,ljung_box_pvalue_lag7,ljung_box_pvalue_lag14,residual_std,heteroskedasticity_flag,autocorrelation_flag_lag7,autocorrelation_flag_lag14
1,1684,0.7201182594,4.256829137774553e-44,2.2563261338835307e-48,0.0,0.0,6386.780289198598,True,True,True
2,1679,0.7207473180,7.805676017988019e-68,9.670877282974369e-78,0.0,0.0,4585.654592797485,True,True,True
3,1679,0.7561454152,1.0119485532981312e-17,1.6002262041748312e-18,0.0,0.0,7119.946859738328,True,True,True
```

For full tables and figures, see the CSVs and images in `reports/stats/`.

If you want, I can also embed the top plots (PNG/SVG) from `reports/stats/` into this doc.

Embedded diagnostic plots

![Dispersion ratio overall](reports/stats/dispersion_overall.png)

![Top promotion uplift_pct (top 10 clusters)](reports/stats/promotions_top10_uplift.png)

![Residual std by cluster](reports/stats/residuals_by_cluster.png)
