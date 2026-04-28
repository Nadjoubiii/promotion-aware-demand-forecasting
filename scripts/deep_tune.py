"""Deeper hyperparameter tuning (random search) for XGBoost, LightGBM.

This script runs a larger random-search budget and saves per-trial results and best models.

Usage:
  python scripts/deep_tune.py --data-path data/processed/training_table.parquet --trials 30

Notes:
- Designed to be run locally (may take time). Set `--trials` to control budget.
- Uses early stopping when supported; falls back if build doesn't accept early-stopping kw.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

try:
    import statsmodels.api as sm
    SM_AVAILABLE = True
except Exception:
    sm = None
    SM_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import load_cluster_daily_dataset


def metrics(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = y_true.replace(0, np.nan)
    mape = float((np.mean(np.abs((y_true - y_pred) / denom))) * 100)
    if np.isnan(mape):
        mape = float(np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-9) * 100)
    return {"mae": float(mae), "rmse": rmse, "mape_pct": mape}


def sample(params):
    return {k: random.choice(v) for k, v in params.items()}


def tune_xgb(X_train, y_train, X_valid, y_valid, out_dir, trials=30, seed=42):
    param_grid = {
        "n_estimators": [500, 800, 1000],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    trials_out = []
    best = None
    for t in range(trials):
        p = sample(param_grid)
        n_est = p.pop("n_estimators")
        print(f"[XGB] Trial {t+1}/{trials} params: n_estimators={n_est}, {p}")
        model = XGBRegressor(n_estimators=n_est, objective="reg:squarederror", n_jobs=-1, random_state=seed + t, **p)
        try:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=30, verbose=False)
        except TypeError:
            model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        m = metrics(y_valid, preds)
        trials_out.append({"trial": t + 1, **p, "n_estimators": n_est, **m})
        print(f"[XGB] Trial {t+1} => MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}")
        if best is None or m['mae'] < best['mae']:
            best = {**m, "params": {**p, "n_estimators": n_est}}
            model.save_model(str(out_dir / "xgboost_deep_best.json"))
            print(f"[XGB] New best saved (MAE={best['mae']:.1f})")
    pd.DataFrame(trials_out).to_csv(out_dir / "xgboost_deep_trials.csv", index=False)
    return best


def tune_lgb(X_train, y_train, X_valid, y_valid, out_dir, trials=30, seed=42):
    if not LGB_AVAILABLE:
        print("[LGB] lightgbm not installed; skipping")
        return None
    param_grid = {
        "n_estimators": [500, 800, 1000],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05],
        "feature_fraction": [0.6, 0.8, 1.0],
        "bagging_fraction": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 2.0],
    }
    trials_out = []
    best = None
    for t in range(trials):
        p = sample(param_grid)
        n_est = p.pop("n_estimators")
        print(f"[LGB] Trial {t+1}/{trials} params: n_estimators={n_est}, {p}")
        model = lgb.LGBMRegressor(n_estimators=n_est, random_state=seed + t, **p)
        try:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=30, verbose=False)
        except TypeError:
            model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        m = metrics(y_valid, preds)
        trials_out.append({"trial": t + 1, **p, "n_estimators": n_est, **m})
        print(f"[LGB] Trial {t+1} => MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}")
        if best is None or m['mae'] < best['mae']:
            best = {**m, "params": {**p, "n_estimators": n_est}}
            try:
                model.booster_.save_model(str(out_dir / "lightgbm_deep_best.txt"))
            except Exception:
                pass
            print(f"[LGB] New best saved (MAE={best['mae']:.1f})")
    pd.DataFrame(trials_out).to_csv(out_dir / "lightgbm_deep_trials.csv", index=False)
    return best


def tiny_sarimax_grid(y_train, y_valid, out_dir):
    if not SM_AVAILABLE:
        print("[SARIMAX] statsmodels not installed; skipping")
        return None
    # keep SARIMAX optional and light: aggregated test only
    ytr = y_train.groupby(level=[0]).sum() if hasattr(y_train, 'index') else y_train
    yva = y_valid.groupby(level=[0]).sum() if hasattr(y_valid, 'index') else y_valid
    orders = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
    results = []
    best = None
    for order in orders:
        print(f"[SARIMAX] Trying order={order}")
        try:
            mod = sm.tsa.statespace.SARIMAX(ytr, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            preds = res.get_forecast(steps=len(yva)).predicted_mean
            m = metrics(yva.reset_index(drop=True), pd.Series(preds))
            results.append({"order": order, **m})
            print(f"[SARIMAX] order={order} => MAE={m['mae']:.1f}")
            if best is None or m['mae'] < best['mae']:
                best = {**m, "order": order}
        except Exception as e:
            print(f"[SARIMAX] order={order} failed: {e}")
            continue
    pd.DataFrame(results).to_csv(out_dir / "sarimax_deep_trials.csv", index=False)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling/deep_tune")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_cluster_daily_dataset(args.data_path, args.cutoff_date)
    X_train = ds.train_df[ds.feature_cols]
    y_train = ds.train_df[ds.cfg.target_col]
    X_valid = ds.valid_df[ds.feature_cols]
    y_valid = ds.valid_df[ds.cfg.target_col]

    best_xgb = tune_xgb(X_train, y_train, X_valid, y_valid, out, trials=args.trials, seed=args.seed)
    best_lgb = tune_lgb(X_train, y_train, X_valid, y_valid, out, trials=args.trials, seed=args.seed)
    best_sarimax = tiny_sarimax_grid(y_train, y_valid, out)

    summary = {"xgboost": best_xgb, "lightgbm": best_lgb, "sarimax": best_sarimax}
    with open(out / 'deep_tune_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print('Deep tuning complete. Summary saved to', out)


if __name__ == '__main__':
    main()
