"""Quick, short hyperparameter tuning for XGBoost, LightGBM, and a small SARIMAX grid.

Usage:
  python scripts/quick_tune.py --data-path data/processed/training_table.parquet

Notes:
- Designed to be fast: small random search trials and early stopping.
- Requires `data/processed/training_table.parquet` to exist (same format as training pipeline).
"""
from __future__ import annotations

import argparse
import json
import os
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
    # ensure pandas Series for safe operations
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    mae = mean_absolute_error(y_true, y_pred)
    # compute RMSE without relying on sklearn 'squared' kw for compatibility
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE: avoid division by zero, use stable definition
    denom = y_true.replace(0, np.nan)
    mape = float((np.mean(np.abs((y_true - y_pred) / denom))) * 100)
    if np.isnan(mape):
        mape = float(np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-9) * 100)
    return {"mae": float(mae), "rmse": rmse, "mape_pct": mape}


def random_search_xgb(X_train, y_train, X_valid, y_valid, output_dir, n_trials=6, seed=42):
    random.seed(seed)
    # include original training defaults (n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
    param_grid = {
        "n_estimators": [200, 500, 800],
        "max_depth": [6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    results = []
    best = None
    for i in range(n_trials):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"[XGB] Trial {i+1}/{n_trials} — params: {params}")
        n_estimators = params.pop("n_estimators", 200)
        model = XGBRegressor(
            n_estimators=n_estimators,
            objective="reg:squarederror",
            random_state=seed + i,
            n_jobs=-1,
            **params,
        )
        try:
            model.fit(
                X_train,
                y_train,
                early_stopping_rounds=20,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        except TypeError:
            print("[XGB] early_stopping_rounds not accepted by this xgboost build; fitting without early stopping")
            model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        m = metrics(y_valid, pd.Series(preds, index=y_valid.index))
        results.append({"trial": i, **params, **m})
        print(f"[XGB] Trial {i+1} results: MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}, MAPE%={m['mape_pct']:.2f}")
        if best is None or m["mae"] < best["mae"]:
            best = {**m, "params": params}
            model.save_model(str(output_dir / "xgboost_best.json"))
            print(f"[XGB] New best MAE={best['mae']:.1f} saved to xgboost_best.json")
    pd.DataFrame(results).to_csv(output_dir / "xgboost_random_search.csv", index=False)
    return best


def random_search_lgb(X_train, y_train, X_valid, y_valid, output_dir, n_trials=6, seed=42):
    random.seed(seed)
    # include original training defaults (n_estimators=500, learning_rate=0.05, num_leaves=63, subsample/bagging=0.8)
    param_grid = {
        "n_estimators": [200, 500, 800],
        "num_leaves": [63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "feature_fraction": [0.6, 0.8, 1.0],
        "bagging_fraction": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 2.0],
    }
    results = []
    best = None
    for i in range(n_trials):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"[LGB] Trial {i+1}/{n_trials} — params: {params}")
        n_estimators = params.pop("n_estimators", 200)
        model = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=seed + i, **params)
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=20,
                verbose=False,
            )
        except TypeError:
            print("[LGB] early_stopping_rounds not accepted by this lightgbm build; fitting without early stopping")
            model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        m = metrics(y_valid, pd.Series(preds, index=y_valid.index))
        results.append({"trial": i, **params, **m})
        print(f"[LGB] Trial {i+1} results: MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}, MAPE%={m['mape_pct']:.2f}")
        if best is None or m["mae"] < best["mae"]:
            best = {**m, "params": params}
            model.booster_.save_model(str(output_dir / "lightgbm_best.txt"))
            print(f"[LGB] New best MAE={best['mae']:.1f} saved to lightgbm_best.txt")
    pd.DataFrame(results).to_csv(output_dir / "lightgbm_random_search.csv", index=False)
    return best


def tiny_sarimax_grid(y_train, y_valid, output_dir):
    # Aggregate series to speed up SARIMAX tuning (short quick check)
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
            print(f"[SARIMAX] order={order} MAE={m['mae']:.1f}, RMSE={m['rmse']:.1f}")
            if best is None or m["mae"] < best["mae"]:
                best = {**m, "order": order}
                print(f"[SARIMAX] New best order={order} MAE={best['mae']:.1f}")
        except Exception as e:
            print(f"[SARIMAX] order={order} failed: {e}")
            continue
    pd.DataFrame(results).to_csv(output_dir / "sarimax_grid.csv", index=False)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling/quick_tune")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_cluster_daily_dataset(args.data_path, args.cutoff_date)
    X_train = ds.train_df[ds.feature_cols]
    y_train = ds.train_df[ds.cfg.target_col]
    X_valid = ds.valid_df[ds.feature_cols]
    y_valid = ds.valid_df[ds.cfg.target_col]

    print('Running quick XGBoost random search...')
    best_xgb = random_search_xgb(X_train, y_train, X_valid, y_valid, out, n_trials=args.trials, seed=args.seed)
    best_lgb = None
    if LGB_AVAILABLE:
        print('Running quick LightGBM random search...')
        best_lgb = random_search_lgb(X_train, y_train, X_valid, y_valid, out, n_trials=args.trials, seed=args.seed)
    else:
        print('[LGB] lightgbm not installed in this environment; skipping LightGBM tuning')

    print('Running tiny SARIMAX grid on aggregated series...')
    try:
        best_sarimax = tiny_sarimax_grid(y_train, y_valid, out)
    except Exception as e:
        best_sarimax = {"error": str(e)}

    summary = {"xgboost": best_xgb, "lightgbm": best_lgb, "sarimax": best_sarimax}
    with open(out / 'quick_tune_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print('Quick tuning complete. Summary saved to', out)


if __name__ == '__main__':
    main()
