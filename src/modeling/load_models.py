"""
Model artifact loader.

Usage
-----
from src.modeling.load_models import load_best_model, BEST_MODEL_KEY

model = load_best_model()          # returns the best model (XGBoost)
lgbm  = load_best_model("lightgbm")
xgb   = load_best_model("xgboost")
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "reports" / "modeling"

# Model ranked #1 by MAE on holdout
BEST_MODEL_KEY = "xgboost"


def load_best_model(name: str = BEST_MODEL_KEY):
    """Load a saved model artifact.

    Parameters
    ----------
    name:
        One of ``"xgboost"`` (default) or ``"lightgbm"``.

    Returns
    -------
    Loaded model object.
    """
    name = name.lower()

    if name == "xgboost":
        import xgboost as xgb

        path = _ARTIFACTS / "xgboost" / "xgboost_model.json"
        if not path.exists():
            raise FileNotFoundError(f"XGBoost artifact not found at {path}")
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster

    if name == "lightgbm":
        import lightgbm as lgb

        path = _ARTIFACTS / "lightgbm" / "lightgbm_model.txt"
        if not path.exists():
            raise FileNotFoundError(f"LightGBM artifact not found at {path}")
        return lgb.Booster(model_file=str(path))

    raise ValueError(f"Unknown model '{name}'. Choose 'xgboost' or 'lightgbm'.")


def load_metrics(model: str | None = None) -> dict:
    """Return evaluation metrics.

    Parameters
    ----------
    model:
        If provided, return metrics for that model only. Otherwise return
        the full comparison dict keyed by model name.
    """
    import csv

    path = _ARTIFACTS / "metrics_comparison.csv"
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["model"]] = {
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "mape_pct": float(row["mape_pct"]),
            }

    if model is not None:
        return rows[model]
    return rows


if __name__ == "__main__":
    metrics = load_metrics()
    print("Model leaderboard (MAE):")
    for name, m in metrics.items():
        print(f"  {name:<35} MAE={m['mae']:,.0f}  MAPE={m['mape_pct']:.1f}%")
