"""Train a CatBoost model on cluster-day features and compare against baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import baseline_predictions, load_cluster_daily_dataset, metrics, save_metrics, save_prediction_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling/catboost")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cluster_daily_dataset(args.data_path, args.cutoff_date)
    X_train = dataset.train_df[dataset.feature_cols].copy()
    y_train = dataset.train_df[dataset.cfg.target_col]
    X_valid = dataset.valid_df[dataset.feature_cols].copy()
    y_valid = dataset.valid_df[dataset.cfg.target_col]

    cat_features = [X_train.columns.get_loc(dataset.cfg.segment_col)]

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=args.seed,
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_features)

    pred_cat = pd.Series(model.predict(X_valid), index=dataset.valid_df.index)
    pred_baseline = baseline_predictions(dataset.valid_df)

    metrics_df = save_metrics(
        output_dir,
        [
            {"model": "catboost_cluster_daily", **metrics(y_valid, pred_cat)},
            {"model": "seasonal_naive_7d", **metrics(y_valid, pred_baseline)},
        ],
    )
    save_prediction_frame(
        output_dir,
        dataset.valid_df,
        dataset.cfg,
        {"pred_catboost": pred_cat},
    )

    feature_importance = pd.DataFrame(
        {"feature": dataset.feature_cols, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    model.save_model(output_dir / "catboost_model.cbm")

    print("CatBoost training complete.")
    print(f"Saved outputs in: {output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
