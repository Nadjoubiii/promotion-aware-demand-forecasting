"""Train an ElasticNet model on cluster-day features and compare against baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import baseline_predictions, load_cluster_daily_dataset, metrics, save_metrics, save_prediction_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--output-dir", default="reports/modeling/elasticnet")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cluster_daily_dataset(args.data_path, args.cutoff_date)
    X_train = dataset.train_df[dataset.feature_cols]
    y_train = dataset.train_df[dataset.cfg.target_col]
    X_valid = dataset.valid_df[dataset.feature_cols]
    y_valid = dataset.valid_df[dataset.cfg.target_col]

    categorical_features = [dataset.cfg.segment_col]
    numeric_features = [column for column in dataset.feature_cols if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=10000, random_state=args.seed)),
        ]
    )
    model.fit(X_train, y_train)

    pred_elastic = pd.Series(model.predict(X_valid), index=dataset.valid_df.index)
    pred_baseline = baseline_predictions(dataset.valid_df)

    metrics_df = save_metrics(
        output_dir,
        [
            {"model": "elasticnet_cluster_daily", **metrics(y_valid, pred_elastic)},
            {"model": "seasonal_naive_7d", **metrics(y_valid, pred_baseline)},
        ],
    )
    save_prediction_frame(
        output_dir,
        dataset.valid_df,
        dataset.cfg,
        {"pred_elasticnet": pred_elastic},
    )

    print("ElasticNet training complete.")
    print(f"Saved outputs in: {output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
