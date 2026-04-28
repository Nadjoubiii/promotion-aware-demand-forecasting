"""Inspect training/validation DataFrame structure for SARIMAX deep-tuning.

Prints index structure, candidate group columns, sample series ids, and suggestions
for re-running `deep_tune_sarimax.py` with the correct grouping column.

Usage:
  python scripts/deep_tune_sarimax_check.py --data-path data/processed/training_table.parquet --cutoff-date 2017-06-30
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.common import load_cluster_daily_dataset


def find_group_columns(df):
    candidates = []
    for name in ("series_id", "cluster_id", "cluster", "group_id", "id"):
        if name in df.columns:
            candidates.append(name)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/training_table.parquet")
    parser.add_argument("--cutoff-date", default="2017-06-30")
    parser.add_argument("--sample-series", type=int, default=5,
                        help="How many sample series ids to print (when available)")
    args = parser.parse_args()

    ds = load_cluster_daily_dataset(args.data_path, args.cutoff_date)
    train = ds.train_df
    valid = ds.valid_df

    print("=== TRAIN DataFrame ===")
    print("shape:", train.shape)
    print("index names:", train.index.names)
    print("columns:", list(train.columns[:50]))
    print()

    print("=== VALID DataFrame ===")
    print("shape:", valid.shape)
    print("index names:", valid.index.names)
    print("columns:", list(valid.columns[:50]))
    print()

    # MultiIndex detection
    if isinstance(train.index, pd.MultiIndex) and train.index.nlevels >= 2:
        print("Train index is MultiIndex with levels:", train.index.names)
        try:
            series_ids = list(train.index.get_level_values(0).unique())
            print(f"Sample {args.sample_series} series ids (level 0):", series_ids[:args.sample_series])
        except Exception as e:
            print("Could not list series ids from MultiIndex level 0:", e)
    else:
        print("Train index is not a MultiIndex (or has <2 levels).")
        candidates = find_group_columns(train)
        print("Candidate group columns found:", candidates)
        if candidates:
            for c in candidates:
                try:
                    vals = train[c].unique()[:args.sample_series]
                    print(f"Sample values for column '{c}':", list(vals))
                except Exception:
                    pass

    # Show a small head of train and valid for manual inspection
    print('\nTrain head:')
    with pd.option_context('display.max_columns', 20, 'display.max_rows', 10):
        print(train.head())

    print('\nValid head:')
    with pd.option_context('display.max_columns', 20, 'display.max_rows', 10):
        print(valid.head())

    print('\nSuggestion:')
    print("- If you see a MultiIndex where level 0 is your series id, `deep_tune_sarimax.py` should pick it up.")
    print("- If your series ids live in a column (e.g., 'cluster_id'), re-run `deep_tune_sarimax.py` with that column present or tell me and I can add a `--group-col` option.")


if __name__ == '__main__':
    main()
