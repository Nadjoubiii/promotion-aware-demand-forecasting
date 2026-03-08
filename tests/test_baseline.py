import pandas as pd

from src.modeling.baseline import seasonal_naive_last_week


def test_seasonal_naive_shift():
    df = pd.DataFrame({"units": [10, 11, 12, 13, 14, 15, 16, 17]})
    pred = seasonal_naive_last_week(df, target_col="units", season_lag=7)
    assert pd.isna(pred.iloc[0])
    assert pred.iloc[7] == 10
