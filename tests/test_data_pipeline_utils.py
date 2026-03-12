import pandas as pd

from src.data.external_sources import merge_external_sources
from src.data.validation import find_missing_dates_per_segment


def test_merge_external_sources_with_global_event():
    base = pd.DataFrame(
        {
            "store_id": [1, 1],
            "product_id": [100, 100],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "units": [10, 12],
        }
    )
    weather = pd.DataFrame(
        {
            "store_id": [1, 1],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "weather_temp_mean": [20.0, 21.0],
            "weather_precip_sum": [0.0, 0.2],
            "weather_wind_max": [12.0, 11.0],
        }
    )
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "event_name": ["Sports Final"],
            "store_id": [pd.NA],
            "event_intensity": [1.5],
        }
    )

    merged = merge_external_sources(base, weather, events)

    assert "weather_temp_mean" in merged.columns
    assert merged.loc[merged["date"] == pd.Timestamp("2024-01-02"), "event_name"].iloc[0] == "Sports Final"


def test_find_missing_dates_per_segment():
    df = pd.DataFrame(
        {
            "store_id": [1, 1],
            "product_id": [100, 100],
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        }
    )
    missing = find_missing_dates_per_segment(df)
    assert len(missing) == 1
    assert missing.iloc[0]["date"] == pd.Timestamp("2024-01-02")
