"""External data loaders and feature merges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


def load_store_coordinates(coords_path: Path) -> pd.DataFrame:
    """Load optional store coordinate mapping for weather enrichment."""
    if not coords_path.exists():
        return pd.DataFrame(columns=["store_id", "latitude", "longitude"])

    coords = pd.read_csv(coords_path)
    required = {"store_id", "latitude", "longitude"}
    missing = required - set(coords.columns)
    if missing:
        raise ValueError(f"Missing columns in {coords_path.name}: {sorted(missing)}")
    return coords[list(required)].dropna().drop_duplicates()


def fetch_open_meteo_daily(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Fetch daily historical weather from Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,windspeed_10m_max",
        "timezone": timezone,
    }

    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json().get("daily", {})

    if not payload or "time" not in payload:
        return pd.DataFrame(columns=["date", "weather_temp_mean", "weather_precip_sum", "weather_wind_max"])

    return pd.DataFrame(
        {
            "date": pd.to_datetime(payload.get("time", [])),
            "weather_temp_mean": payload.get("temperature_2m_mean", []),
            "weather_precip_sum": payload.get("precipitation_sum", []),
            "weather_wind_max": payload.get("windspeed_10m_max", []),
        }
    )


def build_weather_features_for_stores(
    coords_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Fetch and stack weather data for all configured stores."""
    frames: list[pd.DataFrame] = []

    for row in coords_df.itertuples(index=False):
        weather = fetch_open_meteo_daily(
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            timeout_seconds=timeout_seconds,
        )
        if weather.empty:
            continue
        weather["store_id"] = row.store_id
        frames.append(weather)

    if not frames:
        return pd.DataFrame(
            columns=["store_id", "date", "weather_temp_mean", "weather_precip_sum", "weather_wind_max"]
        )

    stacked = pd.concat(frames, ignore_index=True)
    return stacked[["store_id", "date", "weather_temp_mean", "weather_precip_sum", "weather_wind_max"]]


def load_local_events(events_path: Path) -> pd.DataFrame:
    """Load optional local events table.

    Supported schema:
    - date (required)
    - event_name (required)
    - store_id (optional)
    - event_intensity (optional numeric)
    """
    if not events_path.exists():
        return pd.DataFrame(columns=["date", "event_name", "store_id", "event_intensity"])

    events = pd.read_csv(events_path, parse_dates=["date"])
    required = {"date", "event_name"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing columns in {events_path.name}: {sorted(missing)}")

    if "store_id" not in events.columns:
        events["store_id"] = pd.NA
    if "event_intensity" not in events.columns:
        events["event_intensity"] = 1.0

    return events[["date", "event_name", "store_id", "event_intensity"]].copy()


def merge_external_sources(
    base_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    local_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weather and local events into the base dataset."""
    merged = base_df.copy()

    if not weather_df.empty:
        weather_df = weather_df.copy()
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        merged = merged.merge(weather_df, on=["store_id", "date"], how="left")

    if not local_events_df.empty:
        events = local_events_df.copy()
        events["date"] = pd.to_datetime(events["date"])
        store_specific = events[events["store_id"].notna()]
        global_events = events[events["store_id"].isna()].drop(columns=["store_id"])

        if not store_specific.empty:
            store_specific["store_id"] = store_specific["store_id"].astype(merged["store_id"].dtype)
            merged = merged.merge(
                store_specific,
                on=["store_id", "date"],
                how="left",
                suffixes=("", "_store_event"),
            )

        if not global_events.empty:
            merged = merged.merge(
                global_events,
                on=["date"],
                how="left",
                suffixes=("", "_global_event"),
            )

        merged["event_name"] = merged.filter(like="event_name").bfill(axis=1).iloc[:, 0]
        merged["event_intensity"] = (
            merged.filter(like="event_intensity").bfill(axis=1).iloc[:, 0].fillna(0.0)
        )
        drop_cols = [
            col
            for col in merged.columns
            if col.endswith("_store_event") or col.endswith("_global_event")
        ]
        merged = merged.drop(columns=drop_cols)

    return merged
