"""Central configuration for local development."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/forecasting")
    forecast_horizon_days: int = int(os.getenv("FORECAST_HORIZON_DAYS", "7"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    favorita_raw_dir: str = os.getenv("FAVORITA_RAW_DIR", "data/raw/favorita")
    external_data_dir: str = os.getenv("EXTERNAL_DATA_DIR", "data/external")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    weather_timezone: str = os.getenv("WEATHER_TIMEZONE", "UTC")
    weather_request_timeout_seconds: int = int(os.getenv("WEATHER_REQUEST_TIMEOUT_SECONDS", "30"))
    favorita_competition_slug: str = os.getenv(
        "FAVORITA_COMPETITION_SLUG", "favorita-grocery-sales-forecasting"
    )


settings = Settings()
