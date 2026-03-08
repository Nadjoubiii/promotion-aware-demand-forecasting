"""Central configuration for local development."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/forecasting")
    forecast_horizon_days: int = int(os.getenv("FORECAST_HORIZON_DAYS", "7"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))


settings = Settings()
