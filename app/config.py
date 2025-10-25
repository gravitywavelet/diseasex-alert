from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    """Global configuration for API runtime."""
    MODEL_PATH: Path = Field(default=Path("artifacts/model_pipeline.joblib"))
    STATS_PATH: Path = Field(default=Path("artifacts/train_stats.json"))
    DEFAULT_ALERT_COVERAGE: float = Field(default=0.60)  # bottom 60% by P_TREATED
    class Config:
        env_file = ".env"

settings = Settings()