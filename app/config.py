from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = Field(default=8000, alias="PORT")
    model_cache_dir: str = Field(default="/app/models", alias="MODEL_CACHE_DIR")
    max_concurrent_requests: int = Field(default=2, alias="MAX_CONCURRENT_REQUESTS")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
