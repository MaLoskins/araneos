from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SESSION_BACKEND: str = "memory"
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL: int = 7200
    CORS_ORIGINS: list[str] = ["*"]
    API_KEY: str = ""
    LOG_LEVEL: str = "INFO"
    MAX_SESSIONS: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
