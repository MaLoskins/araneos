from functools import lru_cache
from fastapi import Header, HTTPException
from app.config import get_settings
from app.storage.memory import InMemorySessionStore


@lru_cache
def get_session_store():
    settings = get_settings()
    if settings.SESSION_BACKEND == "redis":
        from app.storage.redis import RedisSessionStore
        return RedisSessionStore(settings.REDIS_URL, settings.SESSION_TTL)
    return InMemorySessionStore(max_sessions=settings.MAX_SESSIONS)


async def verify_api_key(x_api_key: str = Header(default=None)):
    """Optional API key auth. Disabled when API_KEY is empty."""
    settings = get_settings()
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
