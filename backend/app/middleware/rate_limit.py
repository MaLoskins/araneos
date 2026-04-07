import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, app, default_rpm: int = 100, train_rpm: int = 5):
        super().__init__(app)
        self.default_rpm = default_rpm
        self.train_rpm = train_rpm
        self._buckets = defaultdict(lambda: {"tokens": default_rpm, "last": time.time()})

    def _get_limit(self, path: str) -> int:
        if "train-gnn" in path:
            return self.train_rpm
        return self.default_rpm

    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        limit = self._get_limit(path)
        key = f"{client_ip}:{path}"

        bucket = self._buckets[key]
        now = time.time()
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(limit, bucket["tokens"] + elapsed * (limit / 60.0))
        bucket["last"] = now

        if bucket["tokens"] < 1:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
            )

        bucket["tokens"] -= 1
        return await call_next(request)
