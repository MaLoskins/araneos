import json
import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class RedisSessionStore:
    def __init__(self, redis_url: str, ttl: int = 7200):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl

    def store(self, data: dict) -> str:
        session_id = str(uuid.uuid4())[:8]
        serialized = json.dumps(data, cls=_NumpyEncoder)
        self._client.setex(f"session:{session_id}", self._ttl, serialized)
        logger.info(f"Stored session {session_id} in Redis (TTL={self._ttl}s)")
        return session_id

    def get(self, session_id: str) -> dict:
        raw = self._client.get(f"session:{session_id}")
        if raw is None:
            raise KeyError(f"Session '{session_id}' not found.")
        return json.loads(raw)

    def delete(self, session_id: str) -> None:
        self._client.delete(f"session:{session_id}")
