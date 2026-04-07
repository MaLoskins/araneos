import logging
import threading
import uuid
from typing import Dict

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    def __init__(self, max_sessions: int = 50):
        self._sessions: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def store(self, data: dict) -> str:
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sessions[session_id] = data
            if len(self._sessions) > self._max_sessions:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
        logger.info(f"Stored session {session_id} ({len(data.get('nodes', []))} nodes)")
        return session_id

    def get(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' not found.")
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
