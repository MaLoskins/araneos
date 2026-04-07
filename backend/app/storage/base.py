from typing import Protocol


class SessionStore(Protocol):
    def store(self, data: dict) -> str:
        """Store graph data, return session ID."""
        ...

    def get(self, session_id: str) -> dict:
        """Retrieve graph data by session ID. Raises KeyError if not found."""
        ...

    def delete(self, session_id: str) -> None:
        """Remove a session."""
        ...
