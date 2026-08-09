"""A REST bot adapter (concept section 11).

Talks to an external phonebot that exposes a small HTTP test contract:

* ``POST {base_url}/session/start``   -> ``{"session_id": ..., "greeting": ...}``
* ``POST {base_url}/session/message`` -> ``{"text": ..., "done": bool}``
* ``POST {base_url}/session/stop``    -> 2xx

This keeps the platform independent of any specific phonebot: the same test
suite runs against the in-process reference bot or a remote one just by swapping
the adapter. ``httpx`` is an optional dependency, imported lazily so the core
platform has no hard HTTP requirement.
"""

from __future__ import annotations

from .base import BotAdapter, BotResponse, BotSession, SessionContext


class RESTBotAdapter(BotAdapter):
    """Drive an external phonebot over a minimal JSON/HTTP contract."""

    def __init__(
        self,
        base_url: str,
        *,
        version: str = "rest-unknown",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout = timeout
        self.headers = headers or {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "RESTBotAdapter requires httpx: pip install 'phonebot-qa[api]'"
                ) from exc
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, headers=self.headers
            )
        return self._client

    async def start_session(self, context: SessionContext) -> BotSession:
        client = self._get_client()
        resp = await client.post(
            "/session/start",
            json={
                "scenario_id": context.scenario_id,
                "metadata": context.metadata,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return BotSession(
            session_id=str(data["session_id"]),
            greeting=data.get("greeting"),
        )

    async def send_text(self, session: BotSession, message: str) -> BotResponse:
        client = self._get_client()
        resp = await client.post(
            "/session/message",
            json={"session_id": session.session_id, "message": message},
        )
        resp.raise_for_status()
        data = resp.json()
        return BotResponse(
            text=str(data.get("text", "")),
            done=bool(data.get("done", False)),
            metadata=data.get("metadata", {}) or {},
        )

    async def stop_session(self, session: BotSession) -> None:
        client = self._get_client()
        try:
            await client.post(
                "/session/stop", json={"session_id": session.session_id}
            )
        finally:
            await client.aclose()
            self._client = None
