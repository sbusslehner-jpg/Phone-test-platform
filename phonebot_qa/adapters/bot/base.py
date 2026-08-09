"""The :class:`BotAdapter` interface (concept section 11).

The test platform stays independent of the phonebot it tests. Every phonebot is
reached through a ``BotAdapter``; swapping the adapter lets the same test suite
run against a REST bot, a WebSocket bot, or (phase 3) a SIP/WebRTC voice bot.

Methods are async to match the conversation runner (section 7). ``send_audio`` is
part of the interface for voice mode but defaults to ``NotImplementedError`` so
text-only adapters need not implement it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...backend.proxy import ToolProxy
from ...observability import EventLog


@dataclass
class SessionContext:
    """What an adapter needs to start a session for one test case.

    For an *in-process* bot (the reference bot) ``proxy`` and ``events`` are the
    live tool gateway and event log, so the bot's tool calls are logged and its
    domain events land on the shared trace. For an *external* bot the proxy is
    exposed as an HTTP tool gateway (section 16) the bot is pointed at, and the
    adapter typically ignores these in-process handles.
    """

    scenario_id: str
    proxy: ToolProxy | None = None
    events: EventLog | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BotResponse:
    """A single bot turn's output."""

    text: str
    # ``done`` lets the bot signal it considers the interaction complete, which
    # the user simulator may use to stop (section 8).
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BotSession:
    """An open conversation session with the bot."""

    session_id: str
    # Optional greeting the bot speaks when it "answers the phone". When present
    # the runner seeds the transcript with it before the first user turn.
    greeting: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


class BotAdapter(ABC):
    """Abstract phonebot adapter."""

    #: Free-form version label recorded on every result (section 26).
    version: str = "unknown"

    @abstractmethod
    async def start_session(self, context: SessionContext) -> BotSession:
        """Open a session and return it (optionally with a greeting)."""

    @abstractmethod
    async def send_text(self, session: BotSession, message: str) -> BotResponse:
        """Send a user utterance and get the bot's reply."""

    async def send_audio(self, session: BotSession, audio: bytes) -> BotResponse:
        """Voice-mode entry point (phase 3). Not required for text adapters."""
        raise NotImplementedError("this adapter does not support voice mode")

    async def stop_session(self, session: BotSession) -> None:
        """Tear down a session. Default is a no-op."""
        return None
