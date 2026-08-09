"""SIP transport (concept sections 11 & 34).

Models a real SIP/RTP call leg: G.711 narrowband audio, PSTN-typical one-way
latency and jitter, and RTP packet loss. Also emits the SIP dialog events a
telephony test needs to assert on (``sip_invite`` … ``sip_bye``).

By default this runs the simulated path so voice suites work in CI. Pointing it
at a real PBX is a matter of supplying a ``stack`` object (e.g. a ``pjsua2``
wrapper) exposing ``connect``/``send``/``disconnect``; the rest of the platform
is unchanged.
"""

from __future__ import annotations

from ...audio.buffer import AudioBuffer
from ...observability import EventLog
from .base import TransportStats
from .loopback import LoopbackTransport


class SIPTransport(LoopbackTransport):
    """SIP/RTP call leg (simulated by default, real stack optional)."""

    name = "sip"

    def __init__(
        self,
        *,
        uri: str = "sip:phonebot@localhost",
        codec: str = "g711",
        latency_ms: int = 120,
        jitter_ms: int = 30,
        packet_loss: float = 0.005,
        seed: int = 0,
        events: EventLog | None = None,
        stack: object | None = None,
    ) -> None:
        super().__init__(
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            packet_loss=packet_loss,
            codec=codec,
            seed=seed,
        )
        self.uri = uri
        self.events = events
        #: Optional real SIP stack (pjsua2 etc.). When set, it takes over I/O.
        self.stack = stack

    async def connect(self, *, session_id: str) -> None:
        if self.events is not None:
            self.events.emit("sip_invite", uri=self.uri, codec=self.codec)
        if self.stack is not None:  # pragma: no cover - needs a real PBX
            await self.stack.connect(session_id=session_id, uri=self.uri)
        await super().connect(session_id=session_id)
        if self.events is not None:
            self.events.emit("sip_answered", uri=self.uri)

    async def disconnect(self) -> None:
        if self.stack is not None:  # pragma: no cover - needs a real PBX
            await self.stack.disconnect()
        await super().disconnect()
        if self.events is not None:
            self.events.emit("sip_bye", uri=self.uri)

    async def send(self, audio: AudioBuffer) -> AudioBuffer:
        if self.stack is not None:  # pragma: no cover - needs a real PBX
            return await self.stack.send(audio)
        return await super().send(audio)

    def stats(self) -> TransportStats:
        stats = super().stats()
        stats.extra.setdefault("uri", self.uri)
        return stats
