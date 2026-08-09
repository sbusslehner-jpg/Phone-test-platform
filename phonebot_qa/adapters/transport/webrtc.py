"""WebRTC transport (concept sections 11 & 34).

Models a browser-style WebRTC leg: Opus wideband audio, lower latency than the
PSTN but more variable jitter, with the ICE/DTLS setup events a WebRTC test wants
to see on the trace. A real ``aiortc`` peer connection can be supplied as
``stack`` without changing anything else.
"""

from __future__ import annotations

from ...audio.buffer import AudioBuffer
from ...observability import EventLog
from .base import TransportStats
from .loopback import LoopbackTransport


class WebRTCTransport(LoopbackTransport):
    """WebRTC peer connection (simulated by default, real stack optional)."""

    name = "webrtc"

    def __init__(
        self,
        *,
        peer: str = "phonebot-web",
        codec: str = "opus",
        latency_ms: int = 60,
        jitter_ms: int = 45,
        packet_loss: float = 0.01,
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
        self.peer = peer
        self.events = events
        self.stack = stack

    async def connect(self, *, session_id: str) -> None:
        if self.events is not None:
            self.events.emit("webrtc_offer", peer=self.peer, codec=self.codec)
            self.events.emit("webrtc_ice_connected", peer=self.peer)
        if self.stack is not None:  # pragma: no cover - needs a real peer
            await self.stack.connect(session_id=session_id, peer=self.peer)
        await super().connect(session_id=session_id)

    async def disconnect(self) -> None:
        if self.stack is not None:  # pragma: no cover - needs a real peer
            await self.stack.disconnect()
        await super().disconnect()
        if self.events is not None:
            self.events.emit("webrtc_closed", peer=self.peer)

    async def send(self, audio: AudioBuffer) -> AudioBuffer:
        if self.stack is not None:  # pragma: no cover - needs a real peer
            return await self.stack.send(audio)
        return await super().send(audio)

    def stats(self) -> TransportStats:
        stats = super().stats()
        stats.extra.setdefault("peer", self.peer)
        return stats
