"""In-process loopback transport (concept §12.2).

The default voice transport: carries audio with configurable one-way latency,
jitter and packet loss but no external dependencies. This is what CI uses, and
what :class:`~phonebot_qa.adapters.transport.sip.SIPTransport` and
:class:`~phonebot_qa.adapters.transport.webrtc.WebRTCTransport` build on to model
their respective network characteristics.
"""

from __future__ import annotations

import random
from array import array

from ...audio.buffer import AudioBuffer
from ...audio.chaos import PACKET_MS
from .base import TransportStats, VoiceTransport


class LoopbackTransport(VoiceTransport):
    """Deterministic in-memory audio path."""

    name = "loopback"

    def __init__(
        self,
        *,
        latency_ms: int = 0,
        jitter_ms: int = 0,
        packet_loss: float = 0.0,
        codec: str = "pcm16",
        seed: int = 0,
    ) -> None:
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.packet_loss = packet_loss
        self.codec = codec
        self.seed = seed
        self._rng = random.Random(seed)
        self._stats = TransportStats(codec=codec)
        self._connected = False

    async def connect(self, *, session_id: str) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def _apply_codec(self, buffer: AudioBuffer) -> AudioBuffer:
        """Band-limit to emulate a narrowband telephony codec."""
        if self.codec in ("pcm16", "opus"):
            return buffer  # wideband: pass through
        # g711/g729: one-pole low-pass ≈ 3.4 kHz + mild quantisation.
        out = array("h")
        prev = 0.0
        alpha = 0.55
        for s in buffer.samples:
            prev = prev + alpha * (s - prev)
            out.append(int(prev) // 64 * 64)
        return AudioBuffer(
            samples=out, sample_rate=buffer.sample_rate, metadata=dict(buffer.metadata)
        )

    async def send(self, audio: AudioBuffer) -> AudioBuffer:
        out = self._apply_codec(audio.copy())

        lost_ms = 0
        if self.packet_loss > 0 and out.samples:
            frame = max(1, int(out.sample_rate * PACKET_MS / 1000))
            for start in range(0, len(out.samples), frame):
                if self._rng.random() < self.packet_loss:
                    end = min(len(out.samples), start + frame)
                    for i in range(start, end):
                        out.samples[i] = 0
                    lost_ms += int(1000 * (end - start) / out.sample_rate)

        jitter = self._rng.randint(0, self.jitter_ms) if self.jitter_ms else 0
        self._stats = TransportStats(
            one_way_latency_ms=self.latency_ms + jitter,
            jitter_ms=jitter,
            packet_loss=self.packet_loss,
            lost_ms=lost_ms,
            codec=self.codec,
        )

        # Fold transport loss into the degradation record so STT accounts for it.
        degradation = dict(out.metadata.get("degradation") or {})
        degradation["packet_loss"] = max(
            float(degradation.get("packet_loss") or 0.0), self.packet_loss
        )
        degradation["lost_ms"] = int(degradation.get("lost_ms") or 0) + lost_ms
        degradation["transport"] = self.name
        degradation["codec"] = self.codec
        out.metadata = {**out.metadata, "degradation": degradation}
        return out

    def stats(self) -> TransportStats:
        return self._stats
