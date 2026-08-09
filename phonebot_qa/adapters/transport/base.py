"""The voice transport interface (concept sections 11 & 34)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ...audio.buffer import AudioBuffer


@dataclass
class TransportStats:
    """What the transport did to the audio — feeds the voice metrics (§18)."""

    one_way_latency_ms: int = 0
    jitter_ms: int = 0
    packet_loss: float = 0.0
    lost_ms: int = 0
    codec: str = "pcm16"
    extra: dict = field(default_factory=dict)


class VoiceTransport(ABC):
    """Carries audio frames between the caller simulator and the phonebot."""

    #: Human-readable transport name recorded on the trace.
    name: str = "transport"

    async def connect(self, *, session_id: str) -> None:
        """Establish the call leg. Default is a no-op."""
        return None

    async def disconnect(self) -> None:
        """Tear down the call leg. Default is a no-op."""
        return None

    @abstractmethod
    async def send(self, audio: AudioBuffer) -> AudioBuffer:
        """Carry ``audio`` toward the bot, returning what actually arrives."""

    @abstractmethod
    def stats(self) -> TransportStats:
        """Statistics for the most recent transmission."""
