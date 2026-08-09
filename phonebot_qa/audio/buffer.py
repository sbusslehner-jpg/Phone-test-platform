"""16-bit PCM audio buffer (concept sections 13 & 34).

A minimal, dependency-free audio container. Telephony-realistic defaults: 8 kHz
mono is the classic narrowband phone rate, but 16 kHz is used by most modern
voice stacks, so that is the default here.

The buffer carries a ``metadata`` dict alongside the samples. The simulated
TTS/STT pair uses it to pass the spoken text through the pipeline as a
side-channel, which is what makes a *deterministic* voice loop possible without
shipping an acoustic model — see :mod:`phonebot_qa.audio.stt` for exactly how
degradation (not the side channel) drives the transcription result.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Full-scale value for signed 16-bit PCM.
INT16_MAX = 32767
INT16_MIN = -32768


@dataclass
class AudioBuffer:
    """Mono signed-16-bit PCM samples plus a sample rate and metadata."""

    samples: array = field(default_factory=lambda: array("h"))
    sample_rate: int = 16000
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def silence(cls, duration_ms: int, sample_rate: int = 16000) -> "AudioBuffer":
        n = max(0, int(sample_rate * duration_ms / 1000))
        return cls(samples=array("h", [0] * n), sample_rate=sample_rate)

    @classmethod
    def from_iterable(
        cls, values: Iterable[float], sample_rate: int = 16000, **metadata: Any
    ) -> "AudioBuffer":
        return cls(
            samples=array("h", (clamp16(v) for v in values)),
            sample_rate=sample_rate,
            metadata=dict(metadata),
        )

    @classmethod
    def tone(
        cls,
        duration_ms: int,
        frequency: float = 220.0,
        amplitude: float = 0.3,
        sample_rate: int = 16000,
    ) -> "AudioBuffer":
        n = max(0, int(sample_rate * duration_ms / 1000))
        peak = amplitude * INT16_MAX
        step = 2.0 * math.pi * frequency / sample_rate
        return cls(
            samples=array("h", (clamp16(peak * math.sin(step * i)) for i in range(n))),
            sample_rate=sample_rate,
        )

    # -- properties -------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration_ms(self) -> int:
        if not self.sample_rate:
            return 0
        return int(round(1000 * len(self.samples) / self.sample_rate))

    @property
    def rms(self) -> float:
        """Root-mean-square amplitude in full-scale units (0..1)."""
        if not self.samples:
            return 0.0
        total = 0.0
        for s in self.samples:
            total += float(s) * float(s)
        return math.sqrt(total / len(self.samples)) / INT16_MAX

    @property
    def peak(self) -> float:
        if not self.samples:
            return 0.0
        return max(abs(int(s)) for s in self.samples) / INT16_MAX

    def clipped_fraction(self) -> float:
        """Share of samples sitting at full scale (a distortion indicator)."""
        if not self.samples:
            return 0.0
        hits = sum(1 for s in self.samples if s >= INT16_MAX or s <= INT16_MIN)
        return hits / len(self.samples)

    # -- operations -------------------------------------------------------- #

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(
            samples=array("h", self.samples),
            sample_rate=self.sample_rate,
            metadata=dict(self.metadata),
        )

    def slice_ms(self, start_ms: int, end_ms: int | None = None) -> "AudioBuffer":
        start = max(0, int(self.sample_rate * start_ms / 1000))
        end = len(self.samples) if end_ms is None else int(self.sample_rate * end_ms / 1000)
        end = max(start, min(len(self.samples), end))
        return AudioBuffer(
            samples=array("h", self.samples[start:end]),
            sample_rate=self.sample_rate,
            metadata=dict(self.metadata),
        )

    def concat(self, other: "AudioBuffer") -> "AudioBuffer":
        if other.sample_rate != self.sample_rate:
            raise ValueError("cannot concatenate buffers with different sample rates")
        merged = array("h", self.samples)
        merged.extend(other.samples)
        return AudioBuffer(
            samples=merged,
            sample_rate=self.sample_rate,
            metadata={**self.metadata, **other.metadata},
        )

    def to_bytes(self) -> bytes:
        return self.samples.tobytes()

    @classmethod
    def from_bytes(
        cls, raw: bytes, sample_rate: int = 16000, **metadata: Any
    ) -> "AudioBuffer":
        samples = array("h")
        samples.frombytes(raw)
        return cls(samples=samples, sample_rate=sample_rate, metadata=dict(metadata))

    def to_wav_bytes(self) -> bytes:
        """A complete RIFF/WAVE file — handy for saving failing calls as artifacts."""
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.to_bytes())
        return buf.getvalue()


def clamp16(value: float) -> int:
    """Clamp a float sample into the signed 16-bit range."""
    v = int(value)
    if v > INT16_MAX:
        return INT16_MAX
    if v < INT16_MIN:
        return INT16_MIN
    return v
