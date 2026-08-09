"""Text-to-speech (concept sections 12.2 & 34).

:class:`TTSEngine` is the interface a real engine (Azure, ElevenLabs, Piper, …)
implements. :class:`DeterministicTTS` is the built-in simulator: it synthesizes a
speech-like waveform whose *duration, amplitude and spectral envelope* are
derived from the text and the requested speaking rate, so downstream components
— the chaos layer, the transport, barge-in timing and the latency budget — all
see realistic audio without any model or network call.

The spoken text also travels in ``AudioBuffer.metadata['transcript']``. The STT
simulator does **not** simply read it back: it recovers it only as well as the
degraded signal allows (see :mod:`phonebot_qa.audio.stt`), which is what makes
noise profiles produce realistic word-error rates.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from array import array

from .buffer import AudioBuffer, clamp16

#: Average speaking rate used to size synthesized utterances.
WORDS_PER_MINUTE = 150.0
#: Per-utterance overhead (breath / leading silence) in milliseconds.
UTTERANCE_PADDING_MS = 120


class TTSEngine(ABC):
    """Turns text into audio."""

    #: Nominal synthesis latency in ms, added to the voice latency budget (§18).
    latency_ms: int = 250

    @abstractmethod
    def synthesize(
        self, text: str, *, speed: float = 1.0, sample_rate: int = 16000
    ) -> AudioBuffer:
        """Render ``text`` as speech audio."""

    def estimate_duration_ms(self, text: str, *, speed: float = 1.0) -> int:
        words = max(1, len(text.split()))
        base = words / WORDS_PER_MINUTE * 60_000.0
        return int(base / max(0.1, speed)) + UTTERANCE_PADDING_MS


class DeterministicTTS(TTSEngine):
    """A dependency-free, reproducible speech simulator.

    Builds a sum of formant-like sinusoids whose pitch and envelope are seeded
    from the text, then applies a syllable-rate amplitude envelope so the signal
    has speech-like on/off structure (which barge-in and endpointing logic need).
    """

    def __init__(self, *, voice: str = "user", latency_ms: int = 250) -> None:
        self.voice = voice
        self.latency_ms = latency_ms

    def _pitch(self, text: str) -> float:
        digest = hashlib.sha256(f"{self.voice}:{text}".encode("utf-8")).digest()
        # Map into a plausible fundamental-frequency range (85..255 Hz).
        return 85.0 + (digest[0] / 255.0) * 170.0

    def synthesize(
        self, text: str, *, speed: float = 1.0, sample_rate: int = 16000
    ) -> AudioBuffer:
        text = text or ""
        duration_ms = self.estimate_duration_ms(text, speed=speed)
        n = max(1, int(sample_rate * duration_ms / 1000))
        f0 = self._pitch(text)
        # Syllable rate ~4 Hz scaled by speaking speed drives the envelope.
        syllable_hz = 4.0 * max(0.1, speed)
        samples = array("h")
        two_pi = 2.0 * math.pi
        for i in range(n):
            t = i / sample_rate
            # Envelope: raised cosine at syllable rate, plus utterance fade-in/out.
            env = 0.5 * (1.0 - math.cos(two_pi * syllable_hz * t))
            fade = min(1.0, i / max(1, sample_rate * 0.02), (n - i) / max(1, sample_rate * 0.02))
            # Fundamental + two formant-ish harmonics.
            value = (
                0.55 * math.sin(two_pi * f0 * t)
                + 0.30 * math.sin(two_pi * f0 * 2.4 * t)
                + 0.15 * math.sin(two_pi * f0 * 3.7 * t)
            )
            samples.append(clamp16(value * env * fade * 0.42 * 32767))
        return AudioBuffer(
            samples=samples,
            sample_rate=sample_rate,
            metadata={
                "transcript": text,
                "voice": self.voice,
                "speed": speed,
                "synthesized_ms": duration_ms,
            },
        )
