"""Speech-to-text (concept sections 12.2, 14 & 34).

:class:`STTEngine` is the interface a real recognizer (Whisper, Deepgram, Azure,
…) implements. :class:`DeterministicSTT` is the built-in simulator.

**How the simulator stays honest.** The utterance travels with the audio in
``metadata['transcript']``, but the simulator never returns it verbatim. It
computes an error probability from the *measured degradation* the signal
actually suffered — SNR, packet loss, speaking rate, gain, clipping — and then
corrupts the words accordingly, seeded so the result is reproducible. So:

* on the ``clean`` profile it transcribes accurately and the task completes;
* on ``restaurant`` (SNR 8 dB, babble) words break up, the bot mishears, and the
  business assertions fail — which is exactly the finding a voice test should
  produce;
* WER is a real edit-distance measurement against the reference, so
  ``voice.stt_wer`` is meaningful and comparable across bot versions (§29).

Swapping in a real recognizer changes nothing else in the platform.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .buffer import AudioBuffer

#: Error probability floor even on pristine audio (real engines are not perfect).
BASE_ERROR_RATE = 0.005
#: SNR (dB) at or above which noise contributes no additional errors.
SNR_CLEAN_DB = 26.0
#: SNR (dB) at which recognition is essentially hopeless.
SNR_FLOOR_DB = 0.0


@dataclass
class STTResult:
    """A transcription plus the quality signals the evaluator needs."""

    text: str
    confidence: float = 1.0
    reference: str | None = None
    wer: float | None = None
    error_probability: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class STTEngine(ABC):
    """Turns audio into text."""

    #: Nominal recognition latency in ms, part of the voice latency budget (§18).
    latency_ms: int = 320

    @abstractmethod
    def transcribe(self, audio: AudioBuffer) -> STTResult:
        """Recognize ``audio``."""


def _get(mapping: dict, key: str, default: float) -> float:
    """Read a numeric degradation field, treating only a missing key as absent."""
    value = mapping.get(key)
    return default if value is None else float(value)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER: word-level Levenshtein distance / reference length."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Iterative Levenshtein over words.
    previous = list(range(len(hyp) + 1))
    for i, r_word in enumerate(ref, start=1):
        current = [i]
        for j, h_word in enumerate(hyp, start=1):
            cost = 0 if r_word == h_word else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / len(ref)


def _corrupt_word(word: str, rng: random.Random) -> str | None:
    """Apply one plausible ASR error to a word (``None`` == deletion)."""
    choice = rng.random()
    if choice < 0.25 or len(word) <= 2:
        return None  # deletion
    chars = list(word)
    if choice < 0.55:  # substitution of one character
        idx = rng.randrange(len(chars))
        chars[idx] = rng.choice("aeioustrnlm0123456789")
    elif choice < 0.8:  # transposition
        idx = rng.randrange(len(chars) - 1)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    else:  # truncation
        chars = chars[: max(1, len(chars) - rng.randint(1, 2))]
    return "".join(chars)


class DeterministicSTT(STTEngine):
    """Reproducible recognizer whose accuracy tracks real audio degradation."""

    def __init__(self, *, latency_ms: int = 320, seed: int = 0, sensitivity: float = 1.0) -> None:
        self.latency_ms = latency_ms
        self.seed = seed
        #: Scales how hard degradation hits accuracy (1.0 == calibrated default).
        self.sensitivity = sensitivity

    # -- degradation -> error probability ---------------------------------- #

    def error_probability(self, audio: AudioBuffer) -> float:
        deg = audio.metadata.get("degradation") or {}
        p = BASE_ERROR_RATE

        snr = deg.get("applied_snr_db")
        if snr is not None:
            span = SNR_CLEAN_DB - SNR_FLOOR_DB
            severity = max(0.0, min(1.0, (SNR_CLEAN_DB - float(snr)) / span))
            # Quadratic: mild noise is survivable, heavy noise is not.
            p += 0.85 * severity**2

        # NOTE: explicit ``is None`` checks, never ``or`` — 0.0 is falsy, and
        # volume=0 / speed=0 (digital silence) is the *most* degraded input
        # there is. Reading it back as the neutral 1.0 would score silence as
        # perfectly intelligible.
        loss = _get(deg, "packet_loss", 0.0)
        p += min(0.6, loss * 4.0)

        speed = _get(deg, "speed", 1.0)
        if speed <= 0.0:
            return 1.0  # no audio survives a zero playback rate
        p += min(0.25, abs(speed - 1.0) * 0.6)

        volume = _get(deg, "volume", 1.0)
        if volume <= 0.0:
            return 1.0  # muted: nothing to recognize
        if volume < 0.6:
            p += min(0.3, (0.6 - volume) * 0.7)

        p += min(0.2, _get(deg, "clipped_fraction", 0.0) * 2.0)

        # Babble is the hardest noise for a recognizer to reject.
        if deg.get("noise") == "babble" and snr is not None:
            p *= 1.15

        return max(0.0, min(0.95, p * self.sensitivity))

    # -- transcription ------------------------------------------------------ #

    def transcribe(self, audio: AudioBuffer) -> STTResult:
        reference = str(audio.metadata.get("transcript", ""))
        p_error = self.error_probability(audio)

        digest = hashlib.sha256(
            f"{self.seed}:{reference}:{p_error:.6f}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        words = reference.split()
        hypothesis: list[str] = []
        for word in words:
            if rng.random() < p_error:
                corrupted = _corrupt_word(word, rng)
                if corrupted is not None:
                    hypothesis.append(corrupted)
            else:
                hypothesis.append(word)

        text = " ".join(hypothesis)
        wer = word_error_rate(reference, text) if reference else None
        return STTResult(
            text=text,
            confidence=round(max(0.0, 1.0 - p_error), 4),
            reference=reference,
            wer=None if wer is None else round(wer, 4),
            error_probability=round(p_error, 4),
            metadata={"degradation": audio.metadata.get("degradation", {})},
        )
