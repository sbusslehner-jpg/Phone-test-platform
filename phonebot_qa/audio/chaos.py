"""Audio chaos layer (concept section 13).

Sits between the caller's TTS and the phonebot and degrades the signal the way a
real phone line does::

    {"snr": 10, "noise": "street", "speed": 1.08, "volume": 0.8, "packet_loss": 0.02}

The concept names FFmpeg/SoX as the implementation; those are perfectly usable
behind this interface, but the built-in implementation is pure Python so noise
testing runs in CI with no binaries and — crucially — is **deterministic given
the seed**, so a call that failed at ``snr=10, noise=street, seed=7`` replays
exactly (section 24).

Every transformation also records what it did in
``AudioBuffer.metadata['degradation']``, which the STT simulator consumes to
produce a realistic word-error rate.
"""

from __future__ import annotations

import math
import random
from array import array
from dataclasses import dataclass, field
from typing import Any

from .buffer import AudioBuffer, clamp16

#: Packets are 20 ms — the standard RTP frame for telephony codecs.
PACKET_MS = 20


@dataclass
class ChaosConfig:
    """One audio degradation profile (concept §13)."""

    name: str = "clean"
    #: Signal-to-noise ratio in dB. ``None`` means no added noise.
    snr_db: float | None = None
    #: Named noise colour: white, pink, brown, babble, hum.
    noise: str = "white"
    #: Playback rate multiplier (1.08 == 8% faster speaker).
    speed: float = 1.0
    #: Linear gain applied to the signal.
    volume: float = 1.0
    #: Fraction of 20 ms packets dropped (0..1).
    packet_loss: float = 0.0
    #: Extra one-way latency contributed by the network, in ms.
    latency_ms: int = 0
    #: Jitter (ms) — variability added to per-packet arrival.
    jitter_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "snr_db": self.snr_db,
            "noise": self.noise,
            "speed": self.speed,
            "volume": self.volume,
            "packet_loss": self.packet_loss,
            "latency_ms": self.latency_ms,
            "jitter_ms": self.jitter_ms,
        }


def _noise_sample(kind: str, rng: random.Random, state: dict[str, float]) -> float:
    """One unit-variance-ish noise sample of the requested colour."""
    white = rng.uniform(-1.0, 1.0)
    if kind == "white":
        return white
    if kind == "pink":
        # One-pole low-passed white noise approximates pink noise.
        state["p"] = 0.98 * state.get("p", 0.0) + 0.02 * white
        return state["p"] * 6.0
    if kind == "brown":
        state["b"] = max(-1.0, min(1.0, state.get("b", 0.0) + 0.02 * white))
        return state["b"] * 3.0
    if kind == "babble":
        # Several detuned tones + noise ≈ background conversation.
        state["t"] = state.get("t", 0.0) + 1.0
        t = state["t"]
        return 0.6 * (
            math.sin(t * 0.011) + math.sin(t * 0.017) + math.sin(t * 0.023)
        ) / 3.0 + 0.4 * white
    if kind == "hum":
        state["t"] = state.get("t", 0.0) + 1.0
        return math.sin(state["t"] * 0.0125) * 0.9 + 0.1 * white
    return white


def _resample(buffer: AudioBuffer, speed: float) -> AudioBuffer:
    """Change playback rate (linear interpolation), preserving sample rate."""
    if abs(speed - 1.0) < 1e-6 or not buffer.samples:
        return buffer
    src = buffer.samples
    out_len = max(1, int(len(src) / max(0.1, speed)))
    out = array("h")
    for i in range(out_len):
        pos = i * speed
        left = int(pos)
        if left >= len(src) - 1:
            out.append(src[-1])
            continue
        frac = pos - left
        out.append(clamp16(src[left] * (1.0 - frac) + src[left + 1] * frac))
    return AudioBuffer(
        samples=out, sample_rate=buffer.sample_rate, metadata=dict(buffer.metadata)
    )


class AudioChaos:
    """Applies a :class:`ChaosConfig` to audio, deterministically per seed."""

    def __init__(self, config: ChaosConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or ChaosConfig()
        self.seed = seed

    def apply(self, buffer: AudioBuffer) -> AudioBuffer:
        cfg = self.config
        rng = random.Random((self.seed, cfg.name, len(buffer)).__hash__())
        out = buffer.copy()

        # 1. Speaking rate.
        out = _resample(out, cfg.speed)

        # 2. Gain.
        if abs(cfg.volume - 1.0) > 1e-6:
            out.samples = array("h", (clamp16(s * cfg.volume) for s in out.samples))

        # 3. Additive noise at the requested SNR.
        applied_snr = None
        if cfg.snr_db is not None and out.samples:
            signal_rms = max(out.rms, 1e-6)
            noise_rms = signal_rms / (10 ** (cfg.snr_db / 20.0))
            state: dict[str, float] = {}
            scale = noise_rms * 32767.0
            noisy = array("h")
            for s in out.samples:
                noisy.append(clamp16(s + _noise_sample(cfg.noise, rng, state) * scale))
            out.samples = noisy
            applied_snr = cfg.snr_db

        # 4. Packet loss — zero whole 20 ms frames, as a jitter buffer would.
        lost_ms = 0
        if cfg.packet_loss > 0 and out.samples:
            frame = max(1, int(out.sample_rate * PACKET_MS / 1000))
            for start in range(0, len(out.samples), frame):
                if rng.random() < cfg.packet_loss:
                    end = min(len(out.samples), start + frame)
                    for i in range(start, end):
                        out.samples[i] = 0
                    lost_ms += int(1000 * (end - start) / out.sample_rate)

        degradation = {
            **cfg.as_dict(),
            "applied_snr_db": applied_snr,
            "lost_ms": lost_ms,
            "clipped_fraction": round(out.clipped_fraction(), 4),
        }
        out.metadata = {**out.metadata, "degradation": degradation}
        return out


def apply_chaos(
    buffer: AudioBuffer, config: ChaosConfig | None = None, *, seed: int = 0
) -> AudioBuffer:
    """Convenience wrapper around :class:`AudioChaos`."""
    return AudioChaos(config, seed=seed).apply(buffer)
