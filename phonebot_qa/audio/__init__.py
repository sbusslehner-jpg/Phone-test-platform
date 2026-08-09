"""Audio layer for voice end-to-end testing (concept sections 12.2, 13, 34).

Phase 3 runs the *same text scenarios* as real phone calls::

    LLM User -> TTS -> Audio Chaos -> SIP/WebRTC -> Phonebot
             -> STT -> Bot Logic -> Bot TTS -> Voice Metrics

Everything here is dependency-free and deterministic: :class:`AudioBuffer` is
plain 16-bit PCM built on :mod:`array`, the chaos layer implements SNR, noise,
speed, volume and packet loss arithmetically, and the TTS/STT engines are
*simulators* behind the same interfaces a real engine implements.

That design is deliberate (concept §36): the platform's value is the harness and
the assertions, so the harness must run in CI with no models, no API keys and no
audio hardware — while a production deployment swaps in a real TTS/STT/telephony
stack behind :class:`TTSEngine`, :class:`STTEngine` and the transport interface
without touching a single scenario.
"""

from __future__ import annotations

from .bargein import BARGE_IN_SLA_MS, BargeInConfig, BargeInController, BargeInResult
from .buffer import AudioBuffer
from .chaos import AudioChaos, ChaosConfig, apply_chaos
from .profiles import NIGHTLY_SWEEP, NOISE_PROFILES, PROFILES, get_profile
from .stt import DeterministicSTT, STTEngine, STTResult, word_error_rate
from .tts import DeterministicTTS, TTSEngine

__all__ = [
    "AudioBuffer",
    "AudioChaos",
    "BARGE_IN_SLA_MS",
    "BargeInConfig",
    "BargeInController",
    "BargeInResult",
    "ChaosConfig",
    "DeterministicSTT",
    "DeterministicTTS",
    "NIGHTLY_SWEEP",
    "NOISE_PROFILES",
    "PROFILES",
    "STTEngine",
    "STTResult",
    "TTSEngine",
    "apply_chaos",
    "get_profile",
    "word_error_rate",
]
