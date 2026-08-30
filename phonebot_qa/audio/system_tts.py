"""A caller who actually speaks — the system's own text-to-speech.

Why this exists next to :class:`DeterministicTTS`: that engine builds a sum of
formant-like sinusoids with a syllable-rate envelope. That is exactly right for
what it was made for — barge-in timing and endpointing need speech-*shaped*
audio, not speech. It is useless the moment a real STT is on the other end.

Measured against CROSS3 on 2026-08-30: a 66-second call with three caller
utterances produced **one** model response — the greeting. Azure transcribed no
word of the sinusoids, its turn detection never saw an utterance end, and the
agent therefore never answered. Every conversational voice scenario was
untestable, and the failure looked like a broken bot.

With this engine (and paced sending, see ``Cross3VoicePhoneClient.send_audio``)
the same call produced four responses and completed a real hand-off to a human.

Deliberately the *system's* voice rather than a cloud service: no key, no
network latency in the measurement, no per-run cost, and the audio is cached so
a suite re-run is free. The trade-off is honest — a synthetic voice is easier to
transcribe than a caller in a moving car. For robustness against noise, wrap the
result with :func:`phonebot_qa.audio.apply_chaos` as the scenarios already do.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from array import array

from .buffer import AudioBuffer
from .tts import TTSEngine

_CACHE = os.path.join(tempfile.gettempdir(), "phonebot-qa-tts")


class SystemTTSUnavailable(RuntimeError):
    """No usable system voice — the caller would be silent."""


class SystemTTS(TTSEngine):
    """Real speech from the operating system (macOS ``say``, Linux ``espeak-ng``).

    :param voice: platform voice name. Defaults to a German voice on both
        platforms, because CROSS3 is a German-language agent.
    :param rate: words per minute.
    """

    #: Locally generated: no network latency belongs in the voice budget (§18).
    latency_ms = 0

    def __init__(self, *, voice: str | None = None, rate: int = 180) -> None:
        self.rate = rate
        if platform.system() == "Darwin" and shutil.which("say"):
            self.backend = "say"
            self.voice = voice or "Anna"
        elif shutil.which("espeak-ng"):
            self.backend = "espeak-ng"
            self.voice = voice or "de"
        else:
            raise SystemTTSUnavailable(
                "Neither macOS `say` nor `espeak-ng` found. Install espeak-ng "
                "(apt-get install -y espeak-ng) or pass another TTSEngine — "
                "DeterministicTTS cannot be transcribed by a real STT."
            )
        os.makedirs(_CACHE, exist_ok=True)

    def synthesize(
        self, text: str, *, speed: float = 1.0, sample_rate: int = 16000
    ) -> AudioBuffer:
        wpm = max(80, int(self.rate * max(0.5, speed)))
        key = hashlib.sha256(
            f"{self.backend}|{self.voice}|{wpm}|{sample_rate}|{text}".encode("utf-8")
        ).hexdigest()[:24]
        raw_path = os.path.join(_CACHE, f"{key}.raw")
        if not os.path.exists(raw_path):
            self._render(text, wpm, sample_rate, raw_path)
        with open(raw_path, "rb") as fh:
            data = fh.read()
        samples = array("h")
        samples.frombytes(data[: len(data) // 2 * 2])
        return AudioBuffer(samples=samples, sample_rate=sample_rate)

    # -- rendering --------------------------------------------------------- #

    def _render(self, text: str, wpm: int, sample_rate: int, out_path: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav = tmp.name
        try:
            if self.backend == "say":
                with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                    aiff = tmp.name
                try:
                    subprocess.run(
                        ["say", "-v", self.voice, "-r", str(wpm), "-o", aiff, text],
                        check=True, capture_output=True,
                    )
                    subprocess.run(
                        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}",
                         "-c", "1", aiff, wav],
                        check=True, capture_output=True,
                    )
                finally:
                    _unlink(aiff)
            else:
                subprocess.run(
                    ["espeak-ng", "-v", self.voice, "-s", str(wpm), "-w", wav, text],
                    check=True, capture_output=True,
                )
                # espeak-ng writes at its own rate; resample if it differs.
                _resample_wav_in_place(wav, sample_rate)
            with open(wav, "rb") as fh:
                blob = fh.read()
            # Find the `data` chunk instead of guessing a 44-byte header.
            i = blob.find(b"data")
            with open(out_path, "wb") as fh:
                fh.write(blob[i + 8:] if i >= 0 else blob[44:])
        finally:
            _unlink(wav)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _resample_wav_in_place(path: str, target_rate: int) -> None:
    """Cheap linear resample so espeak-ng output matches the telephony rate."""
    import audioop  # stdlib, deprecated but present through 3.12
    import wave

    with wave.open(path, "rb") as wf:
        rate, frames = wf.getframerate(), wf.readframes(wf.getnframes())
        channels, width = wf.getnchannels(), wf.getsampwidth()
    if channels > 1:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    if rate != target_rate:
        frames, _ = audioop.ratecv(frames, width, 1, rate, target_rate, None)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(target_rate)
        wf.writeframes(frames)
