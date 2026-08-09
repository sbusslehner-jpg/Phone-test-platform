"""Voice bot adapter (concept sections 11, 12.2 & 18).

Wraps a *text* phonebot with the pipeline a real voice agent runs internally::

    incoming audio -> STT -> dialog logic -> TTS -> outgoing audio

This is what makes Phase 3 possible without rewriting a single scenario: the same
``ReferenceAppointmentBot`` (or any ``BotAdapter``) becomes a voice agent, and the
same booking/cancellation scenarios can be executed as real calls.

Every stage emits its lifecycle events with timing, so the concept's latency
decomposition (section 18) falls out of the trace::

    stt_started/stt_finished · bot_processing_started · tool_called/tool_result
    tts_started/tts_finished · bot_audio_started/bot_audio_finished

A phonebot that is *natively* voice (its own STT/TTS) simply implements
``BotAdapter.send_audio`` directly instead of using this wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...audio.buffer import AudioBuffer
from ...audio.stt import DeterministicSTT, STTEngine, STTResult
from ...audio.tts import DeterministicTTS, TTSEngine
from .base import BotAdapter, BotResponse, BotSession, SessionContext


@dataclass
class VoiceTurnRecord:
    """Per-turn voice diagnostics, consumed by the voice evaluator."""

    heard_text: str = ""
    reference_text: str = ""
    wer: float | None = None
    stt_confidence: float = 1.0
    stt_ms: int = 0
    think_ms: int = 0
    tts_ms: int = 0
    bot_audio_ms: int = 0
    metadata: dict = field(default_factory=dict)


class VoiceBotAdapter(BotAdapter):
    """Turn any text :class:`BotAdapter` into a voice agent."""

    def __init__(
        self,
        text_bot: BotAdapter,
        *,
        stt: STTEngine | None = None,
        tts: TTSEngine | None = None,
        think_ms: int = 400,
        version: str | None = None,
    ) -> None:
        self.text_bot = text_bot
        self.stt = stt or DeterministicSTT()
        self.tts = tts or DeterministicTTS(voice="bot", latency_ms=220)
        self.think_ms = think_ms
        self.version = version or f"{text_bot.version}+voice"
        #: One record per turn, in order.
        self.turn_records: list[VoiceTurnRecord] = []
        self._events = None

    async def start_session(self, context: SessionContext) -> BotSession:
        self._events = context.events
        session = await self.text_bot.start_session(context)
        session.state["voice"] = True
        return session

    async def send_text(self, session: BotSession, message: str) -> BotResponse:
        """Text passthrough (used when a voice session falls back to text)."""
        return await self.text_bot.send_text(session, message)

    async def send_audio(self, session: BotSession, audio: bytes | AudioBuffer) -> BotResponse:
        """Full voice turn: recognize, think, speak.

        Returns the bot's :class:`BotResponse` with the synthesized reply audio
        attached at ``metadata['audio']`` and the per-turn voice diagnostics at
        ``metadata['voice']``.
        """
        events = self._events
        buffer = (
            audio
            if isinstance(audio, AudioBuffer)
            else AudioBuffer.from_bytes(audio)
        )
        turn = session.state.get("turn_index")
        record = VoiceTurnRecord()

        # -- 1. Speech recognition ---------------------------------------- #
        if events is not None:
            events.emit("stt_started", turn=turn, audio_ms=buffer.duration_ms)
        result: STTResult = self.stt.transcribe(buffer)
        record.stt_ms = self.stt.latency_ms
        if events is not None:
            events.clock.advance(record.stt_ms)
            events.emit(
                "stt_finished",
                turn=turn,
                text=result.text,
                confidence=result.confidence,
                wer=result.wer,
            )
        record.heard_text = result.text
        record.reference_text = result.reference or ""
        record.wer = result.wer
        record.stt_confidence = result.confidence

        # -- 2. Dialog logic (the text bot, tool calls and all) ----------- #
        before = events.clock.now_ms if events is not None else 0
        response = await self.text_bot.send_text(session, result.text)
        if events is not None:
            # Tool latency already advanced the clock inside the proxy; add the
            # bot's own thinking time on top.
            events.clock.advance(self.think_ms)
            record.think_ms = events.clock.now_ms - before
        else:
            record.think_ms = self.think_ms

        # -- 3. Speech synthesis ------------------------------------------ #
        if events is not None:
            events.emit("tts_started", turn=turn)
        reply_audio = self.tts.synthesize(response.text)
        record.tts_ms = self.tts.latency_ms
        record.bot_audio_ms = reply_audio.duration_ms
        if events is not None:
            events.clock.advance(record.tts_ms)
            events.emit(
                "tts_finished", turn=turn, audio_ms=reply_audio.duration_ms
            )

        self.turn_records.append(record)
        response.metadata = {
            **response.metadata,
            "audio": reply_audio,
            "voice": record,
        }
        return response

    async def stop_session(self, session: BotSession) -> None:
        await self.text_bot.stop_session(session)
