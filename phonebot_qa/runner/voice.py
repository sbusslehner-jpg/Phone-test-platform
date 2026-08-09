"""Voice end-to-end conversation runner (concept sections 12.2, 13, 14, 18, 34).

Runs an ordinary text scenario as a real phone call::

    UserSimulator -> TTS -> Audio Chaos -> SIP/WebRTC -> Phonebot
                  -> STT -> Bot Logic -> Bot TTS -> (barge-in) -> Voice Metrics

Nothing about the scenario changes: the same YAML that drives a text test drives
the voice test, and the same deterministic business assertions decide PASS/FAIL.
Voice adds *additional* failure modes — misrecognition under noise, missed
barge-in, latency blowouts — on top of the business contract, which is exactly
the split the concept describes in section 12.

The runner mirrors :class:`~phonebot_qa.runner.conversation.ConversationRunner`
and produces the same :class:`RunArtifacts`, so the evaluation pipeline,
regression store and release gate work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..adapters.bot.base import BotAdapter, SessionContext
from ..adapters.bot.voice import VoiceBotAdapter
from ..adapters.transport.base import VoiceTransport
from ..adapters.transport.loopback import LoopbackTransport
from ..audio.bargein import BargeInConfig, BargeInController, BargeInResult
from ..audio.chaos import AudioChaos, ChaosConfig
from ..audio.profiles import get_profile
from ..audio.stt import DeterministicSTT
from ..audio.tts import DeterministicTTS, TTSEngine
from ..backend.faults import FaultInjector
from ..backend.proxy import ToolProxy
from ..backend.tools import ToolRegistry, default_registry
from ..backend.world import World
from ..models import Conversation, Turn
from ..observability import EventLog
from ..simulator.base import UserSimulator
from .conversation import RunArtifacts


@dataclass
class VoiceConfig:
    """Voice-mode configuration for a run (concept §13/§14)."""

    #: Named audio profile (clean, street, car, bad_connection, ...).
    profile: str = "clean"
    #: Transport: "loopback" | "sip" | "webrtc".
    transport: str = "loopback"
    #: Optional barge-in test for this run.
    barge_in: BargeInConfig | None = None
    #: Whether the bot under test supports barge-in at all.
    supports_barge_in: bool = True
    barge_in_detection_ms: int = 120
    barge_in_stop_ms: int = 60
    sample_rate: int = 16000
    #: Extra overrides applied on top of the named profile.
    overrides: dict[str, Any] = field(default_factory=dict)

    def chaos_config(self) -> ChaosConfig:
        base = get_profile(self.profile)
        # ``overrides`` also carries assertion-only keys (max_wer,
        # barge_in_sla_ms, ...) that are consumed by the evaluator, not the
        # chaos layer — keep only real ChaosConfig fields.
        fields = set(ChaosConfig.__dataclass_fields__)
        tweaks = {k: v for k, v in self.overrides.items() if k in fields}
        if not tweaks:
            return base
        return ChaosConfig(**{**base.as_dict(), **tweaks})

    @classmethod
    def from_scenario(cls, scenario) -> "VoiceConfig":
        """Read an ``audio:`` block off a scenario, falling back to defaults."""
        raw = dict(getattr(scenario, "audio", None) or {})
        barge_raw = raw.pop("barge_in", None)
        cfg = cls(
            profile=raw.pop("profile", "clean"),
            transport=raw.pop("transport", "loopback"),
            supports_barge_in=raw.pop("supports_barge_in", True),
            barge_in_detection_ms=raw.pop("barge_in_detection_ms", 120),
            barge_in_stop_ms=raw.pop("barge_in_stop_ms", 60),
            sample_rate=raw.pop("sample_rate", 16000),
            overrides=raw,
        )
        if barge_raw:
            if isinstance(barge_raw, dict):
                cfg.barge_in = BargeInConfig(**barge_raw)
            else:
                cfg.barge_in = BargeInConfig()
        return cfg


def build_transport(
    config: VoiceConfig, *, events: EventLog, seed: int
) -> VoiceTransport:
    """Instantiate the configured transport (concept §11)."""
    chaos = config.chaos_config()
    kind = config.transport
    if kind == "sip":
        from ..adapters.transport.sip import SIPTransport

        return SIPTransport(seed=seed, events=events, latency_ms=chaos.latency_ms or 120)
    if kind == "webrtc":
        from ..adapters.transport.webrtc import WebRTCTransport

        return WebRTCTransport(seed=seed, events=events, latency_ms=chaos.latency_ms or 60)
    return LoopbackTransport(
        latency_ms=chaos.latency_ms,
        jitter_ms=chaos.jitter_ms,
        seed=seed,
    )


class VoiceConversationRunner:
    """Executes one scenario as a voice call."""

    def __init__(
        self,
        bot: BotAdapter,
        *,
        config: VoiceConfig | None = None,
        user_tts: TTSEngine | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        # Keep the *text* bot; the voice wrapper is built per run so it can be
        # seeded (recognition must vary with the case seed) and so concurrent
        # cases never share the wrapper's per-call turn records.
        self._template = bot if isinstance(bot, VoiceBotAdapter) else None
        self.text_bot = bot.text_bot if isinstance(bot, VoiceBotAdapter) else bot
        self.config = config or VoiceConfig()
        self.user_tts = user_tts or DeterministicTTS(voice="caller")
        self._registry = registry

    def _build_voice_bot(self, seed: int) -> VoiceBotAdapter:
        """A fresh, seed-scoped voice wrapper for one call."""
        template = self._template
        if template is not None:
            stt = template.stt
            # Re-seed the default recognizer so noise varies per case; a
            # caller-supplied custom engine is used as given.
            if isinstance(stt, DeterministicSTT):
                stt = DeterministicSTT(
                    latency_ms=stt.latency_ms, seed=seed, sensitivity=stt.sensitivity
                )
            return VoiceBotAdapter(
                template.text_bot,
                stt=stt,
                tts=template.tts,
                think_ms=template.think_ms,
                version=template.version,
            )
        return VoiceBotAdapter(self.text_bot, stt=DeterministicSTT(seed=seed))

    async def run(self, *, scenario, simulator: UserSimulator, seed: int = 0) -> RunArtifacts:
        config = self.config
        # Seed-scoped voice wrapper: recognition varies with the case seed.
        voice_bot = self._build_voice_bot(seed)
        registry = self._registry or default_registry()
        world = World(scenario.initial_state)
        events = EventLog()
        faults = FaultInjector(scenario.faults, seed=seed)
        proxy = ToolProxy(registry, world, events, faults)

        chaos = AudioChaos(config.chaos_config(), seed=seed)
        transport = build_transport(config, events=events, seed=seed)
        barge_controller = BargeInController(
            detection_ms=config.barge_in_detection_ms,
            stop_ms=config.barge_in_stop_ms,
            supports_barge_in=config.supports_barge_in,
        )

        conversation = Conversation()
        error: str | None = None
        barge_results: list[BargeInResult] = []
        wer_values: list[float] = []

        context = SessionContext(
            scenario_id=scenario.id,
            proxy=proxy,
            events=events,
            initial_state=scenario.initial_state,
            metadata={"seed": seed, "mode": "voice", "profile": config.profile},
        )

        try:
            session = await voice_bot.start_session(context)
            events.emit("session_started", mode="voice", profile=config.profile)
            await transport.connect(session_id=session.session_id)

            history: list[dict[str, str]] = []
            last_bot_message: str | None = None
            if session.greeting:
                history.append({"role": "assistant", "content": session.greeting})
                last_bot_message = session.greeting
                events.emit("bot_message", payload={"greeting": True})

            turn_index = 0
            while True:
                if turn_index >= scenario.limits.max_turns:
                    events.emit("limit_reached", reason="max_turns")
                    break
                if events.clock.now_ms / 1000.0 >= scenario.limits.max_duration_seconds:
                    events.emit("limit_reached", reason="max_duration")
                    break

                user_turn = await simulator.next_turn(history, last_bot_message)
                if not user_turn.text and (user_turn.finished or simulator.finished):
                    break

                turn_index += 1
                session.state["turn_index"] = turn_index

                # -- 1. Caller speaks -------------------------------------- #
                events.emit("user_audio_started", turn=turn_index, text=user_turn.text)
                spoken = self.user_tts.synthesize(
                    user_turn.text, sample_rate=config.sample_rate
                )
                events.clock.advance(spoken.duration_ms)
                events.emit(
                    "user_audio_finished", turn=turn_index, audio_ms=spoken.duration_ms
                )
                # Voice "latency" is *response* latency: from the moment the
                # caller stops speaking to the moment the bot starts speaking.
                # The caller's own speech and the bot's playback are call
                # duration, not responsiveness — mixing them in would make every
                # long utterance look like a slow bot (concept §18).
                t0 = events.clock.now_ms

                # -- 2. Chaos + transport ---------------------------------- #
                degraded = chaos.apply(spoken)
                arriving = await transport.send(degraded)
                stats = transport.stats()
                if stats.one_way_latency_ms:
                    events.clock.advance(stats.one_way_latency_ms)
                events.emit(
                    "audio_transported",
                    turn=turn_index,
                    transport=transport.name,
                    latency_ms=stats.one_way_latency_ms,
                    lost_ms=stats.lost_ms,
                    codec=stats.codec,
                )

                # -- 3. Bot: STT -> logic -> TTS --------------------------- #
                response = await voice_bot.send_audio(session, arriving)
                record = response.metadata.get("voice")
                if record is not None and record.wer is not None:
                    wer_values.append(record.wer)

                reply_audio = response.metadata.get("audio")
                bot_audio_ms = reply_audio.duration_ms if reply_audio is not None else 0
                # Response latency is measured here, before playback begins.
                latency = events.clock.now_ms - t0
                events.emit(
                    "bot_audio_started",
                    turn=turn_index,
                    audio_ms=bot_audio_ms,
                    response_latency_ms=latency,
                )

                # -- 4. Optional barge-in ---------------------------------- #
                barge = None
                if config.barge_in is not None:
                    barge = barge_controller.interrupt(
                        bot_audio_ms=bot_audio_ms,
                        config=config.barge_in,
                        events=events,
                        turn=turn_index,
                    )
                    if barge.attempted:
                        barge_results.append(barge)

                if barge is not None and barge.attempted and barge.detected:
                    # The bot stopped early; only the played part counts.
                    events.clock.advance(barge.bot_audio_stop_ms or bot_audio_ms)
                else:
                    events.clock.advance(bot_audio_ms)
                events.emit("bot_audio_finished", turn=turn_index)

                events.emit(
                    "bot_message", turn=turn_index, text=response.text, done=response.done
                )
                events.emit("turn_completed", turn=turn_index, latency_ms=latency)

                conversation.turns.append(
                    Turn(
                        index=turn_index,
                        # The transcript records what the caller *said*; what the
                        # bot heard is on the trace (stt_finished) and in the
                        # per-turn voice record.
                        user=user_turn.text,
                        bot=response.text,
                        latency_ms=latency,
                    )
                )
                history.append({"role": "user", "content": user_turn.text})
                history.append({"role": "assistant", "content": response.text})
                last_bot_message = response.text

                if response.done:
                    break
                if user_turn.finished or simulator.finished:
                    break

            await transport.disconnect()
            await voice_bot.stop_session(session)
            events.emit("session_ended")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            events.emit("run_error", error=error)

        conversation.duration_ms = events.clock.now_ms
        avg_wer = sum(wer_values) / len(wer_values) if wer_values else None
        return RunArtifacts(
            conversation=conversation,
            events=events.events,
            tool_calls=proxy.calls,
            final_state=world.snapshot(),
            world=world,
            error=error,
            metadata={
                "mode": "voice",
                "profile": config.profile,
                "transport": transport.name,
                "stt_wer": avg_wer,
                "barge_in": [b.as_dict() for b in barge_results],
                "voice_turns": list(voice_bot.turn_records),
            },
        )
