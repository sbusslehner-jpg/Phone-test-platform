"""Turn-based voice runner for CROSS3 over the phone-media relay (concept §12.2).

Drives a scripted caller through CROSS3's real voice bridge — TTS → audio chaos →
``/ws/phone-media`` (SLIN 8 kHz) → CROSS3 → bot audio back — and produces the
platform's standard :class:`~phonebot_qa.runner.conversation.RunArtifacts`, so the
existing evaluation pipeline scores the call: response latency (§18), barge-in
against the 300 ms SLA (§14), hangup behaviour, and — when a real
:class:`~phonebot_qa.audio.stt.STTEngine` is supplied — a transcript for the LLM
judge. Backend truth (did the booking land?) is read back from CROSS3's mock via
an injected ``state_reader``.

Honest boundaries (see docs/TESTING_CROSS3.md):
* The relay carries no tool calls, so tool/backend assertions stay on the chat
  channel; here the backend check is the post-call ``state_reader``.
* Without a real STT, the bot transcript is empty — barge-in / latency / hangup
  still score; conversation quality does not.
* CROSS3 does its own STT/TTS internally, so the internal STT/LLM/TTS latency
  spans are not visible on the relay; ``response latency`` is the meaningful,
  measurable timing here.

Because CROSS3 voice is a native full-duplex bot (not the platform's
TTS→transport→VoiceBotAdapter chain), it has its own runner rather than going
through the standard engine.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from ..adapters.transport.cross3_phone import Cross3VoicePhoneClient
from ..audio.chaos import AudioChaos
from ..audio.stt import STTEngine
from ..audio.tts import DeterministicTTS, TTSEngine
from ..backend.world import World
from ..evaluation.pipeline import EvaluationPipeline
from ..models import Conversation, Turn
from ..observability import EventLog
from ..scenario.knowledge import isolate_user_knowledge
from ..simulator.base import UserSimulator
from ..simulator.personas import resolve_persona
from ..simulator.scripted import ScriptedSimulator
from .conversation import RunArtifacts
from .voice import VoiceConfig

#: CROSS3's relay speaks SLIN at 8 kHz.
VOICE_SAMPLE_RATE = 8000

#: async ``state_reader(client) -> dict`` merged into the world (backend truth).
StateReader = Callable[[Cross3VoicePhoneClient], Awaitable[dict]]


class Cross3VoiceRunner:
    """Run one scenario as a real voice call against CROSS3."""

    def __init__(
        self,
        client_factory: Callable[[], Cross3VoicePhoneClient],
        *,
        stt: STTEngine | None = None,
        user_tts: TTSEngine | None = None,
        state_reader: StateReader | None = None,
        quiet_ms: int = 600,
    ) -> None:
        self.client_factory = client_factory
        self.stt = stt
        self.user_tts = user_tts or DeterministicTTS(voice="caller")
        self.state_reader = state_reader
        self.quiet_ms = quiet_ms

    def _caller_audio(self, text: str, chaos: AudioChaos):
        spoken = self.user_tts.synthesize(text, sample_rate=VOICE_SAMPLE_RATE)
        return chaos.apply(spoken)

    def _transcribe(self, turn) -> tuple[str, float | None]:
        if self.stt is None or not turn.audio:
            return "", None
        result = self.stt.transcribe(turn.to_buffer())
        return result.text, result.wer

    async def run(self, *, scenario, simulator: UserSimulator, seed: int = 0) -> RunArtifacts:
        cfg = VoiceConfig.from_scenario(scenario)
        chaos = AudioChaos(cfg.chaos_config(), seed=seed)
        state = scenario.initial_state or {}
        did = str(state.get("did") or state.get("tenant_id") or "AT997")
        caller_id = str(state.get("caller_phone") or "")

        world = World(scenario.initial_state)
        events = EventLog()
        conversation = Conversation()
        error: str | None = None
        barge_records: list[dict] = []
        wer_values: list[float] = []
        client = self.client_factory()
        call_start = time.monotonic()

        try:
            await client.connect(did=did, caller_id=caller_id)
            events.emit("session_started", mode="voice", profile=cfg.profile)

            # The agent greets first. If the scenario tests barge-in, interrupt
            # that greeting; otherwise just listen to it.
            history: list[dict[str, str]] = []
            last_bot: str | None = None
            if cfg.barge_in is not None:
                interrupt_text = cfg.barge_in.utterance or "Nein, warten Sie bitte kurz."
                interrupt_audio = self._caller_audio(interrupt_text, chaos)
                greeting = await client.bot_turn_with_barge_in(
                    interrupt_audio,
                    interrupt_after_ms=cfg.barge_in.interrupt_after_ms,
                    quiet_ms=self.quiet_ms,
                )
                # Only score barge-in if the probe actually fired. A greeting too
                # short to talk over leaves interrupt_attempted False → record
                # nothing, so voice_assertions reports "the test did not run"
                # (voice:barge_in_attempted) rather than a false "bot did not stop".
                if greeting.interrupt_attempted:
                    self._record_barge_in(events, greeting, cfg, turn_index=0)
                    barge_records.append(self._barge_dict(greeting, cfg))
            else:
                greeting = await client.next_bot_turn(quiet_ms=self.quiet_ms)
            self._emit_bot_turn(events, greeting, turn=0)
            greet_text, _ = self._transcribe(greeting)
            last_bot = greet_text or None

            turn_index = 0
            if greeting.ended:
                # The agent ended the call at the greeting (hangup or transfer) —
                # emit the end event and skip the turn loop; the socket is closed,
                # so trying to speak would surface as a spurious run error (§12.2).
                self._emit_call_end(events, greeting, turn=0)
            else:
                while True:
                    if turn_index >= scenario.limits.max_turns:
                        events.emit("limit_reached", reason="max_turns")
                        break
                    user_turn = await simulator.next_turn(history, last_bot)
                    if not user_turn.text and (user_turn.finished or simulator.finished):
                        break

                    turn_index += 1
                    events.emit("user_audio_started", turn=turn_index, text=user_turn.text)
                    caller_audio = self._caller_audio(user_turn.text, chaos)
                    t0 = time.monotonic()
                    await client.send_audio(caller_audio)
                    events.emit(
                        "user_audio_finished", turn=turn_index, audio_ms=caller_audio.duration_ms
                    )

                    bot_turn = await client.next_bot_turn(quiet_ms=self.quiet_ms)
                    latency = bot_turn.first_audio_latency_ms
                    if latency is None:
                        latency = int((time.monotonic() - t0) * 1000)

                    bot_text, wer = self._transcribe(bot_turn)
                    if wer is not None:
                        wer_values.append(wer)
                    events.emit("stt_started", turn=turn_index)
                    events.emit("stt_finished", turn=turn_index, text=bot_text, wer=wer)
                    self._emit_bot_turn(events, bot_turn, turn=turn_index, latency=latency)

                    conversation.turns.append(
                        Turn(index=turn_index, user=user_turn.text, bot=bot_text, latency_ms=latency)
                    )
                    history.append({"role": "user", "content": user_turn.text})
                    history.append({"role": "assistant", "content": bot_text})
                    last_bot = bot_text or last_bot

                    if bot_turn.ended:
                        self._emit_call_end(events, bot_turn, turn=turn_index)
                        break
                    if user_turn.finished or simulator.finished:
                        break

            # Backend truth: read CROSS3's mock state (did the booking land?).
            if self.state_reader is not None:
                try:
                    snapshot = await self.state_reader(client)
                    if isinstance(snapshot, dict):
                        for collection, records in snapshot.items():
                            if isinstance(records, list):
                                for rec in records:
                                    world.put(collection, rec)
                except Exception as exc:  # pragma: no cover - reader is user-supplied
                    events.emit("state_read_error", error=f"{type(exc).__name__}: {exc}")

            await client.send_stop()
            await client.close()
            events.emit("session_ended")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            events.emit("run_error", error=error)
            try:
                await client.close()
            except Exception:  # pragma: no cover
                pass

        conversation.duration_ms = int((time.monotonic() - call_start) * 1000)
        avg_wer = sum(wer_values) / len(wer_values) if wer_values else None
        return RunArtifacts(
            conversation=conversation,
            events=events.events,
            tool_calls=[],  # the relay carries no tool calls (see module docstring)
            final_state=world.snapshot(),
            world=world,
            error=error,
            metadata={
                "mode": "voice",
                "channel": "cross3-phone",
                "profile": cfg.profile,
                "stt_wer": avg_wer,
                "barge_in": barge_records,
            },
        )

    # -- event helpers ----------------------------------------------------- #

    def _emit_bot_turn(self, events, bot_turn, *, turn, latency=None) -> None:
        # Barge-in events are emitted by _record_barge_in, not here, to avoid
        # double counting when the greeting turn was a barge-in probe.
        events.emit("bot_audio_started", turn=turn, audio_ms=bot_turn.duration_ms)
        events.emit("bot_audio_finished", turn=turn)
        if latency is not None:
            events.emit("turn_completed", turn=turn, latency_ms=latency)

    def _emit_call_end(self, events, bot_turn, *, turn) -> None:
        # The agent ended the call: a hand-off to a human is a transfer, anything
        # else is a hangup. Both stop the conversation (concept §12.2).
        if bot_turn.transferred_to is not None or bot_turn.hangup_reason == "transfer":
            events.emit("transfer", turn=turn, to=bot_turn.transferred_to)
        else:
            events.emit("hangup", turn=turn, reason=bot_turn.hangup_reason)

    def _barge_dict(self, turn, cfg) -> dict:
        within = None
        if turn.stop_latency_ms is not None:
            within = turn.stop_latency_ms <= cfg.barge_in_sla_ms
        return {
            "attempted": True,
            "barge_in_detected": turn.barge_in,
            "stop_latency_ms": turn.stop_latency_ms,
            "user_audio_lost_ms": turn.user_audio_lost_ms,
            "within_sla": within,
        }

    def _record_barge_in(self, events, bot_turn, cfg, *, turn_index=0) -> None:
        events.emit("interrupt_start", turn=turn_index, at_ms=cfg.barge_in.interrupt_after_ms)
        if bot_turn.barge_in:
            events.emit(
                "barge_in_detected",
                turn=turn_index,
                stop_latency_ms=bot_turn.stop_latency_ms,
                user_audio_lost_ms=bot_turn.user_audio_lost_ms,
                within_sla=(
                    bot_turn.stop_latency_ms is not None
                    and bot_turn.stop_latency_ms <= cfg.barge_in_sla_ms
                ),
            )
        else:
            events.emit("barge_in_missed", turn=turn_index)


async def run_cross3_voice_suite(
    scenarios,
    client_factory: Callable[[], Cross3VoicePhoneClient],
    *,
    stt: STTEngine | None = None,
    state_reader: StateReader | None = None,
    personas: dict | None = None,
    seeds: list[int] | None = None,
    bot_version: str = "cross3-voice",
    quiet_ms: int = 600,
):
    """Run voice scenarios against CROSS3 and aggregate a :class:`RunSummary`.

    ``client_factory`` returns a fresh :class:`Cross3VoicePhoneClient` per case.
    Callers are scripted from ``user.user_visible.redteam_lines`` (the free-form
    heuristic caller cannot react to audio it never transcribes).
    """
    from ..orchestrator.engine import summarize

    pipeline = EvaluationPipeline()
    runner = Cross3VoiceRunner(
        client_factory, stt=stt, state_reader=state_reader, quiet_ms=quiet_ms
    )
    results = []
    for scenario in scenarios:
        persona = resolve_persona(scenario.user.persona, personas or {})
        knowledge = isolate_user_knowledge(scenario, persona)
        for seed in seeds or [0]:
            lines = scenario.user.user_visible.get("redteam_lines", [])
            simulator = ScriptedSimulator(knowledge, lines=list(lines), seed=seed)
            artifacts = await runner.run(scenario=scenario, simulator=simulator, seed=seed)
            results.append(
                await pipeline.evaluate(
                    scenario,
                    artifacts,
                    case_id=f"{scenario.id}__seed{seed}__voice",
                    persona_id=scenario.user.persona,
                    bot_version=bot_version,
                    mode="voice",
                    seed=seed,
                )
            )
    return summarize(results, bot_version)
