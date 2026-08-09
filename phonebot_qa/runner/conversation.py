"""The conversation runner (concept section 7).

Orchestrates one multi-turn conversation between a :class:`UserSimulator` and a
:class:`BotAdapter`, wiring in the mock backend, tool proxy and event log. It
owns session lifecycle, turn counting, the timeout/turn limits and transcript
logging, and returns the raw artifacts (transcript, events, tool calls, final
backend state) that the evaluation pipeline consumes.

Timing is measured on the logical clock (see
:class:`~phonebot_qa.observability.events.Clock`): tool calls advance it by their
simulated latency, and each bot turn adds a nominal "thinking" latency so the
latency metrics (section 18/26) are meaningful and reproducible.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..adapters.bot.base import BotAdapter, SessionContext
from ..backend.faults import FaultInjector
from ..backend.proxy import ToolProxy
from ..backend.tools import ToolRegistry, default_registry
from ..backend.world import World
from ..degradation import compile_patterns, is_degraded
from ..models import Conversation, Event, ToolCall, Turn
from ..observability import EventLog
from ..simulator.base import UserSimulator


@dataclass
class RunnerConfig:
    """Tunable runner parameters."""

    # Nominal per-turn bot processing latency (STT+LLM+TTS-ish) in text mode.
    bot_think_ms: int = 400
    registry_factory: Any = default_registry
    # Bot-agnostische Fallback-Muster (M3/P1, regex-fähig): jeder Bot-Turn, der
    # matcht, wird als degraded markiert — zusätzlich zu Adaptern, die selbst
    # ``BotResponse.metadata["degraded"]`` setzen. Default: keine (der
    # Cross3Adapter bringt seine CROSS3-Muster selbst mit).
    fallback_patterns: Sequence[str] | None = None


@dataclass
class RunArtifacts:
    """Everything produced by one run, for the evaluation pipeline."""

    conversation: Conversation
    events: list[Event]
    tool_calls: list[ToolCall]
    final_state: dict[str, Any]
    world: World
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationRunner:
    """Runs a single scenario conversation end-to-end."""

    def __init__(
        self,
        bot: BotAdapter,
        *,
        config: RunnerConfig | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.bot = bot
        self.config = config or RunnerConfig()
        self._registry = registry

    async def run(
        self,
        *,
        scenario,
        simulator: UserSimulator,
        seed: int = 0,
    ) -> RunArtifacts:
        registry = self._registry or self.config.registry_factory()
        world = World(scenario.initial_state)
        events = EventLog()
        faults = FaultInjector(scenario.faults, seed=seed)
        proxy = ToolProxy(registry, world, events, faults)

        conversation = Conversation()
        error: str | None = None
        fallback_patterns = compile_patterns(self.config.fallback_patterns)

        context = SessionContext(
            scenario_id=scenario.id,
            proxy=proxy,
            events=events,
            initial_state=scenario.initial_state,
            metadata={"seed": seed},
        )

        session = None
        try:
            session = await self.bot.start_session(context)
            events.emit("session_started")

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
                events.emit("user_message", turn=turn_index, text=user_turn.text)
                history.append({"role": "user", "content": user_turn.text})

                t0 = events.clock.now_ms
                wall_t0 = time.perf_counter()
                response = await self.bot.send_text(session, user_turn.text)
                # Echte Wanduhr-Millisekunden dieses Turns (M3): nur Messwert,
                # fließt NICHT in die logische Clock — Determinismus bleibt.
                wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                # Nominal bot processing latency on top of any tool latency.
                events.clock.advance(self.config.bot_think_ms)
                latency = events.clock.now_ms - t0

                # Degradations-Erkennung (M3/P1): Adapter-Markierung ODER die
                # bot-agnostischen Runner-Muster.
                degraded = bool(response.metadata.get("degraded")) or is_degraded(
                    response.text, fallback_patterns
                )
                if degraded:
                    events.emit("bot_degraded", turn=turn_index)

                events.emit(
                    "bot_message", turn=turn_index, text=response.text, done=response.done
                )
                events.emit("turn_completed", turn=turn_index, latency_ms=latency)
                conversation.turns.append(
                    Turn(
                        index=turn_index,
                        user=user_turn.text,
                        bot=response.text,
                        latency_ms=latency,
                        wall_latency_ms=round(wall_ms, 3),
                        degraded=degraded,
                    )
                )
                history.append({"role": "assistant", "content": response.text})
                last_bot_message = response.text

                if response.done:
                    break
                if user_turn.finished or simulator.finished:
                    break
        except Exception as exc:  # capture and surface as an ERROR result
            error = f"{type(exc).__name__}: {exc}"
            events.emit("run_error", error=error)
        finally:
            # stop_session läuft IMMER, auch wenn der Lauf mittendrin crasht
            # (M3/P2): Adapter räumen dort Sitzungs-Ressourcen auf und
            # entwaffnen z. B. einen noch scharfen CROSS3-Fault — der sonst in
            # den nächsten Case durchsickern würde. ``session_ended`` bleibt
            # wie bisher dem fehlerfreien Lauf vorbehalten.
            if session is not None:
                try:
                    await self.bot.stop_session(session)
                except Exception as exc:  # pragma: no cover - defensive
                    if error is None:
                        error = f"{type(exc).__name__}: {exc}"
                        events.emit("run_error", error=error)
                else:
                    if error is None:
                        events.emit("session_ended")

        conversation.duration_ms = events.clock.now_ms
        return RunArtifacts(
            conversation=conversation,
            events=events.events,
            tool_calls=proxy.calls,
            final_state=world.snapshot(),
            world=world,
            error=error,
        )
