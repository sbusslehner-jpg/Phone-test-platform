"""Barge-in testing (concept section 14).

Interruptions must be tested explicitly. The scenario schedules an interrupt at
``interrupt_after_ms`` into the bot's utterance::

    Bot:  "Sehr gerne, ich kann Ihren Termin..."
    t = 850 ms
    User: "Nein, ich möchte ihn eigentlich absagen."

The controller measures the timeline the concept calls for — ``interrupt_start``,
``barge_in_detected``, ``bot_audio_stop``, ``user_stt_start``,
``bot_response_start`` — and produces::

    {"barge_in_detected": true, "stop_latency_ms": 180, "user_audio_lost_ms": 70}

against the SLA **barge-in detection < 300 ms**.

``user_audio_lost_ms`` is the part of the caller's speech that landed while the
bot was still talking over them and its recognizer had not yet opened — the
classic cause of "the bot ignored what I said".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..observability import EventLog

#: Concept §14 SLA: the bot must detect an interruption within this budget.
BARGE_IN_SLA_MS = 300


@dataclass
class BargeInConfig:
    """How and when to interrupt the bot."""

    #: Interrupt this many ms into the bot's utterance.
    interrupt_after_ms: int = 850
    #: What the caller says over the bot.
    utterance: str | None = None
    #: Only interrupt if the bot is still speaking after this much audio.
    min_bot_audio_ms: int = 300


@dataclass
class BargeInResult:
    """Measured barge-in behaviour for one interruption."""

    attempted: bool = False
    detected: bool = False
    interrupt_start_ms: int = 0
    detected_at_ms: int | None = None
    bot_audio_stop_ms: int | None = None
    stop_latency_ms: int | None = None
    user_audio_lost_ms: int | None = None
    within_sla: bool | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "barge_in_detected": self.detected,
            "stop_latency_ms": self.stop_latency_ms,
            "user_audio_lost_ms": self.user_audio_lost_ms,
            "within_sla": self.within_sla,
        }


class BargeInController:
    """Injects an interruption and measures how the bot handles it.

    ``detection_ms`` and ``stop_ms`` describe the *bot under test*: how long its
    voice-activity detector takes to notice the caller, and how long it then
    takes to actually stop its audio. A bot that does not support barge-in at all
    is modelled with ``supports_barge_in=False`` — it keeps talking, the caller's
    whole utterance is lost, and the test fails.
    """

    def __init__(
        self,
        *,
        detection_ms: int = 120,
        stop_ms: int = 60,
        supports_barge_in: bool = True,
    ) -> None:
        self.detection_ms = detection_ms
        self.stop_ms = stop_ms
        self.supports_barge_in = supports_barge_in

    def interrupt(
        self,
        *,
        bot_audio_ms: int,
        config: BargeInConfig,
        events: EventLog | None = None,
        turn: int | None = None,
        sla_ms: int = BARGE_IN_SLA_MS,
    ) -> BargeInResult:
        """Interrupt an in-progress bot utterance and measure the response.

        ``sla_ms`` lets a scenario tighten or relax the detection budget; it must
        match the value the evaluator asserts against, otherwise
        :attr:`BargeInResult.within_sla` and the ``voice:barge_in_sla``
        assertion could disagree.
        """
        result = BargeInResult()
        if bot_audio_ms < config.min_bot_audio_ms:
            # The bot finished before the caller could reasonably interrupt.
            result.metadata["skipped"] = "bot utterance shorter than min_bot_audio_ms"
            return result
        if config.interrupt_after_ms >= bot_audio_ms:
            result.metadata["skipped"] = "interrupt point is past the end of the utterance"
            return result

        result.attempted = True
        if config.utterance and events is not None:
            # Record what the caller talked over the bot with, so the trace
            # explains the interruption rather than just timing it.
            events.emit(
                "interrupt_utterance", turn=turn, text=config.utterance
            )
        start = config.interrupt_after_ms
        result.interrupt_start_ms = start
        if events is not None:
            events.emit("interrupt_start", turn=turn, at_ms=start)

        if not self.supports_barge_in:
            # The bot talks over the caller for the rest of its utterance.
            result.detected = False
            result.user_audio_lost_ms = bot_audio_ms - start
            result.within_sla = False
            if events is not None:
                events.emit(
                    "barge_in_missed",
                    turn=turn,
                    user_audio_lost_ms=result.user_audio_lost_ms,
                )
            return result

        detected_at = start + self.detection_ms
        stop_at = detected_at + self.stop_ms
        result.detected = True
        result.detected_at_ms = detected_at
        result.bot_audio_stop_ms = stop_at
        result.stop_latency_ms = stop_at - start
        # Everything the caller said before the bot went quiet is degraded.
        result.user_audio_lost_ms = max(0, min(bot_audio_ms, stop_at) - start)
        result.within_sla = result.stop_latency_ms <= sla_ms

        if events is not None:
            events.emit(
                "barge_in_detected",
                turn=turn,
                at_ms=detected_at,
                stop_latency_ms=result.stop_latency_ms,
                user_audio_lost_ms=result.user_audio_lost_ms,
                within_sla=result.within_sla,
            )
            events.emit("bot_audio_stop", turn=turn, at_ms=stop_at)
            events.emit("user_stt_start", turn=turn, at_ms=stop_at)
        return result
