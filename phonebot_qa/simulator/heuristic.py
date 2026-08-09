"""A goal-driven heuristic simulator (concept sections 8-10).

This is the default cooperative caller. It is *LLM-shaped* — it reads a goal, a
persona and the running conversation and decides the next utterance — but it is
implemented with deterministic rules so the platform runs with no API keys and
every run is perfectly reproducible given the seed. A real LLM simulator (see
:mod:`phonebot_qa.simulator.llm`) plugs into the same interface.

It only ever reads its :class:`~phonebot_qa.scenario.knowledge.KnowledgeView`, so
it cannot see the evaluator's expectations (section 10). Persona traits shape
*phrasing* (verbosity, impatience filler); the *semantics* (target time, an
explicit correction) come from the caller's own visible knowledge, which keeps
the expected outcome deterministic regardless of persona.
"""

from __future__ import annotations

import random
from typing import Any

from .base import SimulatorTurn, UserSimulator

_CONFIRM_REQUEST_KW = (
    "ist das korrekt",
    "korrekt?",
    "richtig?",
    "wirklich absagen",
    "soll ich das so",
    "bestätig",
    "ist das so richtig",
)
_CLARIFY_TIME_KW = (
    "auf welchen zeitpunkt",
    "für wann",
    "welche uhrzeit",
    "wann möchten",
    "wann möchtest",
)
_NOT_AVAILABLE_KW = ("nicht frei", "nicht verfügbar", "leider nicht frei")
_NOT_FOUND_KW = ("finde leider keinen", "keinen termin auf ihren", "finde keinen termin")


def _any(text: str, kws: tuple[str, ...]) -> bool:
    return any(k in text for k in kws)


class HeuristicSimulator(UserSimulator):
    """Cooperative, persona-flavoured caller pursuing a single goal."""

    #: internal safety cap; the runner also enforces scenario ``max_turns``.
    _MAX_UTTERANCES = 10

    def __init__(self, knowledge, *, seed: int = 0) -> None:
        super().__init__(knowledge, seed=seed)
        self._rng = random.Random(seed)
        self._opened = False
        self._phase = "opening"
        self._utterances = 0
        self._unknown_streak = 0

    # -- knowledge accessors ---------------------------------------------- #

    def _visible(self, *keys: str, default: Any = None) -> Any:
        vis = self.knowledge.user_visible
        for k in keys:
            if k in vis:
                return vis[k]
        return default

    def _goal_attr(self, name: str) -> Any:
        # UserGoal allows extra fields (Pydantic ``extra="allow"``); those are
        # reachable via getattr and also collected in ``model_extra``.
        value = getattr(self.goal, name, None)
        if value is None and self.goal.model_extra:
            value = self.goal.model_extra.get(name)
        return value

    def _desired_time(self) -> str | None:
        return (
            self._goal_attr("target_datetime")
            or self._visible("desired_time", "target_datetime")
        )

    def _correction(self) -> dict | None:
        corr = self._visible("correction")
        return corr if isinstance(corr, dict) else None

    def _alternative_time(self) -> str | None:
        return self._visible("alternative_time")

    # -- persona flavour --------------------------------------------------- #

    def _flavour(self, core: str) -> str:
        """Add persona-driven, semantics-preserving phrasing."""
        persona = self.persona
        prefix = ""
        suffix = ""
        if persona.patience == "low" and self._rng.random() < 0.5:
            prefix = "Schnell bitte, "
        if persona.verbosity == "long" and self._rng.random() < 0.6:
            suffix = " Vielen Dank für Ihre Hilfe."
        if persona.language_proficiency == "low":
            core = core.replace("möchte", "will").replace("Ich würde gerne", "Ich will")
        text = f"{prefix}{core}{suffix}".strip()
        # Capitalise after a lowercase prefix join.
        return text[0].upper() + text[1:] if text else text

    # -- main -------------------------------------------------------------- #

    async def next_turn(
        self, history: list[dict[str, str]], last_bot_message: str | None
    ) -> SimulatorTurn:
        self._utterances += 1
        if self._utterances > self._MAX_UTTERANCES:
            self.finished = True
            return SimulatorTurn(text="Ich versuche es später noch einmal. Auf Wiederhören.", finished=True)

        if not self._opened:
            self._opened = True
            return SimulatorTurn(text=self._opening())

        bot = (last_bot_message or "").lower()

        if _any(bot, _NOT_FOUND_KW):
            self.finished = True
            return SimulatorTurn(
                text="Das ist seltsam, ich melde mich später noch einmal.",
                finished=True,
            )

        if _any(bot, _NOT_AVAILABLE_KW):
            alt = self._alternative_time()
            if alt:
                return SimulatorTurn(
                    text=self._flavour(f"Dann nehmen wir bitte {alt}.")
                )
            self.finished = True
            return SimulatorTurn(
                text="Schade, dann lasse ich es für heute.", finished=True
            )

        if _any(bot, _CLARIFY_TIME_KW):
            target = self._desired_time()
            if target:
                return SimulatorTurn(text=self._flavour(f"Am {target}, bitte."))
            return SimulatorTurn(text="Am liebsten so bald wie möglich.")

        if _any(bot, _CONFIRM_REQUEST_KW):
            self._unknown_streak = 0
            if self._phase == "await_correct":
                self._phase = "await_confirm"
                desired = self._desired_time()
                return SimulatorTurn(
                    text=self._flavour(
                        f"Nein, entschuldigung — eigentlich möchte ich auf {desired}."
                    )
                )
            self._phase = "closing"
            return SimulatorTurn(text=self._confirm_phrase())

        # Unrecognised bot turn — restate the goal, then give up if it persists.
        self._unknown_streak += 1
        if self._unknown_streak >= 3:
            self.finished = True
            return SimulatorTurn(
                text="Das klappt gerade nicht, ich probiere es später. Danke.",
                finished=True,
            )
        return SimulatorTurn(text=self._restate())

    # -- utterance builders ------------------------------------------------ #

    def _opening(self) -> str:
        goal_type = self.goal.type
        correction = self._correction()
        desired = self._desired_time()

        if goal_type in ("move_appointment", "reschedule_appointment"):
            if correction and correction.get("mistaken_time"):
                self._phase = "await_correct"
                mistaken = correction["mistaken_time"]
                return self._flavour(
                    f"Guten Tag, ich möchte meinen Termin verschieben, und zwar auf {mistaken}."
                )
            self._phase = "await_confirm"
            return self._flavour(
                f"Guten Tag, ich möchte meinen Termin verschieben, und zwar auf {desired}."
            )
        if goal_type in ("cancel_appointment",):
            self._phase = "await_confirm"
            return self._flavour("Guten Tag, ich möchte meinen Termin absagen.")
        if goal_type in ("book_appointment", "create_appointment"):
            self._phase = "await_confirm"
            return self._flavour(
                f"Guten Tag, ich möchte einen neuen Termin vereinbaren, am {desired}."
            )
        # Generic fallback.
        self._phase = "await_confirm"
        return self._flavour(f"Guten Tag, es geht um {goal_type}.")

    def _restate(self) -> str:
        desired = self._desired_time()
        if self.goal.type in ("move_appointment", "reschedule_appointment") and desired:
            return self._flavour(f"Ich möchte meinen Termin auf {desired} verschieben.")
        if self.goal.type == "cancel_appointment":
            return self._flavour("Ich möchte meinen Termin absagen.")
        return self._flavour("Können Sie mir bitte weiterhelfen?")

    def _confirm_phrase(self) -> str:
        if self.persona.verbosity == "short":
            return "Ja."
        return self._flavour("Ja, genau, das ist korrekt.")
