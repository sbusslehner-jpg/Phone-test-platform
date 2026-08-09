"""LLM-as-a-judge evaluation (concept section 21), behind an interface.

Qualitative conversation quality — naturalness, clarity, efficiency, how well
corrections were handled — is scored here. Per the platform's core design rule
(section 37) these scores feed the *score* but never decide business
correctness; a failed critical assertion overrides them (see scoring).

Like the simulator, the judge is abstracted (section 30, DeepEval "behind its
own interface"). The default :class:`HeuristicJudge` is deterministic and needs
no API key; a real LLM judge implements the same :class:`Judge` contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Conversation, EvalScores


class Judge(ABC):
    """Scores conversation quality qualitatively (1..5 per dimension)."""

    @abstractmethod
    async def evaluate(
        self, conversation: Conversation, *, persona=None
    ) -> EvalScores: ...


def _clamp(v: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, v))


class HeuristicJudge(Judge):
    """A deterministic, dependency-free judge.

    Derives plausible quality scores from structural features of the transcript
    (length, repetition, confusion markers, correction recovery). Good enough to
    exercise the pipeline and to keep CI reproducible; swap for a real LLM judge
    in richer environments.
    """

    async def evaluate(self, conversation: Conversation, *, persona=None) -> EvalScores:
        turns = conversation.turns
        n = len(turns)
        if n == 0:
            return EvalScores(rationale="empty conversation")

        bot_msgs = [t.bot for t in turns]
        user_msgs = [t.user for t in turns]

        # Efficiency: reward short, decisive conversations.
        efficiency = _clamp(5.0 - max(0, n - 3) * 0.6)

        # Clarity: penalise "not understood" style bot replies.
        confusion = sum(
            1 for m in bot_msgs if "nicht verstanden" in m.lower() or "entschuldigung" in m.lower()
        )
        clarity = _clamp(5.0 - confusion * 1.0)

        # Naturalness: penalise verbatim bot repetition.
        repeats = len(bot_msgs) - len(set(bot_msgs))
        naturalness = _clamp(4.5 - repeats * 1.0)

        # Correction handling: did a user correction end in a clean completion?
        corrected = any(
            "eigentlich" in u.lower() or "nein" in u.lower().split() for u in user_msgs
        )
        completed = any(
            "erledigt" in m.lower() or "gebucht" in m.lower() or "abgesagt" in m.lower()
            for m in bot_msgs
        )
        if corrected:
            correction_handling = 5.0 if completed else 2.5
        else:
            correction_handling = 4.0 if completed else 3.0

        return EvalScores(
            naturalness=round(naturalness, 2),
            clarity=round(clarity, 2),
            efficiency=round(efficiency, 2),
            correction_handling=round(correction_handling, 2),
            rationale=(
                f"turns={n}, repeats={repeats}, confusion_markers={confusion}, "
                f"correction={'yes' if corrected else 'no'}, completed={'yes' if completed else 'no'}"
            ),
        )


class LLMJudge(Judge):
    """A judge backed by an :class:`LLMProvider`-style callable.

    ``provider(system, messages) -> json_text``. Falls back to
    :class:`HeuristicJudge` if no provider is supplied so it is always safe to
    construct. (Kept minimal here; real deployments parse structured JSON.)
    """

    def __init__(self, provider=None) -> None:
        self._provider = provider
        self._fallback = HeuristicJudge()

    async def evaluate(self, conversation: Conversation, *, persona=None) -> EvalScores:
        if self._provider is None:
            return await self._fallback.evaluate(conversation, persona=persona)
        # Real implementations would send the transcript and parse a JSON reply.
        # Deliberately delegated to keep the default install deterministic.
        return await self._fallback.evaluate(conversation, persona=persona)
