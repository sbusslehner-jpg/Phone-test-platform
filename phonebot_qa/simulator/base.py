"""The :class:`UserSimulator` interface (concept section 8).

Implementations receive the caller's knowledge view plus the conversation
history and produce the next user utterance. They also decide when the caller is
"finished" (goal achieved or given up). Keeping this an interface means the
platform does not depend on any single LLM or framework — DeepEval, a raw LLM,
a local model or a scripted simulator can all be dropped in (section 8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..scenario.knowledge import KnowledgeView


@dataclass
class SimulatorTurn:
    """One simulated user utterance plus control signals."""

    text: str
    # The simulator signals completion here rather than the runner guessing.
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class UserSimulator(ABC):
    """Abstract caller. One instance drives one conversation."""

    def __init__(self, knowledge: KnowledgeView, *, seed: int = 0) -> None:
        self.knowledge = knowledge
        self.seed = seed
        self.finished = False

    @property
    def persona(self):
        return self.knowledge.persona

    @property
    def goal(self):
        return self.knowledge.goal

    @abstractmethod
    async def next_turn(
        self, history: list[dict[str, str]], last_bot_message: str | None
    ) -> SimulatorTurn:
        """Produce the next user utterance given the conversation so far.

        ``history`` is a role-tagged list of ``{"role", "content"}`` messages
        (the platform's transcript). ``last_bot_message`` is the most recent
        bot turn (``None`` before the bot has spoken).
        """
