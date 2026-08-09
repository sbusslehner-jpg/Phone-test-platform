"""Pluggable LLM user simulator (concept section 8).

Keeps the platform independent of any specific LLM/framework. An
:class:`LLMProvider` is any callable that turns a system prompt + chat messages
into a string reply — wire in Anthropic, a local model, or DeepEval's simulator
behind it. When no provider is supplied the simulator transparently falls back
to the deterministic :class:`~phonebot_qa.simulator.heuristic.HeuristicSimulator`
so tests and CI run with no API keys.

The prompt is built ONLY from the caller's knowledge view (section 10); the
evaluator's expectations are never in scope, so a real LLM caller cannot cheat.
"""

from __future__ import annotations

from typing import Protocol

from .base import SimulatorTurn, UserSimulator
from .heuristic import HeuristicSimulator


class LLMProvider(Protocol):
    """Minimal provider contract: (system, messages) -> assistant text."""

    def __call__(
        self, system: str, messages: list[dict[str, str]]
    ) -> str: ...  # pragma: no cover - interface


class LLMSimulator(UserSimulator):
    """Drive the caller with a real LLM behind :class:`LLMProvider`."""

    def __init__(
        self,
        knowledge,
        *,
        provider: LLMProvider | None = None,
        seed: int = 0,
        max_utterances: int = 12,
    ) -> None:
        super().__init__(knowledge, seed=seed)
        self._provider = provider
        self._max = max_utterances
        self._count = 0
        # Fallback keeps the same knowledge view => same isolation guarantees.
        self._fallback = HeuristicSimulator(knowledge, seed=seed)

    def _system_prompt(self) -> str:
        p = self.persona
        vis = self.knowledge.user_visible
        return (
            "Du simulierst einen echten Telefon-Anrufer. Bleibe in der Rolle.\n"
            f"Ziel: {self.goal.type} ({self.goal.model_dump()}).\n"
            f"Persona: Geduld={p.patience}, Ausführlichkeit={p.verbosity}, "
            f"Technik-Affinität={p.tech_savvy}, Sprachniveau={p.language_proficiency}.\n"
            f"Dein Wissen (nur das darfst du kennen): {vis}.\n"
            "Antworte kurz und natürlich, immer als Anrufer. Nenne konkrete "
            "Zeiten so, wie sie dir bekannt sind. Wenn dein Ziel erfüllt ist, "
            "beende das Gespräch höflich."
        )

    async def next_turn(
        self, history: list[dict[str, str]], last_bot_message: str | None
    ) -> SimulatorTurn:
        if self._provider is None:
            turn = await self._fallback.next_turn(history, last_bot_message)
            self.finished = self._fallback.finished
            return turn

        self._count += 1
        if self._count > self._max:
            self.finished = True
            return SimulatorTurn(text="Auf Wiederhören.", finished=True)

        # Caller's perspective: bot turns are "user" messages to the LLM.
        messages = [
            {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
            for m in history
        ]
        text = self._provider(self._system_prompt(), messages).strip()
        finished = any(
            token in text.lower() for token in ("wiederhören", "tschüss", "danke, das war alles")
        )
        self.finished = finished
        return SimulatorTurn(text=text, finished=finished)
