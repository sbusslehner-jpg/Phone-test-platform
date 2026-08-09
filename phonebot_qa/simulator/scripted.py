"""A fully-scripted simulator (concept section 8).

Plays a fixed list of utterances regardless of what the bot says. This is the
most reproducible simulator and is ideal for tight unit tests and for red-team
scripts where the attack sequence must be exact (section 22). It finishes when
its script is exhausted.
"""

from __future__ import annotations

from .base import SimulatorTurn, UserSimulator


class ScriptedSimulator(UserSimulator):
    """Emit a predetermined sequence of user turns."""

    def __init__(self, knowledge, *, lines: list[str], seed: int = 0) -> None:
        super().__init__(knowledge, seed=seed)
        self._lines = list(lines)
        self._i = 0

    async def next_turn(
        self, history: list[dict[str, str]], last_bot_message: str | None
    ) -> SimulatorTurn:
        if self._i >= len(self._lines):
            self.finished = True
            return SimulatorTurn(text="", finished=True)
        line = self._lines[self._i]
        self._i += 1
        if self._i >= len(self._lines):
            self.finished = True
        return SimulatorTurn(text=line, finished=self.finished)
