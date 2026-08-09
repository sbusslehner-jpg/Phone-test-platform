"""User simulation (concept sections 8-10).

The user simulator plays a realistic caller. It is kept behind an interface so
the platform is independent of any specific LLM or framework: a scripted
simulator (deterministic), a heuristic goal-driven simulator (LLM-shaped but
dependency-free) and pluggable LLM/DeepEval simulators all satisfy the same
:class:`UserSimulator` contract.

Crucially, a simulator only ever receives a
:class:`~phonebot_qa.scenario.knowledge.KnowledgeView` — never the evaluator-only
expectations (section 10).
"""

from __future__ import annotations

from .base import SimulatorTurn, UserSimulator
from .heuristic import HeuristicSimulator
from .scripted import ScriptedSimulator

__all__ = [
    "HeuristicSimulator",
    "ScriptedSimulator",
    "SimulatorTurn",
    "UserSimulator",
]
