"""phonebot_qa — automated QA & red-team platform for phonebots / voice agents.

A CI/CD system for voice agents. The public surface re-exports the core domain
models and the high-level entry points so callers can do::

    from phonebot_qa import load_scenario, run_case

See ``README.md`` and ``docs/`` for the architecture (concept sections 2-37).
"""

from __future__ import annotations

from .models import (
    AssertionResult,
    CaseResult,
    Conversation,
    EvalScores,
    Event,
    EventType,
    Finding,
    Persona,
    Scenario,
    ScoreBreakdown,
    ToolCall,
    Turn,
    UserGoal,
    VoiceMetrics,
)
from .scenario.loader import load_scenario, load_scenarios

__all__ = [
    "AssertionResult",
    "CaseResult",
    "Conversation",
    "EvalScores",
    "Event",
    "EventType",
    "Finding",
    "Persona",
    "Scenario",
    "ScoreBreakdown",
    "ToolCall",
    "Turn",
    "UserGoal",
    "VoiceMetrics",
    "load_scenario",
    "load_scenarios",
]

__version__ = "0.1.0"
