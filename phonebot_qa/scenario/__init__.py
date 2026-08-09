"""Scenario engine — the source of truth of the platform (concept section 5)."""

from __future__ import annotations

from .knowledge import KnowledgeView, isolate_user_knowledge
from .loader import (
    ScenarioValidationError,
    load_persona,
    load_personas,
    load_scenario,
    load_scenarios,
)

__all__ = [
    "KnowledgeView",
    "ScenarioValidationError",
    "isolate_user_knowledge",
    "load_persona",
    "load_personas",
    "load_scenario",
    "load_scenarios",
]
