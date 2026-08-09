"""Test-case generation (concept section 6).

The orchestrator expands a suite request into the cross product

    Scenario × Persona × Seed × Mode × Bot Version

Each combination is one concrete, reproducible :class:`TestCase` with a stable,
human-readable ``case_id``. Reproducibility is the point: the ``seed`` fully
determines simulator/fault behaviour, so any case can be re-run identically
(section 24).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Scenario


@dataclass(frozen=True)
class TestCase:
    """One concrete, reproducible test case."""

    case_id: str
    scenario: Scenario
    persona_id: str | None
    seed: int
    mode: str = "text"
    bot_version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


def _slug(value: Any) -> str:
    return str(value).replace(" ", "-")


def generate_cases(
    scenarios: list[Scenario],
    *,
    personas: list[str | None] | None = None,
    seeds: list[int] | None = None,
    modes: list[str] | None = None,
    bot_version: str = "unknown",
) -> list[TestCase]:
    """Produce the full cross product of test cases.

    ``personas`` defaults to each scenario's own persona (or the platform
    default). ``seeds`` defaults to ``[0]`` and ``modes`` to ``["text"]``.
    Red-team scenarios ignore the persona axis (their caller is scripted), so
    they are emitted once per (seed × mode) to avoid meaningless duplication.
    """
    seeds = seeds or [0]
    modes = modes or ["text"]
    cases: list[TestCase] = []

    for scenario in scenarios:
        # Scripted callers (red-team attacks, discovery probes) ignore the
        # persona axis — their utterances are fixed, so fanning out over
        # personas would only duplicate identical conversations.
        is_redteam = "redteam" in scenario.tags or bool(
            scenario.user.user_visible.get("redteam_lines")
        )
        if personas is None:
            scenario_personas: list[str | None] = [scenario.user.persona]
        else:
            scenario_personas = list(personas)
        if is_redteam:
            scenario_personas = [scenario.user.persona]

        for persona_id in scenario_personas:
            for seed in seeds:
                for mode in modes:
                    pslug = _slug(persona_id or "default")
                    case_id = (
                        f"{scenario.id}__{pslug}__seed{seed}__{mode}__{_slug(bot_version)}"
                    )
                    cases.append(
                        TestCase(
                            case_id=case_id,
                            scenario=scenario,
                            persona_id=persona_id,
                            seed=seed,
                            mode=mode,
                            bot_version=bot_version,
                        )
                    )
    return cases
