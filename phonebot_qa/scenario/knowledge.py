"""Knowledge isolation (concept section 10).

The user simulator must only ever see what a real caller would plausibly know.
It must NOT see the expected tool calls, required events or backend
expectations — otherwise evaluation leaks into the simulated caller and every
test trivially "passes". :func:`isolate_user_knowledge` produces the *only*
view of a scenario that is allowed to reach a simulator.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..models import Persona, Scenario, UserGoal


class KnowledgeView(BaseModel):
    """The strictly user-visible projection of a scenario.

    This object is what the simulator receives. It deliberately excludes
    ``Scenario.expected`` (required/forbidden events, tool-call counts, expected
    DB state) and the raw ``initial_state`` — only ``user.user_visible`` and the
    caller's own goal/persona pass through.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    goal: UserGoal
    persona: Persona
    # Only what a caller would know coming into the conversation (section 10).
    user_visible: dict[str, Any]


def isolate_user_knowledge(scenario: Scenario, persona: Persona) -> KnowledgeView:
    """Build the leak-proof view handed to the user simulator.

    Any attempt to read evaluator-only data from this view is impossible —
    those fields simply aren't present. This is enforced structurally rather
    than by convention so a mis-written simulator cannot cheat.
    """
    return KnowledgeView(
        scenario_id=scenario.id,
        goal=scenario.user.goal,
        persona=persona,
        user_visible=dict(scenario.user.user_visible),
    )
