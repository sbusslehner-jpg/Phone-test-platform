"""Turn a production call trace into a runnable scenario (concept §35).

A production trace is whatever your telephony/agent stack already logs: the
transcript, the tool calls it made, the backend state before and after, and
(optionally) which invariant looked wrong. From that we synthesize:

* a :class:`Scenario` whose ``initial_state`` is the *pre-call* backend state
  and whose scripted caller replays exactly what the real caller said;
* expectations derived from the intended outcome — either supplied explicitly by
  the analyst, or inferred conservatively as "no unconfirmed write, no
  unauthorized access, no PII leak";
* a :class:`RegressionCase` freezing it all.

The generated scenario is a normal scenario: it runs in CI forever after.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import Scenario
from ..regression.store import RegressionCase

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.\-]+")


class ProductionTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user: str = ""
    bot: str = ""


class ProductionTrace(BaseModel):
    """A production call as exported by the telephony / agent stack."""

    model_config = ConfigDict(extra="ignore")

    call_id: str
    #: Backend state before the call — becomes the scenario's initial_state.
    initial_state: dict[str, Any] = Field(default_factory=dict)
    #: What the caller actually said, in order.
    turns: list[ProductionTurn] = Field(default_factory=list)
    #: Tools the production bot invoked (for the analyst's reference).
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    #: Backend state after the call (what actually happened — often the bug).
    final_state: dict[str, Any] = Field(default_factory=dict)
    #: What *should* have happened. When omitted we fall back to safety-only.
    expected_state: dict[str, dict[str, Any]] | None = None
    expected_events: list[str] = Field(default_factory=list)
    forbidden_events: list[str] = Field(default_factory=list)
    #: Analyst's note on why this call was flagged.
    reason: str = ""
    persona_id: str | None = None
    mode: str = "text"
    bot_version: str = "production"
    recorded_at: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ProductionTrace":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> "ProductionTrace":
        if isinstance(payload, str):
            payload = json.loads(payload)
        return cls.model_validate(payload)


def _scenario_id(call_id: str) -> str:
    return f"prod_{_SAFE_ID.sub('-', call_id)}"


def scenario_from_trace(trace: ProductionTrace) -> Scenario:
    """Synthesize a replayable scenario from a production call.

    The caller is scripted with the real utterances (so the exact wording that
    triggered the bug is preserved), and expectations default to the always-on
    safety contract when the analyst did not state an intended outcome.
    """
    lines = [t.user for t in trace.turns if t.user]
    safety = ["no_unauthorized_access", "no_pii_leak"]
    expected: dict[str, Any] = {
        "database": dict(trace.expected_state or {}),
        "required_events": list(trace.expected_events),
        "forbidden_events": list(trace.forbidden_events),
        "safety_invariants": safety,
    }
    return Scenario.model_validate(
        {
            "id": _scenario_id(trace.call_id),
            "description": (
                trace.reason or f"Aus Produktionsanruf {trace.call_id} erzeugtes Szenario."
            ),
            "tags": ["production", "regression"],
            "initial_state": dict(trace.initial_state),
            "user": {
                "goal": {"type": "production_replay", "call_id": trace.call_id},
                "persona": trace.persona_id,
                "user_visible": {"redteam_lines": lines},
            },
            "expected": expected,
            "limits": {"max_turns": max(4, len(lines) + 2)},
        }
    )


def regression_case_from_trace(
    trace: ProductionTrace, *, seed: int = 0
) -> RegressionCase:
    """Freeze a production failure directly as a regression case (section 24)."""
    scenario = scenario_from_trace(trace)
    return RegressionCase(
        id=f"regr_{scenario.id}_{trace.persona_id or 'default'}_{trace.mode}"
        f"_seed{seed}_{trace.bot_version}",
        scenario=scenario.model_dump(mode="json"),
        persona_id=trace.persona_id,
        seed=seed,
        mode=trace.mode,
        bot_version=trace.bot_version,
        reason=trace.reason or f"production call {trace.call_id}",
        created_at=trace.recorded_at,
        source_case_id=trace.call_id,
    )
