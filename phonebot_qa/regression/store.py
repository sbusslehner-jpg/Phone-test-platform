"""File-based regression store (concept section 24).

A :class:`RegressionCase` freezes everything needed to replay a failure exactly:
a self-contained copy of the scenario, the persona, the seed, the fault config
and the reason it was captured. Storing the *scenario snapshot* (not a reference)
means the case still reproduces even after the original scenario file evolves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import CaseResult, Scenario


class RegressionCase(BaseModel):
    """A frozen, replayable failure (section 24)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    scenario: dict[str, Any]  # full serialized Scenario (self-contained)
    persona_id: str | None = None
    seed: int = 0
    mode: str = "text"
    bot_version: str = "unknown"
    reason: str = ""
    # Optional provenance; passed in by the caller (no wall-clock reads here).
    created_at: str | None = None
    source_case_id: str | None = None

    def to_scenario(self) -> Scenario:
        return Scenario.model_validate(self.scenario)


class RegressionStore:
    """Persist and load regression cases as JSON files under a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, case_id: str) -> Path:
        return self.root / f"{case_id}.json"

    def add_from_result(
        self,
        result: CaseResult,
        scenario: Scenario,
        *,
        reason: str | None = None,
        created_at: str | None = None,
    ) -> RegressionCase:
        """Build (but do not save) a regression case from a failed result."""
        case = RegressionCase(
            id=f"regr_{scenario.id}_{result.persona_id or 'default'}_{result.seed}",
            scenario=scenario.model_dump(mode="json"),
            persona_id=result.persona_id,
            seed=result.seed,
            mode=result.mode,
            bot_version=result.bot_version,
            reason=reason or (result.critical_failure or "captured failure"),
            created_at=created_at,
            source_case_id=result.case_id,
        )
        return case

    def save(self, case: RegressionCase) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(case.id)
        path.write_text(
            json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def capture(
        self,
        result: CaseResult,
        scenario: Scenario,
        *,
        reason: str | None = None,
        created_at: str | None = None,
    ) -> RegressionCase:
        """Build and immediately persist a regression case."""
        case = self.add_from_result(
            result, scenario, reason=reason, created_at=created_at
        )
        self.save(case)
        return case

    def load(self, case_id: str) -> RegressionCase:
        return RegressionCase.model_validate_json(self._path(case_id).read_text("utf-8"))

    def load_all(self) -> list[RegressionCase]:
        if not self.root.exists():
            return []
        cases = [
            RegressionCase.model_validate_json(p.read_text("utf-8"))
            for p in sorted(self.root.glob("*.json"))
        ]
        return cases

    def scenarios(self) -> list[Scenario]:
        """Reconstruct scenarios for the regression suite."""
        return [c.to_scenario() for c in self.load_all()]
