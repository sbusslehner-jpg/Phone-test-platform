"""Core models, scenario loading and knowledge isolation."""

from __future__ import annotations

import pytest

from phonebot_qa.models import Scenario
from phonebot_qa.scenario import (
    ScenarioValidationError,
    isolate_user_knowledge,
    load_scenarios,
)
from phonebot_qa.simulator.personas import resolve_persona
from tests.conftest import SCENARIOS_DIR


def test_scenario_rejects_unknown_fields():
    with pytest.raises(Exception):
        Scenario.model_validate(
            {"id": "x", "user": {"goal": {"type": "t"}}, "bogus_field": 1}
        )


def test_scenario_id_validation():
    with pytest.raises(Exception):
        Scenario.model_validate({"id": "bad id!", "user": {"goal": {"type": "t"}}})


def test_knowledge_isolation_hides_expectations(move_scenario):
    persona = resolve_persona(move_scenario.user.persona)
    view = isolate_user_knowledge(move_scenario, persona)
    # Only user-visible knowledge is present.
    assert view.user_visible == {"desired_time": "2026-08-14T10:00:00+02:00"}
    # Evaluator-only expectations are structurally absent.
    assert not hasattr(view, "expected")
    assert not hasattr(view, "initial_state")
    # The view is a frozen model: its fields cannot be reassigned.
    with pytest.raises(Exception):
        view.user_visible = {"leak": 1}  # type: ignore[misc]


def test_all_shipped_scenarios_load_and_are_unique():
    scenarios = load_scenarios(SCENARIOS_DIR)
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids))
    assert "move_appointment_001" in ids


def test_duplicate_scenario_ids_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "id: dup\nuser:\n  goal:\n    type: t\n", encoding="utf-8"
    )
    (tmp_path / "b.yaml").write_text(
        "id: dup\nuser:\n  goal:\n    type: t\n", encoding="utf-8"
    )
    with pytest.raises(ScenarioValidationError):
        load_scenarios(tmp_path)
