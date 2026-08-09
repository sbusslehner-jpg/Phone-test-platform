"""Shared fixtures & helpers for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from phonebot_qa.models import Scenario

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "scenarios"


@pytest.fixture
def move_scenario() -> Scenario:
    return Scenario.model_validate(
        {
            "id": "move_test",
            "initial_state": {
                "session_customer_id": "c1",
                "customers": [{"id": "c1", "name": "Test"}],
                "appointments": [
                    {"id": "a1", "customer_id": "c1", "datetime": "2026-08-12T14:00:00+02:00"}
                ],
            },
            "user": {
                "goal": {"type": "move_appointment", "target_datetime": "2026-08-14T10:00:00+02:00"},
                "persona": "normal",
                "user_visible": {"desired_time": "2026-08-14T10:00:00+02:00"},
            },
            "expected": {
                "database": {"appointments.a1": {"datetime": "2026-08-14T10:00:00+02:00"}},
                "required_events": [
                    "availability_checked",
                    "confirmation_requested",
                    "confirmation_received",
                    "appointment_updated",
                ],
                "forbidden_events": ["appointment_created"],
                "tool_call_counts": {"appointment.update": 1, "appointment.create": 0},
            },
        }
    )
