"""Red-team scenarios: the reference bot must refuse and cause no side effects."""

from __future__ import annotations

import pytest

from phonebot_qa.adapters.bot import ReferenceAppointmentBot
from phonebot_qa.orchestrator.engine import run_suite
from phonebot_qa.redteam import ATTACKS, redteam_scenarios


@pytest.mark.parametrize("attack_id", [a.id for a in ATTACKS])
async def test_each_redteam_attack_is_defended(attack_id):
    scenario = next(s for s in redteam_scenarios() if s.id == f"redteam_{attack_id}")
    summary = await run_suite([scenario], bot=ReferenceAppointmentBot())
    result = summary.results[0]
    assert result.result == "PASS", result.critical_failure


async def test_no_writes_happen_during_redteam():
    # A read (appointment.list) is fine; no *write* may ever succeed.
    write_tools = {"appointment.update", "appointment.create", "appointment.cancel"}
    summary = await run_suite(redteam_scenarios(), bot=ReferenceAppointmentBot())
    for r in summary.results:
        writes = [c for c in r.tool_calls if c.tool in write_tools and c.status == "success"]
        assert writes == [], f"{r.scenario_id} performed writes: {writes}"
