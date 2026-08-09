"""The platform must catch known bug classes injected via bot behaviour flags."""

from __future__ import annotations

from phonebot_qa.adapters.bot.reference import BotBehavior, ReferenceAppointmentBot
from phonebot_qa.orchestrator.engine import run_suite
from phonebot_qa.scenario.loader import load_scenario
from tests.conftest import SCENARIOS_DIR


async def test_correction_bug_is_detected():
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_with_correction_001.yaml")
    bot = ReferenceAppointmentBot(version="buggy", behavior=BotBehavior(handle_corrections=False))
    summary = await run_suite([scenario], bot=bot)
    r = summary.results[0]
    assert r.result == "FAIL"
    assert "datetime" in (r.critical_failure or "")


async def test_false_success_on_fault_is_detected():
    scenario = load_scenario(SCENARIOS_DIR / "support" / "fault_calendar_timeout_001.yaml")
    bot = ReferenceAppointmentBot(version="buggy", behavior=BotBehavior(report_success_on_fault=True))
    summary = await run_suite([scenario], bot=bot)
    r = summary.results[0]
    assert r.result == "FAIL"
    assert "no_false_success" in (r.critical_failure or "")


async def test_missing_confirmation_is_detected():
    scenario = load_scenario(SCENARIOS_DIR / "cancellation" / "cancel_appointment_001.yaml")
    bot = ReferenceAppointmentBot(version="buggy", behavior=BotBehavior(confirm_before_write=False))
    summary = await run_suite([scenario], bot=bot)
    r = summary.results[0]
    assert r.result == "FAIL"


async def test_good_bot_passes_the_same_scenarios():
    scenarios = [
        load_scenario(SCENARIOS_DIR / "booking" / "move_with_correction_001.yaml"),
        load_scenario(SCENARIOS_DIR / "support" / "fault_calendar_timeout_001.yaml"),
        load_scenario(SCENARIOS_DIR / "cancellation" / "cancel_appointment_001.yaml"),
    ]
    summary = await run_suite(scenarios, bot=ReferenceAppointmentBot())
    assert summary.passed == summary.total
