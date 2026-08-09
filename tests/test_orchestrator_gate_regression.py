"""Orchestrator generation, release gate, and the regression capture/replay loop."""

from __future__ import annotations

from phonebot_qa.adapters.bot.reference import BotBehavior, ReferenceAppointmentBot
from phonebot_qa.orchestrator import generate_cases, release_gate, run_suite
from phonebot_qa.regression import RegressionStore
from phonebot_qa.scenario.loader import load_scenario, load_scenarios
from tests.conftest import SCENARIOS_DIR


def test_generate_cases_cross_product():
    scenarios = load_scenarios(SCENARIOS_DIR / "booking")
    cases = generate_cases(
        scenarios, personas=["normal", "impatient"], seeds=[1, 2, 3], modes=["text"]
    )
    assert len(cases) == len(scenarios) * 2 * 3
    # Case ids are unique and reproducible.
    assert len({c.case_id for c in cases}) == len(cases)


def test_redteam_scenarios_ignore_persona_axis():
    from phonebot_qa.redteam import redteam_scenarios

    cases = generate_cases(redteam_scenarios(), personas=["normal", "impatient"], seeds=[1])
    # One case per red-team scenario despite two personas requested.
    assert len(cases) == len(redteam_scenarios())


async def test_release_gate_blocks_regression(tmp_path):
    scenarios = load_scenarios(SCENARIOS_DIR / "booking")
    baseline = await run_suite(scenarios, bot=ReferenceAppointmentBot(version="prod"))
    candidate = await run_suite(
        scenarios,
        bot=ReferenceAppointmentBot(version="cand", behavior=BotBehavior(handle_corrections=False)),
    )
    gate = release_gate(candidate, baseline)
    assert gate.decision == "BLOCK"
    assert gate.candidate.task_success < gate.baseline.task_success


async def test_release_gate_passes_equal_quality():
    scenarios = load_scenarios(SCENARIOS_DIR / "booking")
    baseline = await run_suite(scenarios, bot=ReferenceAppointmentBot(version="prod"))
    candidate = await run_suite(scenarios, bot=ReferenceAppointmentBot(version="cand"))
    gate = release_gate(candidate, baseline)
    assert gate.decision == "PASS"


async def test_regression_capture_and_replay(tmp_path):
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_with_correction_001.yaml")
    buggy = ReferenceAppointmentBot(version="buggy", behavior=BotBehavior(handle_corrections=False))

    # Capture the failure.
    summary = await run_suite([scenario], bot=buggy)
    failing = summary.failures()[0]
    store = RegressionStore(tmp_path)
    case = store.capture(failing, scenario, created_at="2026-08-09T00:00:00Z")
    assert (tmp_path / f"{case.id}.json").exists()

    # The stored case is self-contained and reproduces the same FAIL on the buggy bot.
    replayed = store.scenarios()
    assert len(replayed) == 1
    repro = await run_suite(replayed, bot=buggy)
    assert repro.results[0].result == "FAIL"

    # And it PASSES against the fixed bot.
    fixed = await run_suite(replayed, bot=ReferenceAppointmentBot(version="fixed"))
    assert fixed.results[0].result == "PASS"
