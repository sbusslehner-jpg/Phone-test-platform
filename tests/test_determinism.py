"""Reproducibility: a fixed seed must produce a byte-for-byte identical run."""

from __future__ import annotations

from phonebot_qa.orchestrator.engine import run_suite
from phonebot_qa.scenario.loader import load_scenarios
from tests.conftest import SCENARIOS_DIR


async def test_same_seed_is_identical():
    scenarios = load_scenarios(SCENARIOS_DIR / "booking")
    a = await run_suite(scenarios, seeds=[42])
    b = await run_suite(scenarios, seeds=[42])
    for ra, rb in zip(a.results, b.results):
        assert [t.bot for t in ra.conversation.turns] == [t.bot for t in rb.conversation.turns]
        assert [e.type for e in ra.events] == [e.type for e in rb.events]
        assert ra.score.total == rb.score.total


async def test_different_personas_still_pass():
    scenarios = load_scenarios(SCENARIOS_DIR / "booking")
    summary = await run_suite(
        scenarios, persona_ids=["normal", "impatient", "confused", "terse"], seeds=[1, 2]
    )
    assert summary.passed == summary.total, [f.critical_failure for f in summary.failures()]
