"""End-to-end conversation runner + evaluation pipeline."""

from __future__ import annotations

import pytest

from phonebot_qa.adapters.bot import ReferenceAppointmentBot
from phonebot_qa.evaluation import EvaluationPipeline
from phonebot_qa.runner import ConversationRunner
from phonebot_qa.scenario import isolate_user_knowledge
from phonebot_qa.simulator import HeuristicSimulator
from phonebot_qa.simulator.personas import resolve_persona


async def _run(scenario, *, bot=None, seed=0):
    persona = resolve_persona(scenario.user.persona)
    kv = isolate_user_knowledge(scenario, persona)
    sim = HeuristicSimulator(kv, seed=seed)
    bot = bot or ReferenceAppointmentBot(version="v-test")
    art = await ConversationRunner(bot).run(scenario=scenario, simulator=sim, seed=seed)
    return await EvaluationPipeline().evaluate(
        scenario, art, case_id="c", persona_id=scenario.user.persona, bot_version=bot.version
    )


async def test_happy_path_move_passes(move_scenario):
    result = await _run(move_scenario)
    assert result.result == "PASS"
    assert result.critical_failure is None
    assert result.score.business == 1.0
    assert result.assertion_summary()["db:appointments.a1.datetime"] is True


async def test_wrong_expected_state_forces_fail(move_scenario):
    # Corrupt the expectation: the bot books the real time, but we expect another.
    move_scenario.expected.database["appointments.a1"]["datetime"] = "1999-01-01T00:00:00+02:00"
    result = await _run(move_scenario)
    assert result.result == "FAIL"
    assert "db:appointments.a1.datetime" in result.critical_failure


async def test_forbidden_event_forces_fail(move_scenario):
    move_scenario.expected.forbidden_events.append("appointment_updated")
    result = await _run(move_scenario)
    assert result.result == "FAIL"


async def test_latency_metrics_present(move_scenario):
    result = await _run(move_scenario)
    assert result.latency.turns >= 1
    assert result.latency.avg_latency_ms > 0
    assert result.latency.p95_latency_ms > 0


async def test_critical_failure_overrides_high_soft_scores(move_scenario):
    # Even with a perfect conversation, a broken expectation must FAIL (section 27).
    move_scenario.expected.required_events.append("event_that_never_happens")
    result = await _run(move_scenario)
    assert result.result == "FAIL"
    # Soft conversation score can still be high; the verdict is still FAIL.
    assert result.eval_scores is not None
