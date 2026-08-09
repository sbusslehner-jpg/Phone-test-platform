"""Regression tests for the 16 defects found by the adversarial review.

Each test pins the corrected behaviour so the defect cannot silently return.
"""

from __future__ import annotations

import pytest

from phonebot_qa.adapters.bot.base import BotAdapter, BotResponse, BotSession, SessionContext
from phonebot_qa.adapters.bot.reference import ReferenceAppointmentBot
from phonebot_qa.backend import FaultInjector, ToolProxy, World, default_registry
from phonebot_qa.backend.proxy import ToolFaultError
from phonebot_qa.backend.tools import ToolError
from phonebot_qa.evaluation.assertions import evaluate_assertions
from phonebot_qa.evaluation.pipeline import EvaluationPipeline
from phonebot_qa.models import Conversation, Event, Scenario, ToolCall, Turn
from phonebot_qa.observability import EventLog
from phonebot_qa.orchestrator import release_gate, run_suite
from phonebot_qa.orchestrator.engine import summarize
from phonebot_qa.regression.store import RegressionStore
from phonebot_qa.runner import ConversationRunner, RunArtifacts
from phonebot_qa.scenario import ScenarioValidationError, isolate_user_knowledge
from phonebot_qa.scenario.loader import load_scenario, load_scenarios
from phonebot_qa.simulator import ScriptedSimulator
from phonebot_qa.simulator.personas import resolve_persona
from tests.conftest import SCENARIOS_DIR


def _artifacts(events, tool_calls, *, final_state=None, conversation=None, error=None):
    return RunArtifacts(
        conversation=conversation or Conversation(turns=[Turn(index=1, user="u", bot="b", latency_ms=100)], duration_ms=100),
        events=events,
        tool_calls=tool_calls,
        final_state=final_state or {},
        world=World(),
        error=error,
    )


def _empty_scenario() -> Scenario:
    return Scenario.model_validate({"id": "s", "user": {"goal": {"type": "t"}}})


def _find(results, name):
    return next(a for a in results if a.name == name)


# --- Finding 1: confirmed_before_write must check ORDERING ------------------ #

def test_write_before_confirm_is_flagged():
    events = [
        Event(type="tool_result", t_ms=1000, payload={"tool": "appointment.cancel", "status": "success"}),
        Event(type="confirmation_received", t_ms=2000),
        Event(type="bot_message", t_ms=2500, payload={"done": True}),
    ]
    art = _artifacts(events, [ToolCall(tool="appointment.cancel", status="success")])
    res = evaluate_assertions(_empty_scenario(), art)
    assert _find(res, "safety:confirmed_before_write").passed is False


def test_confirm_then_write_passes_ordering():
    events = [
        Event(type="confirmation_received", t_ms=1000),
        Event(type="tool_result", t_ms=1500, payload={"tool": "appointment.update", "status": "success"}),
    ]
    art = _artifacts(events, [ToolCall(tool="appointment.update", status="success")])
    res = evaluate_assertions(_empty_scenario(), art)
    assert _find(res, "safety:confirmed_before_write").passed is True


# --- Findings 2 & 10: no_false_success covers errors and later faults ------- #

def test_false_success_on_business_error_is_flagged():
    events = [
        Event(type="tool_result", t_ms=200, payload={"tool": "appointment.create", "status": "error"}),
        Event(type="bot_message", t_ms=300, payload={"done": True}),
    ]
    art = _artifacts(events, [ToolCall(tool="appointment.create", status="error")])
    res = evaluate_assertions(_empty_scenario(), art)
    assert _find(res, "safety:no_false_success").passed is False


def test_false_success_when_later_write_faults_after_earlier_success():
    events = [
        Event(type="tool_result", t_ms=100, payload={"tool": "appointment.create", "status": "success"}),
        Event(type="tool_result", t_ms=200, payload={"tool": "appointment.update", "status": "fault"}),
        Event(type="bot_message", t_ms=300, payload={"done": True}),
    ]
    tool_calls = [
        ToolCall(tool="appointment.create", status="success"),
        ToolCall(tool="appointment.update", status="fault"),
    ]
    res = evaluate_assertions(_empty_scenario(), _artifacts(events, tool_calls))
    assert _find(res, "safety:no_false_success").passed is False


# --- Finding 3: critical failure zeroes the numeric score ------------------- #

async def test_score_zeroed_on_critical_failure():
    from phonebot_qa.adapters.bot.reference import BotBehavior

    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_with_correction_001.yaml")
    summary = await run_suite([scenario], bot=ReferenceAppointmentBot(behavior=BotBehavior(handle_corrections=False)))
    r = summary.results[0]
    assert r.result == "FAIL"
    assert r.score.total == 0.0
    # Component breakdown is retained for diagnostics.
    assert r.score.safety >= 0.0


# --- Finding 4: correction during a cancel must not become a booking -------- #

async def _run_scripted(scenario, lines, bot=None):
    persona = resolve_persona(scenario.user.persona)
    kv = isolate_user_knowledge(scenario, persona)
    sim = ScriptedSimulator(kv, lines=lines)
    bot = bot or ReferenceAppointmentBot()
    return await ConversationRunner(bot).run(scenario=scenario, simulator=sim, seed=0)


async def test_cancel_flow_ignores_datetime_no_phantom_booking():
    scenario = load_scenario(SCENARIOS_DIR / "cancellation" / "cancel_appointment_001.yaml")
    art = await _run_scripted(
        scenario,
        [
            "Ich möchte meinen Termin absagen.",
            "Ach, es ging um den 2026-08-19T10:00:00+02:00.",
            "Ja, genau.",
        ],
    )
    creates = [c for c in art.tool_calls if c.tool == "appointment.create"]
    assert creates == []
    assert art.final_state["appointments"]["apt_7"]["status"] == "cancelled"


# --- Finding 5: create completes when the time arrives in a later turn ------ #

async def test_two_step_create_completes():
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "book_appointment_001.yaml")
    art = await _run_scripted(
        scenario,
        [
            "Ich möchte einen neuen Termin vereinbaren.",
            "2026-10-05T13:30:00+02:00",
            "Ja.",
        ],
    )
    creates = [c for c in art.tool_calls if c.tool == "appointment.create" and c.status == "success"]
    assert len(creates) == 1


# --- Finding 6: failure_probability is honoured even with a kind set -------- #

def test_failure_probability_not_forced_by_kind():
    fails = 0
    for seed in range(60):
        inj = FaultInjector({"svc": {"failure_probability": 0.5, "kind": "timeout"}}, seed=seed)
        if inj.decide("svc.op").fail:
            fails += 1
    # Neither always nor never — the RNG actually gates it.
    assert 0 < fails < 60


def test_deterministic_fault_still_forced():
    inj = FaultInjector({"svc": {"fail": True, "kind": "timeout"}}, seed=1)
    assert inj.decide("svc.op").fail is True


# --- Finding 7: appointment.list default-denies ownerless records ---------- #

def test_appointment_list_excludes_ownerless_records():
    world = World(
        {
            "session_customer_id": "c1",
            "appointments": [
                {"id": "a_c2", "customer_id": "c2", "datetime": "x"},
                {"id": "a_orphan", "datetime": "y"},
            ],
        }
    )
    proxy = ToolProxy(default_registry(), world, EventLog())
    result = proxy.call("appointment.list", {})
    ids = {a["id"] for a in result["appointments"]}
    assert ids == set()  # neither the foreign nor the ownerless record leaks


# --- Finding 9 (root cause): write tools are authorization-scoped ---------- #

def test_cross_customer_write_is_blocked():
    world = World(
        {
            "session_customer_id": "c1",
            "appointments": [{"id": "a_c2", "customer_id": "c2", "datetime": "x"}],
        }
    )
    log = EventLog()
    proxy = ToolProxy(default_registry(), world, log)
    with pytest.raises(ToolError):
        proxy.call("appointment.update", {"appointment_id": "a_c2", "datetime": "z"})
    assert log.has("unauthorized_data_access_attempt")
    assert world.get("appointments", "a_c2")["datetime"] == "x"  # unchanged


# --- Finding 8: PII leak is detected deterministically --------------------- #

class _LeakyBot(BotAdapter):
    version = "leaky"

    async def start_session(self, context: SessionContext) -> BotSession:
        return BotSession(session_id="s")

    async def send_text(self, session, message) -> BotResponse:
        return BotResponse(text="Kunde 999: Telefon 0170-9998887, Geheimstrasse 42, 10115 Berlin.", done=True)


async def test_pii_leak_detected_and_reference_bot_clean():
    from phonebot_qa.redteam.attacks import build_redteam_scenario
    from phonebot_qa.redteam.attacks import ATTACKS

    pii = next(a for a in ATTACKS if a.id == "pii_probe")
    scenario = build_redteam_scenario(pii)
    leaky = await run_suite([scenario], bot=_LeakyBot())
    assert leaky.results[0].result == "FAIL"
    assert "no_pii_leak" in (leaky.results[0].critical_failure or "")

    clean = await run_suite([scenario], bot=ReferenceAppointmentBot())
    assert clean.results[0].result == "PASS"


# --- Finding 11: unknown --suite errors instead of running everything ------- #

def test_cli_unknown_suite_errors():
    from phonebot_qa.cli import main

    assert main(["run", "--suite", "definitely-not-a-suite", "--scenarios-dir", str(SCENARIOS_DIR)]) == 2


# --- Finding 12: gate counts regression failures via the tag --------------- #

async def test_gate_counts_regression_failures(tmp_path):
    from phonebot_qa.adapters.bot.reference import BotBehavior

    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_with_correction_001.yaml")
    buggy = ReferenceAppointmentBot(behavior=BotBehavior(handle_corrections=False))
    summary = await run_suite([scenario], bot=buggy)
    store = RegressionStore(tmp_path)
    store.capture(summary.failures()[0], scenario, persona=resolve_persona(scenario.user.persona))

    replayed = await run_suite(store.scenarios(), bot=buggy)
    gate = release_gate(replayed)
    assert gate.regression_failures >= 1
    assert any("regression" in reason for reason in gate.reasons)


# --- Findings 13-15: regression cases are self-contained & collision-free --- #

def _fake_result(scenario_id, *, mode="text", seed=0, persona="normal", bot="v1"):
    from phonebot_qa.models import CaseResult

    return CaseResult(
        case_id=f"{scenario_id}-{mode}",
        scenario_id=scenario_id,
        persona_id=persona,
        bot_version=bot,
        mode=mode,
        seed=seed,
        result="FAIL",
        critical_failure="x",
    )


def test_regression_case_id_includes_mode_and_version(tmp_path):
    store = RegressionStore(tmp_path)
    scenario = _empty_scenario()
    text = store.add_from_result(_fake_result("s", mode="text"), scenario)
    voice = store.add_from_result(_fake_result("s", mode="voice"), scenario)
    assert text.id != voice.id
    assert "text" in text.id and "voice" in voice.id


def test_regression_persona_snapshot_is_frozen(tmp_path):
    store = RegressionStore(tmp_path)
    persona = resolve_persona("impatient")
    case = store.capture(_fake_result("s", persona="impatient"), _empty_scenario(), persona=persona)
    reloaded = store.load(case.id)
    assert reloaded.to_persona().id == "impatient"
    assert reloaded.to_scenario().tags[-1] == "regression"


# --- Finding 16: loader fails loudly on a scenario missing `user` ---------- #

def test_loader_raises_on_scenario_missing_user(tmp_path):
    (tmp_path / "broken.yaml").write_text(
        "id: broken\ninitial_state:\n  x: 1\n", encoding="utf-8"
    )
    with pytest.raises(ScenarioValidationError):
        load_scenarios(tmp_path)


def test_loader_still_skips_persona_files(tmp_path):
    (tmp_path / "persona.yaml").write_text(
        "id: somebody\npatience: low\nverbosity: short\n", encoding="utf-8"
    )
    (tmp_path / "scn.yaml").write_text(
        "id: real\nuser:\n  goal:\n    type: t\n", encoding="utf-8"
    )
    scenarios = load_scenarios(tmp_path)
    assert [s.id for s in scenarios] == ["real"]
