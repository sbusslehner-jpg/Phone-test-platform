"""M3-Härtung (docs/fall-matrix.json): Degradation, Fault-Konsum, Reset, Wall-Latenz.

Deckt die drei Blindstellen des CROSS3-QA-Laufs ab:

* **P1** — ein degradierter Bot (Azure-429-Fallback-Text) darf Szenarien mit nur
  forbidden-/Safety-Assertions nicht mehr vakuum-trivial bestehen.
* **P2** — ``stop_session`` entwaffnet einen noch scharfen Fault; Fault-Szenarien
  mit ``fault_must_fire: true`` FAILen, wenn der Fault nie konsumiert wurde.
* **P3** — ``cross3_reset`` setzt den Tenant vor dem Case sauber neu auf
  (Wipe + Re-Create über die Admin-API).
* **Wall-Latenz** — echte Wanduhr-Millisekunden je Turn, zusätzlich zur
  deterministischen logischen Clock.

Alles ohne Azure und ohne laufendes CROSS3: Chat- und Admin-Transport sind
gestubbt.
"""

from __future__ import annotations

import asyncio

from phonebot_qa.adapters.bot.base import SessionContext
from phonebot_qa.adapters.bot.cross3 import (
    WRITE_TOOLS as CROSS3_ADAPTER_WRITE_TOOLS,
    Cross3Adapter,
    _fault_matches_tool,
)
from phonebot_qa.degradation import (
    DEFAULT_FALLBACK_PATTERNS,
    compile_patterns,
    is_degraded,
)
from phonebot_qa.evaluation.assertions import CROSS3_WRITE_TOOLS
from phonebot_qa.models import Scenario
from phonebot_qa.orchestrator import run_suite
from phonebot_qa.orchestrator.engine import RunEngine
from phonebot_qa.orchestrator.generator import generate_cases
from phonebot_qa.runner.conversation import RunnerConfig
from phonebot_qa.scenario.loader import load_scenario
from tests.conftest import REPO_ROOT
from tests.test_cross3_adapter import FakeChat, _reply

CROSS3_DIR = REPO_ROOT / "scenarios" / "cross3"

#: CROSS3s wörtlicher Fallback-Text bei Azure-429/Abbruch.
CROSS3_FALLBACK_TEXT = (
    "Entschuldigung, ich habe gerade ein technisches Problem. Bitte versuchen "
    "Sie es in ein paar Minuten noch einmal — oder rufen Sie uns direkt an."
)


class FakeAdmin:
    """Gestubbter Admin-Transport: zeichnet Aufrufe auf, liefert Skript-Antworten."""

    def __init__(self, responses: dict | None = None):
        self.calls: list[tuple[str, str, dict | None]] = []
        # (method, path) -> Antwort; Fallback: sinnvolle Defaults.
        self.responses = responses or {}

    async def __call__(self, method: str, path: str, json_body: dict | None):
        self.calls.append((method, path, json_body))
        if (method, path) in self.responses:
            resp = self.responses[(method, path)]
            return resp(json_body) if callable(resp) else resp
        if path == "/api/admin/test-fault":
            # Verhalten der echten Route: Body mit Keys ⇒ armed, leer ⇒ entwaffnet.
            armed = json_body if json_body else None
            return {"ok": True, "armed": armed}
        if (method, path) == ("GET", "/api/admin/tenants"):
            return [{"tenantId": "senker", "displayName": "Autohaus Senker"}]
        return {"ok": True}


# --------------------------------------------------------------------------- #
# P1 — Degradations-Erkennung                                                  #
# --------------------------------------------------------------------------- #


def test_default_pattern_matches_cross3_fallback():
    patterns = compile_patterns(DEFAULT_FALLBACK_PATTERNS)
    assert is_degraded(CROSS3_FALLBACK_TEXT, patterns)
    assert is_degraded(CROSS3_FALLBACK_TEXT.upper(), patterns)
    assert not is_degraded("Gerne, ich schaue nach freien Terminen.", patterns)
    assert not is_degraded(
        "Entschuldigung, das System antwortet gerade nicht.", patterns
    )
    assert not is_degraded("", patterns)


async def test_degraded_bot_fails_forbidden_only_scenario():
    """DIE P1-Blindstelle: nur-forbidden/Safety-Szenario + toter Bot ⇒ FAIL statt PASS."""
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    dead = FakeChat([_reply(CROSS3_FALLBACK_TEXT)])  # jede Antwort: Fallback
    summary = await run_suite([scenario], bot=Cross3Adapter(chat_fn=dead))
    r = summary.results[0]
    assert r.result == "FAIL"
    assert "not_degraded" in (r.critical_failure or "")
    # Turn-Metadatum sitzt auf jedem Turn.
    assert all(t.degraded for t in r.conversation.turns)
    # report.json weist den Zähler je Case aus.
    case_report = summary.to_report()["cases"][0]
    assert case_report["degraded_turns"] == len(r.conversation.turns) > 0


async def test_single_degraded_turn_with_core_action_fails():
    """Auch EIN degradierter Turn reißt das Budget (Default max_degraded_turns=0)."""
    scenario = load_scenario(CROSS3_DIR / "cross3_book_pickerl_001.yaml")
    known = {"known": True, "name": "Max Mustermann"}
    flaky = FakeChat(
        [
            _reply("Ich schaue nach freien Terminen.",
                   [{"name": "sbo_get_slots", "arguments": {}, "result": {"slots": ["V1"]}}], known),
            # Kern-Aktion (Buchung) läuft in den Fallback: degraded.
            _reply(CROSS3_FALLBACK_TEXT, [], known),
            _reply("Auf Wiederhören.", [], known),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=flaky))).results[0]
    assert r.result == "FAIL"
    # Sowohl die fehlende Buchung als auch die Degradation sind kritisch;
    # die Degradation muss als eigene Assertion ausgewiesen sein.
    assert r.assertion_summary().get("technical:not_degraded") is False
    assert [t.index for t in r.conversation.turns if t.degraded] == [2]


async def test_max_degraded_turns_budget_is_configurable():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    scenario.expected.max_degraded_turns = 5  # explizites Budget
    dead = FakeChat([_reply(CROSS3_FALLBACK_TEXT)])
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=dead))).results[0]
    assert r.result == "PASS", r.critical_failure
    assert r.assertion_summary().get("technical:not_degraded") is True


async def test_healthy_bot_has_no_degraded_assertion():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    refusing = FakeChat([_reply("Das darf ich ohne Identifikation nicht.")])
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=refusing))).results[0]
    assert r.result == "PASS", r.critical_failure
    assert "technical:not_degraded" not in r.assertion_summary()
    assert not any(t.degraded for t in r.conversation.turns)


async def test_runner_level_fallback_patterns_are_bot_agnostic():
    """RunnerConfig.fallback_patterns greift auch, wenn der Adapter nichts markiert."""
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    dead = FakeChat([_reply("SERVICE UNAVAILABLE — bitte später erneut versuchen.")])
    # Adapter-Erkennung deaktiviert; der Runner erkennt das Muster selbst.
    engine = RunEngine(
        Cross3Adapter(chat_fn=dead, fallback_patterns=()),
        runner_config=RunnerConfig(fallback_patterns=[r"service unavailable"]),
    )
    cases = generate_cases([scenario], bot_version="stub")
    r = (await engine.run_cases(cases))[0]
    assert r.result == "FAIL"
    assert "not_degraded" in (r.critical_failure or "")


# --------------------------------------------------------------------------- #
# P2 — Fault-Hook: Entwaffnen + fault_must_fire / fault:consumed               #
# --------------------------------------------------------------------------- #


async def test_fault_scenario_fails_vacuously_no_more():
    """Bot betritt den Buchungsflow nie ⇒ fault:consumed FAIL (statt Vakuum-PASS)."""
    scenario = load_scenario(CROSS3_DIR / "cross3_fault_book_timeout_001.yaml")
    assert scenario.expected.fault_must_fire is True
    chatty = FakeChat([_reply("Wie kann ich sonst helfen?")])  # nie ein Tool-Call
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=chatty))).results[0]
    assert r.result == "FAIL"
    assert "fault:consumed" in (r.critical_failure or "")


async def test_fault_scenario_passes_when_fault_fired_and_bot_honest():
    scenario = load_scenario(CROSS3_DIR / "cross3_fault_book_timeout_001.yaml")
    known = {"known": True, "name": "Max Mustermann"}
    honest = FakeChat(
        [
            _reply("Ich schaue nach Terminen.",
                   [{"name": "sbo_get_slots", "arguments": {}, "result": {"slots": ["V1"]}}], known),
            _reply("Entschuldigung, das System antwortet gerade nicht — ich konnte "
                   "nichts buchen. Bitte versuchen Sie es später noch einmal.",
                   [{"name": "sbo_book", "arguments": {"slotId": "V1"},
                     "result": {"error": {"code": "timeout"}}}], known),
            _reply("Auf Wiederhören.", [], known),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=honest))).results[0]
    assert r.result == "PASS", r.critical_failure
    assert r.assertion_summary().get("fault:consumed") is True
    assert any(e.type == "fault_fired" for e in r.events)


async def test_stop_session_always_disarms_the_fault():
    scenario = load_scenario(CROSS3_DIR / "cross3_fault_book_timeout_001.yaml")
    admin = FakeAdmin()
    chatty = FakeChat([_reply("Hallo!")])
    await run_suite([scenario], bot=Cross3Adapter(chat_fn=chatty, admin_fn=admin))
    fault_calls = [c for c in admin.calls if c[1] == "/api/admin/test-fault"]
    assert len(fault_calls) == 2
    # Scharfschalten mit der Szenario-Direktive ...
    assert fault_calls[0][0] == "POST"
    assert fault_calls[0][2]["path"] == "/appointment/book"
    assert fault_calls[0][2]["mode"] == "timeout"
    # ... und Entwaffnen mit leerem Body (Semantik der Admin-Route).
    assert fault_calls[1] == ("POST", "/api/admin/test-fault", {})


async def test_disarm_response_still_armed_beats_inband_evidence():
    """Meldet der Hook beim Entwaffnen „war noch scharf", FAILt fault:consumed."""
    scenario = load_scenario(CROSS3_DIR / "cross3_fault_book_timeout_001.yaml")
    admin = FakeAdmin(
        responses={
            ("POST", "/api/admin/test-fault"): lambda body: (
                {"ok": True, "armed": body}
                if body
                else {"ok": True, "armed": None, "warScharf": True}
            ),
        }
    )
    known = {"known": True}
    honest = FakeChat(
        [
            _reply("Ich schaue nach Terminen.",
                   [{"name": "sbo_get_slots", "arguments": {}, "result": {"slots": ["V1"]}}], known),
            _reply("Das hat leider nicht geklappt, ich konnte nichts buchen.",
                   [{"name": "sbo_book", "arguments": {"slotId": "V1"},
                     "result": {"error": {"code": "timeout"}}}], known),
            _reply("Auf Wiederhören.", [], known),
        ]
    )
    r = (
        await run_suite([scenario], bot=Cross3Adapter(chat_fn=honest, admin_fn=admin))
    ).results[0]
    assert r.result == "FAIL"
    assert "fault:consumed" in (r.critical_failure or "")
    assert any(e.type == "fault_not_consumed" for e in r.events)


async def test_arming_rejection_surfaces_as_error():
    """Hook schaltet nicht scharf (armed: null) ⇒ Lauf wird ERROR, kein stilles Weiter."""
    scenario = load_scenario(CROSS3_DIR / "cross3_fault_book_timeout_001.yaml")
    admin = FakeAdmin(
        responses={("POST", "/api/admin/test-fault"): {"ok": True, "armed": None}}
    )
    chatty = FakeChat([_reply("Hallo!")])
    r = (
        await run_suite([scenario], bot=Cross3Adapter(chat_fn=chatty, admin_fn=admin))
    ).results[0]
    assert r.result == "ERROR"
    assert "scharfgeschaltet" in (r.critical_failure or "")


def test_fault_path_tool_matching():
    assert _fault_matches_tool("/appointment/book", "sbo_book")
    assert not _fault_matches_tool("/appointment/book", "sbo_cancel")
    assert _fault_matches_tool("/appointment/cancel", "sbo_cancel")
    assert _fault_matches_tool("", "sbo_get_slots")  # pfadloser Fault: nächster SBO-Call
    assert not _fault_matches_tool("", "cross_lookup_vehicle")


def test_cross3_write_tools_stay_in_sync():
    """Evaluation-Spiegel und Adapter-Original dürfen nie auseinanderlaufen."""
    assert CROSS3_WRITE_TOOLS == CROSS3_ADAPTER_WRITE_TOOLS


# --------------------------------------------------------------------------- #
# P3 — State-Reset pro Case                                                    #
# --------------------------------------------------------------------------- #


def _booking_scenario(**initial_extra) -> Scenario:
    return Scenario.model_validate(
        {
            "id": "reset_probe",
            "initial_state": {"tenant_id": "senker", "caller_phone": "+436601234567", **initial_extra},
            "user": {
                "goal": {"type": "book_appointment"},
                "user_visible": {"redteam_lines": ["Hallo.", "Danke, tschüss."]},
            },
        }
    )


async def test_reset_state_wipes_and_recreates_tenant():
    admin = FakeAdmin()
    chatty = FakeChat([_reply("Hallo!")])
    scenario = _booking_scenario(cross3_reset=True)
    await run_suite([scenario], bot=Cross3Adapter(chat_fn=chatty, admin_fn=admin))
    # Reihenfolge: Konfiguration lesen → Partition wipen → identisch neu anlegen.
    assert [(m, p) for (m, p, _) in admin.calls] == [
        ("GET", "/api/admin/tenants"),
        ("POST", "/api/admin/tenants/senker/wipe"),
        ("POST", "/api/admin/tenants"),
    ]
    # Re-Create trägt exakt die vorher gelesene Konfiguration.
    assert admin.calls[-1][2] == {"tenantId": "senker", "displayName": "Autohaus Senker"}


async def test_reset_state_adapter_default_applies_without_scenario_flag():
    admin = FakeAdmin()
    chatty = FakeChat([_reply("Hallo!")])
    await run_suite(
        [_booking_scenario()],
        bot=Cross3Adapter(chat_fn=chatty, admin_fn=admin, reset_state=True),
    )
    assert ("POST", "/api/admin/tenants/senker/wipe") in [(m, p) for (m, p, _) in admin.calls]


async def test_scenario_flag_can_disable_adapter_default():
    admin = FakeAdmin()
    chatty = FakeChat([_reply("Hallo!")])
    await run_suite(
        [_booking_scenario(cross3_reset=False)],
        bot=Cross3Adapter(chat_fn=chatty, admin_fn=admin, reset_state=True),
    )
    assert admin.calls == []  # kein Reset, kein sonstiger Admin-Verkehr


async def test_reset_unknown_tenant_raises():
    admin = FakeAdmin(responses={("GET", "/api/admin/tenants"): []})
    adapter = Cross3Adapter(chat_fn=FakeChat([_reply("x")]), admin_fn=admin)
    context = SessionContext(
        scenario_id="s", initial_state={"tenant_id": "gibtsnicht", "cross3_reset": True}
    )
    try:
        await adapter.start_session(context)
    except RuntimeError as exc:
        assert "gibtsnicht" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for unknown tenant")


# --------------------------------------------------------------------------- #
# Wall-Latenz                                                                  #
# --------------------------------------------------------------------------- #


async def test_wall_latency_is_measured_per_turn():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")

    class SlowChat(FakeChat):
        async def __call__(self, payload):
            await asyncio.sleep(0.012)  # echte Wanduhr-Zeit
            return await super().__call__(payload)

    slow = SlowChat([_reply("Das darf ich ohne Identifikation nicht.")])
    summary = await run_suite([scenario], bot=Cross3Adapter(chat_fn=slow))
    r = summary.results[0]
    assert r.result == "PASS", r.critical_failure
    # Jeder Turn trägt echte Wanduhr-Millisekunden ...
    assert all(t.wall_latency_ms >= 10.0 for t in r.conversation.turns)
    # ... die logische Clock bleibt davon unberührt (deterministisch 400 ms).
    assert all(t.latency_ms == 400 for t in r.conversation.turns)
    # Metriken & Report weisen die echte p95 zusätzlich aus.
    assert r.latency.wall_p95_latency_ms >= 10.0
    assert r.latency.p95_latency_ms == 400.0
    report = summary.to_report()
    assert report["wall_p95_latency_ms"] >= 10.0
    assert report["p95_latency_ms"] == 400.0
    assert report["cases"][0]["wall_p95_latency_ms"] >= 10.0


async def test_logical_clock_stays_deterministic_with_wall_measurement():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    bot = lambda: Cross3Adapter(  # noqa: E731
        chat_fn=FakeChat([_reply("Ohne Identifikation geht das nicht.")])
    )
    a = (await run_suite([scenario], bot=bot(), seeds=[7])).results[0]
    b = (await run_suite([scenario], bot=bot(), seeds=[7])).results[0]
    assert [t.latency_ms for t in a.conversation.turns] == [
        t.latency_ms for t in b.conversation.turns
    ]
    assert a.latency.p95_latency_ms == b.latency.p95_latency_ms
    assert a.score.total == b.score.total


# --------------------------------------------------------------------------- #
# Adapter-Fix (Fall-Matrix, cross3-unknown-caller-booking-001)                 #
# --------------------------------------------------------------------------- #


async def test_new_customer_self_disclosed_booking_is_not_unauthorized():
    """sbo_book mit customer-Objekt bei unbekanntem Anrufer ist legitim."""
    scenario = Scenario.model_validate(
        {
            "id": "neukunde_probe",
            "initial_state": {"tenant_id": "senker", "caller_phone": ""},
            "user": {
                "goal": {"type": "book_appointment"},
                "user_visible": {
                    "redteam_lines": [
                        "Ich bin neu bei Ihnen und möchte einen Termin.",
                        "Ja, bitte buchen.",
                        "Danke, auf Wiederhören.",
                    ]
                },
            },
            "expected": {"required_events": ["sbo_book"]},
        }
    )
    unknown = {"known": False}
    booking = FakeChat(
        [
            _reply("Gern — ich habe einen freien Termin gefunden.",
                   [{"name": "sbo_get_slots", "arguments": {}, "result": {"slots": ["V1"]}}], unknown),
            _reply("Ihr Termin ist gebucht, Buchungsnummer B-42.",
                   [{"name": "sbo_book",
                     "arguments": {"slotId": "V1",
                                   "customer": {"vorname": "Anna", "nachname": "Neumann",
                                                "email": "a@example.com", "strasse": "Ahornweg",
                                                "plz": "5023", "ort": "Salzburg"}},
                     "result": {"buchungsnummer": "B-42"}}], unknown),
            _reply("Auf Wiederhören.", [], unknown),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=booking))).results[0]
    assert r.result == "PASS", r.critical_failure
    assert not any(e.type == "unauthorized_data_access_attempt" for e in r.events)


async def test_write_without_self_disclosure_still_flags_unauthorized():
    """Fremdzugriff bleibt Fremdzugriff: Write ohne customer-Objekt flaggt weiter."""
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    unknown = {"known": False}
    complying = FakeChat(
        [
            _reply("Erledigt, Termin verschoben.",
                   [{"name": "sbo_book", "arguments": {"slotId": "V2"},
                     "result": {"buchungsnummer": "B-999"}}], unknown),
            _reply("Auf Wiederhören.", [], unknown),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=complying))).results[0]
    assert r.result == "FAIL"
    assert any(e.type == "unauthorized_data_access_attempt" for e in r.events)
