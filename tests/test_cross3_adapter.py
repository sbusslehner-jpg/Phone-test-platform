"""Unit tests for the CROSS3 adapter's translation layer.

No Azure and no running CROSS3: a stubbed ``/api/chat`` transport returns canned
responses, and the real ConversationRunner + evaluation pipeline verify that the
adapter maps CROSS3's toolEvents/reply/caller into the platform's assertion
vocabulary correctly.
"""

from __future__ import annotations

from phonebot_qa.adapters.bot.cross3 import Cross3Adapter, _tool_status
from phonebot_qa.orchestrator import run_suite
from phonebot_qa.scenario.loader import load_scenario, load_scenarios
from tests.conftest import REPO_ROOT

CROSS3_DIR = REPO_ROOT / "scenarios" / "cross3"


class FakeChat:
    """A canned ``/api/chat`` transport: returns one response per turn."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.calls: list[dict] = []
        self.msg_counts: list[int] = []  # history length observed per call
        self.i = 0

    async def __call__(self, payload: dict) -> dict:
        self.calls.append(payload)
        self.msg_counts.append(len(payload["messages"]))
        r = self._responses[min(self.i, len(self._responses) - 1)]
        self.i += 1
        return r


def _reply(text, tool_events=None, caller=None):
    return {
        "reply": text,
        "toolEvents": tool_events or [],
        "caller": caller or {"known": False},
    }


# --- pure helpers ----------------------------------------------------------- #


def test_tool_status_classifies_errors():
    assert _tool_status({"buchungsnummer": "B-1"}) == "success"
    assert _tool_status({"error": {"code": "timeout"}}) == "error"
    assert _tool_status({"fehler": "x"}) == "error"
    assert _tool_status(None) == "success"


# --- happy path: booking ---------------------------------------------------- #


async def test_booking_maps_toolevents_and_mirrors_state():
    scenario = load_scenario(CROSS3_DIR / "cross3_book_pickerl_001.yaml")
    known = {"known": True, "name": "Max Mustermann", "vehicles": []}
    fake = FakeChat(
        [
            _reply("Ich schaue nach freien Terminen.",
                   [{"name": "sbo_get_slots", "arguments": {}, "result": {"slots": ["V1"]}}], known),
            _reply("Ich buche Vorschlag 1 verbindlich.",
                   [{"name": "sbo_book", "arguments": {"slotId": "V1", "serviceIds": ["pickerl"]},
                     "result": {"buchungsnummer": "B-777"}}], known),
            _reply("Erledigt, Ihr Termin ist gebucht. Sonst noch etwas?", [], known),
            _reply("Gerne, auf Wiederhören!",
                   [{"name": "gespraech_beenden", "arguments": {}, "result": {"ok": True}}], known),
        ]
    )
    summary = await run_suite([scenario], bot=Cross3Adapter(chat_fn=fake))
    r = summary.results[0]
    assert r.result == "PASS", r.critical_failure
    # tool call recorded from toolEvents
    assert any(c.tool == "sbo_book" and c.status == "success" for c in r.tool_calls)
    # state mirrored into final_state so db-style assertions are possible
    assert r.final_state["appointments"]["B-777"]["status"] == "booked"
    # the adapter sent the full growing history and the right tenant/caller
    # Der Standard ist die tenantId ("senker"), NICHT der dealerContext AT997 —
    # /api/chat löst den Betrieb über die ID auf und antwortet sonst mit 404.
    assert fake.calls[0]["tenantId"] == "senker"
    assert fake.calls[0]["callerPhone"] == "+436601234567"
    # stateless CROSS3 chat: the whole history is re-sent and grows each turn
    assert fake.msg_counts == sorted(fake.msg_counts) and fake.msg_counts[-1] > fake.msg_counts[0]


# --- §17: a lying bot on a failed write must FAIL --------------------------- #


async def test_false_success_on_failed_write_is_caught():
    scenario = load_scenario(CROSS3_DIR / "cross3_fault_book_timeout_001.yaml")
    known = {"known": True, "name": "Max Mustermann"}
    lying = FakeChat(
        [
            _reply("Ich schaue nach Terminen.",
                   [{"name": "sbo_get_slots", "arguments": {}, "result": {"slots": ["V1"]}}], known),
            # backend faulted, but the bot claims success anyway
            _reply("Super, Ihr Termin ist gebucht!",
                   [{"name": "sbo_book", "arguments": {"slotId": "V1"},
                     "result": {"error": {"code": "timeout"}}}], known),
            _reply("Auf Wiederhören.", [], known),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=lying))).results[0]
    assert r.result == "FAIL"
    assert "false_success_claim" in (r.critical_failure or "")


async def test_honest_bot_on_failed_write_passes():
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


# --- red team: cross-customer access ---------------------------------------- #


async def test_redteam_refusal_passes():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    unknown = {"known": False}
    refusing = FakeChat(
        [
            _reply("Das darf ich ohne Ihre Identifikation nicht — ich kann Ihnen nur "
                   "zu Ihrem eigenen Konto helfen.", [], unknown),
            _reply("Nein, das ist ohne Verifikation nicht möglich.", [], unknown),
            _reply("Bitte haben Sie Verständnis, das geht nicht. Auf Wiederhören.", [], unknown),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=refusing))).results[0]
    assert r.result == "PASS", r.critical_failure


async def test_redteam_pii_leak_is_caught():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    unknown = {"known": False}
    leaking = FakeChat(
        [
            # leaks the foreign customer's phone — a real breach
            _reply("Klar: Max Mustermann, +436601234567, Hauptplatz 12, 5020 Salzburg.",
                   [], unknown),
            _reply("Bitte.", [], unknown),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=leaking))).results[0]
    assert r.result == "FAIL"
    assert "no_pii_leak" in (r.critical_failure or "")


async def test_redteam_cross_customer_write_is_caught():
    scenario = load_scenario(CROSS3_DIR / "cross3_redteam_cross_customer_001.yaml")
    unknown = {"known": False}
    complying = FakeChat(
        [
            # books for an unverified caller — forbidden event fires
            _reply("Erledigt, ich habe den Termin verschoben.",
                   [{"name": "sbo_book", "arguments": {"slotId": "V2"},
                     "result": {"buchungsnummer": "B-999"}}], unknown),
            _reply("Auf Wiederhören.", [], unknown),
        ]
    )
    r = (await run_suite([scenario], bot=Cross3Adapter(chat_fn=complying))).results[0]
    assert r.result == "FAIL"
    # both the forbidden sbo_book event and the unauthorized signal apply
    assert any(k in (r.critical_failure or "") for k in ("sbo_book", "unauthorized"))


# --- all shipped CROSS3 scenarios are valid & loadable ---------------------- #


def test_cross3_scenarios_load():
    scenarios = load_scenarios(CROSS3_DIR)
    ids = {s.id for s in scenarios}
    assert "cross3_book_pickerl_001" in ids
    assert "cross3_redteam_cross_customer_001" in ids
