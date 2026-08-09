"""Turn-based CROSS3 voice runner, verified against a fake /ws/phone-media server.

The fake speaks CROSS3's real relay protocol with VAD-like turn-taking (reply
after the caller goes quiet), plus per-``did`` behaviours for barge-in and
hangup. No CROSS3 and no Azure needed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("websockets")
import websockets  # noqa: E402

from phonebot_qa.adapters.transport.cross3_phone import (  # noqa: E402
    SLIN_FRAME_BYTES,
    Cross3VoicePhoneClient,
)
from phonebot_qa.audio.stt import STTEngine, STTResult  # noqa: E402
from phonebot_qa.models import Scenario  # noqa: E402
from phonebot_qa.runner.cross3_voice import run_cross3_voice_suite  # noqa: E402

TOKEN = "voice-token-abc"


def _frames(n):
    return b"\x10\x20" * (SLIN_FRAME_BYTES // 2) * n


async def _send_burst(ws, n=3):
    for _ in range(n):
        await ws.send(_frames(1))


async def _handler(ws):
    req = ws.request
    if req.path != "/ws/phone-media" or req.headers.get("x-relay-token") != TOKEN:
        await ws.close(4401, "unauthorized")
        return
    start = json.loads(await ws.recv())
    did = start.get("did", "")
    await _send_burst(ws)  # greeting (bot speaks first)

    # Some agents end the call the instant they finish greeting (no anliegen).
    if "greethangup" in did:
        await ws.send(json.dumps({"type": "hangup", "reason": "kein_anliegen"}))
        return

    caller_frames = 0
    replied = 0
    cleared = False
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.03)
        except asyncio.TimeoutError:
            if caller_frames > replied:  # caller went quiet → react (VAD-like)
                replied = caller_frames
                if "hangup" in did:
                    await ws.send(json.dumps({"type": "hangup", "reason": "agent_ende"}))
                    break
                if "transfer" in did:
                    await ws.send(json.dumps({"type": "transfer", "to": "+43512555000"}))
                    break
                await _send_burst(ws)
            continue
        except Exception:
            break
        if isinstance(msg, (bytes, bytearray)):
            caller_frames += 1
            # Barge-in: Azure semantic-VAD clears the bot's own audio on
            # speech_STARTED — the first frame the caller talks over the bot,
            # not after they go quiet. A small, deterministic reaction delay
            # keeps the measured stop latency non-zero and well within SLA.
            if "bargein" in did and not cleared:
                cleared = True
                await asyncio.sleep(0.02)
                await ws.send(json.dumps({"type": "clear"}))
        else:
            data = json.loads(msg)
            if data.get("type") == "stop":
                break


class _Server:
    async def __aenter__(self):
        self._server = await websockets.serve(_handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    def factory(self, token=TOKEN):
        port = self.port
        return lambda: Cross3VoicePhoneClient(f"ws://127.0.0.1:{port}", relay_token=token)


class _FakeSTT(STTEngine):
    latency_ms = 80

    def transcribe(self, audio) -> STTResult:
        return STTResult(text="verstanden, ich helfe gern", confidence=1.0)


def _scenario(did, *, audio=None, expected=None, lines=None):
    return Scenario.model_validate(
        {
            "id": f"voice_{did}",
            "tags": ["cross3", "voice"],
            "initial_state": {"tenant_id": "AT997", "did": did, "caller_phone": "+436601234567"},
            "audio": audio or {"profile": "clean"},
            "user": {
                "goal": {"type": "book_appointment"},
                "persona": "normal",
                "user_visible": {"redteam_lines": lines or ["Guten Tag, ich möchte einen Termin.", "Ja, passt."]},
            },
            "expected": expected or {},
            "limits": {"max_turns": 6, "max_duration_seconds": 300},
        }
    )


async def test_normal_voice_call_produces_scored_result():
    async with _Server() as srv:
        scn = _scenario("AT997-normal")
        summary = await run_cross3_voice_suite(
            [scn], srv.factory(), quiet_ms=120
        )
    r = summary.results[0]
    assert r.result == "PASS", r.critical_failure
    assert r.mode == "voice"
    assert r.latency.turns >= 1
    assert r.voice is not None  # voice metrics attached


async def test_stt_transcript_is_populated():
    async with _Server() as srv:
        scn = _scenario("AT997-normal")
        summary = await run_cross3_voice_suite(
            [scn], srv.factory(), stt=_FakeSTT(), quiet_ms=120
        )
    r = summary.results[0]
    assert any(t.bot == "verstanden, ich helfe gern" for t in r.conversation.turns)


async def test_barge_in_within_sla_passes():
    async with _Server() as srv:
        scn = _scenario(
            "AT997-bargein",
            audio={"profile": "clean", "barge_in": {"interrupt_after_ms": 30}, "barge_in_sla_ms": 300},
        )
        summary = await run_cross3_voice_suite([scn], srv.factory(), quiet_ms=120)
    r = summary.results[0]
    assert r.voice.barge_in_detected is True
    assert r.result == "PASS", r.critical_failure
    assert any(a.name == "voice:barge_in_sla" and a.passed for a in r.assertions)


async def test_barge_in_sla_breach_fails():
    async with _Server() as srv:
        # SLA of 0 ms: any real stop latency breaches it → critical FAIL.
        scn = _scenario(
            "AT997-bargein",
            audio={"profile": "clean", "barge_in": {"interrupt_after_ms": 30}, "barge_in_sla_ms": 0},
        )
        summary = await run_cross3_voice_suite([scn], srv.factory(), quiet_ms=120)
    r = summary.results[0]
    assert r.result == "FAIL"
    assert "barge_in_sla" in (r.critical_failure or "")


async def test_hangup_ends_the_call():
    async with _Server() as srv:
        scn = _scenario("AT997-hangup", lines=["Auf Wiederhören."])
        summary = await run_cross3_voice_suite([scn], srv.factory(), quiet_ms=120)
    r = summary.results[0]
    assert any(e.type == "hangup" for e in r.events)


async def test_state_reader_enables_backend_assertion():
    async def reader(_client):
        return {"appointments": [{"id": "A997-1", "status": "booked", "datetime": "2026-08-20T09:00:00+02:00"}]}

    async with _Server() as srv:
        scn = _scenario(
            "AT997-normal",
            expected={"database": {"appointments.A997-1": {"status": "booked"}}},
        )
        summary = await run_cross3_voice_suite(
            [scn], srv.factory(), state_reader=reader, quiet_ms=120
        )
    r = summary.results[0]
    assert r.result == "PASS", r.critical_failure
    assert r.final_state["appointments"]["A997-1"]["status"] == "booked"


async def test_transfer_ends_the_call():
    async with _Server() as srv:
        scn = _scenario("AT997-transfer", lines=["Ich möchte bitte einen Mitarbeiter sprechen."])
        summary = await run_cross3_voice_suite([scn], srv.factory(), quiet_ms=120)
    r = summary.results[0]
    # A hand-off to a human is a transfer event, not a spurious run error.
    assert any(e.type == "transfer" for e in r.events)
    assert not any(e.type == "run_error" for e in r.events)


async def test_greeting_hangup_is_not_a_run_error():
    async with _Server() as srv:
        scn = _scenario("AT997-greethangup", lines=["Hallo?"])
        summary = await run_cross3_voice_suite([scn], srv.factory(), quiet_ms=120)
    r = summary.results[0]
    # The agent hung up at the greeting: emit the end, skip the turn loop —
    # never surface it as a run error (which would force a false FAIL).
    assert any(e.type == "hangup" for e in r.events)
    assert not any(e.type == "run_error" for e in r.events)
    assert r.error is None


async def test_barge_in_that_never_fired_fails_as_not_run():
    async with _Server() as srv:
        # Greeting is far too short to still be talking after 5 s, so the probe
        # never fires. That must read as "the test did not run", not a false
        # "bot did not stop" — a distinct, honest, critical failure.
        scn = _scenario(
            "AT997-normal",
            audio={"profile": "clean", "barge_in": {"interrupt_after_ms": 5000}, "barge_in_sla_ms": 300},
        )
        summary = await run_cross3_voice_suite([scn], srv.factory(), quiet_ms=120)
    r = summary.results[0]
    assert r.result == "FAIL"
    assert "barge_in_attempted" in (r.critical_failure or "")
    assert r.voice.barge_in_detected is None
