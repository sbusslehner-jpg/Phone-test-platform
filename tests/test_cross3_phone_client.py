"""Tests for the CROSS3 /ws/phone-media relay-protocol client.

A fake ``/ws/phone-media`` server speaks CROSS3's real relay protocol (token
auth, start frame, binary SLIN audio, ``clear`` on barge-in, ``hangup``); the
actual client drives it over a real websocket. No CROSS3 and no Azure needed.
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

TOKEN = "test-relay-token-0123456789"


def _bot_frames(n: int) -> bytes:
    # n frames of 20 ms SLIN (arbitrary non-silent PCM).
    return b"\x11\x22" * (SLIN_FRAME_BYTES // 2) * n


async def _handler(ws):
    """Minimal, protocol-accurate fake of CROSS3's phone-media bridge."""
    req = ws.request
    if req.path != "/ws/phone-media" or req.headers.get("x-relay-token") != TOKEN:
        await ws.close(4401, "unauthorized")
        return
    start = json.loads(await ws.recv())
    assert start["type"] == "start"
    did = start.get("did", "")

    # The agent greets first (initialResponseEvent in the real bridge).
    for _ in range(5):
        await ws.send(_bot_frames(1))

    caller_turns = 0
    async for msg in ws:
        if isinstance(msg, (bytes, bytearray)):
            # A caller audio frame arrived. React once per burst boundary:
            # we treat each recv of caller audio as "the caller is speaking".
            continue
        data = json.loads(msg)
        if data.get("type") == "caller_done":  # test-only turn delimiter
            caller_turns += 1
            if "hangup" in did and caller_turns >= 1:
                await ws.send(json.dumps({"type": "hangup", "reason": "test_end"}))
                break
            if "bargein" in did:
                await ws.send(json.dumps({"type": "clear"}))
            for _ in range(4):
                await ws.send(_bot_frames(1))
        elif data.get("type") == "stop":
            await ws.send(json.dumps({"type": "hangup", "reason": "agent_ende"}))
            break


class _Server:
    def __init__(self):
        self._server = None
        self.port = None

    async def __aenter__(self):
        self._server = await websockets.serve(_handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    def client(self, token=TOKEN):
        return Cross3VoicePhoneClient(f"ws://127.0.0.1:{self.port}", relay_token=token)


async def _caller_turn(client, pcm: bytes):
    """Send a caller utterance and mark its end for the fake's turn logic."""
    await client.send_audio(pcm)
    await client._ws.send(json.dumps({"type": "caller_done"}))


async def test_greeting_and_reply_turns():
    async with _Server() as srv:
        client = srv.client()
        await client.connect(did="AT997-normal", caller_id="+436601234567")

        greeting = await client.next_bot_turn(quiet_ms=120)
        assert len(greeting.audio) > 0
        assert greeting.duration_ms > 0
        assert not greeting.ended

        await _caller_turn(client, _bot_frames(3))
        reply = await client.next_bot_turn(quiet_ms=120)
        assert len(reply.audio) > 0
        assert reply.barge_in is False
        await client.close()


async def test_start_frame_carries_did_and_caller():
    seen = {}

    async def capture(ws):
        seen_start = json.loads(await ws.recv())
        seen.update(seen_start)
        await ws.close()

    async with websockets.serve(capture, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Cross3VoicePhoneClient(f"ws://127.0.0.1:{port}", relay_token=TOKEN)
        await client.connect(did="AT997", caller_id="+436601234567", call_uuid="u-1")
        await client.close()

    assert seen["type"] == "start"
    assert seen["did"] == "AT997"
    assert seen["callerId"] == "+436601234567"
    assert seen["format"] == "slin"
    assert seen["callUuid"] == "u-1"


async def test_barge_in_is_surfaced():
    async with _Server() as srv:
        client = srv.client()
        await client.connect(did="AT997-bargein")
        await client.next_bot_turn(quiet_ms=120)  # greeting

        await _caller_turn(client, _bot_frames(3))
        turn = await client.next_bot_turn(quiet_ms=120)
        assert turn.barge_in is True
        assert any(c.get("type") == "clear" for c in turn.control)
        await client.close()


async def test_hangup_ends_the_call():
    async with _Server() as srv:
        client = srv.client()
        await client.connect(did="AT997-hangup")
        await client.next_bot_turn(quiet_ms=120)  # greeting

        await _caller_turn(client, _bot_frames(2))
        turn = await client.next_bot_turn(quiet_ms=120)
        assert turn.ended is True
        assert turn.hangup_reason == "test_end"
        assert client.ended is True
        await client.close()


async def test_transfer_ends_the_call():
    """A ``{"type":"transfer"}`` control frame ends the call, like a hangup."""

    async def handler(ws):
        await ws.recv()  # start
        await ws.send(json.dumps({"type": "transfer", "to": "+43512555000"}))
        await ws.close()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Cross3VoicePhoneClient(f"ws://127.0.0.1:{port}", relay_token=TOKEN)
        await client.connect(did="AT997")
        turn = await client.next_bot_turn(quiet_ms=120, max_ms=2000)
        assert turn.ended is True
        assert turn.transferred_to == "+43512555000"
        assert turn.hangup_reason == "transfer"
        assert client.ended is True
        await client.close()


async def test_barge_in_probe_paces_and_measures_stop_latency():
    """The probe streams the interrupt in real time and times the app's ``clear``.

    The stop latency is measured from the *first* interrupt frame (when the
    caller starts talking over the bot), and the caller audio lost equals that
    latency — the bot held the floor for exactly that long (§14).
    """

    async def handler(ws):
        await ws.recv()  # start
        for _ in range(20):  # a greeting long enough to talk over
            await ws.send(_bot_frames(1))
        cleared = False
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    if not cleared:  # clear on speech_started (first frame)
                        cleared = True
                        await asyncio.sleep(0.02)  # realistic reaction time
                        await ws.send(json.dumps({"type": "clear"}))
                elif json.loads(msg).get("type") == "stop":
                    break
        except Exception:
            pass

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Cross3VoicePhoneClient(f"ws://127.0.0.1:{port}", relay_token=TOKEN)
        await client.connect(did="AT997-bargein")
        turn = await client.bot_turn_with_barge_in(
            _bot_frames(20), interrupt_after_ms=20, quiet_ms=120
        )
        assert turn.interrupt_attempted is True
        assert turn.barge_in is True
        assert turn.stop_latency_ms is not None and turn.stop_latency_ms > 0
        assert turn.user_audio_lost_ms == turn.stop_latency_ms
        await client.close()


async def test_barge_in_probe_not_attempted_when_turn_too_short():
    """If the bot's turn is too short to reach ``interrupt_after_ms``, the probe
    never fires — ``interrupt_attempted`` stays False so the runner can report
    *the test did not run* instead of a false *bot did not stop*."""

    async def handler(ws):
        await ws.recv()  # start
        for _ in range(3):  # a very short greeting
            await ws.send(_bot_frames(1))
        try:
            async for msg in ws:
                if not isinstance(msg, (bytes, bytearray)) and json.loads(msg).get("type") == "stop":
                    break
        except Exception:
            pass

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Cross3VoicePhoneClient(f"ws://127.0.0.1:{port}", relay_token=TOKEN)
        await client.connect(did="AT997-bargein")
        turn = await client.bot_turn_with_barge_in(
            _bot_frames(5), interrupt_after_ms=5000, quiet_ms=100
        )
        assert turn.interrupt_attempted is False
        assert turn.barge_in is False
        assert turn.stop_latency_ms is None
        await client.close()


async def test_audio_is_chunked_into_slin_frames():
    sent = []

    async def capture(ws):
        await ws.recv()  # start
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                sent.append(len(msg))
            else:
                break

    async with websockets.serve(capture, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = Cross3VoicePhoneClient(f"ws://127.0.0.1:{port}", relay_token=TOKEN)
        await client.connect(did="AT997")
        # 3.5 frames worth of audio → 4 frames, last one short.
        await client.send_audio(b"\x01\x02" * (SLIN_FRAME_BYTES // 2 * 3 + 40))
        await client._ws.send(json.dumps({"type": "stop"}))
        await client.close()

    assert sent[:3] == [SLIN_FRAME_BYTES, SLIN_FRAME_BYTES, SLIN_FRAME_BYTES]
    assert 0 < sent[3] <= SLIN_FRAME_BYTES


async def test_bad_token_is_rejected():
    async with _Server() as srv:
        client = srv.client(token="wrong-token")
        try:
            await client.connect(did="AT997")
        except Exception:
            return  # rejected at/after handshake — acceptable
        # Or the connection was accepted then closed 4401 → no bot audio arrives.
        turn = await client.next_bot_turn(quiet_ms=120, max_ms=1000)
        assert turn.ended and not turn.audio
        await client.close()
