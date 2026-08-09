"""Client for CROSS3's phone-media relay protocol (concept §11/§12.2/§34).

CROSS3's telephone channel is a *full-duplex* media bridge:

    PSTN → Peoplefone → Asterisk + Relay ──WSS /ws/phone-media──▶ App ──▶ Azure Realtime

The Asterisk relay is deliberately dumb: it authenticates with ``x-relay-token``,
sends one ``{"type":"start", did, callerId, callUuid, format:"slin"}`` text frame,
then streams **binary SLIN audio** (8 kHz, 16-bit LE mono) in both directions.
Control text frames flow app→relay: ``{"type":"clear"}`` on barge-in (the app
cancelled its response because the caller spoke over it) and
``{"type":"hangup", reason}`` at end; the relay sends ``{"type":"stop"}`` when the
caller hangs up.

This client **is the relay** for testing purposes: it speaks that exact protocol,
so the platform can drive CROSS3's real voice bridge (media handling, Azure
Realtime STT/TTS, semantic-VAD turn detection, barge-in) without a PSTN carrier
or Asterisk. It is verified against a fake ``/ws/phone-media`` server
(``tests/test_cross3_phone_client.py``) — no CROSS3 and no Azure needed.

Honest scope (see docs/TESTING_CROSS3.md):
* The voice channel does **not** forward tool calls over the relay (tools run
  inside the bridge), so tool/backend assertions stay on the chat channel;
  voice tests cover STT/TTS/turn-taking/barge-in/hangup and — via a post-call
  read of the SBO mock — whether the booking actually landed.
* Scoring what CROSS3 *said* needs a real STT engine on the returned audio; the
  deterministic simulator STT only knows the platform's own generated audio.
* A full turn-by-turn scored run needs the platform's streaming voice path; this
  module delivers the verified protocol client + a native-voice ``BotAdapter``.

Requires the ``websockets`` package (``pip install 'phonebot-qa[voice]'``).
"""

from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass, field

from ...audio.buffer import AudioBuffer

#: SLIN telephony audio: 8 kHz, 16-bit LE mono; 20 ms == 320 bytes == 160 samples.
SLIN_SAMPLE_RATE = 8000
SLIN_FRAME_BYTES = 320


@dataclass
class BotTurn:
    """One collected stretch of bot audio plus the control signals around it."""

    audio: bytes = b""  # raw SLIN16 8 kHz
    barge_in: bool = False  # the app sent {"type":"clear"} (it detected an interrupt)
    ended: bool = False  # the call ended (hangup / stop / socket close)
    hangup_reason: str | None = None
    control: list[dict] = field(default_factory=list)  # every text frame seen

    @property
    def duration_ms(self) -> int:
        return int(1000 * (len(self.audio) // 2) / SLIN_SAMPLE_RATE)

    def to_buffer(self) -> AudioBuffer:
        return AudioBuffer.from_bytes(self.audio, sample_rate=SLIN_SAMPLE_RATE)


class Cross3VoicePhoneClient:
    """Speak CROSS3's ``/ws/phone-media`` relay protocol as a test caller.

    ``connect_fn`` lets tests inject an already-open websocket-like object
    (with async ``send`` / ``recv`` / ``close``); in production it is ``None`` and
    the client dials the real WSS endpoint.
    """

    def __init__(
        self,
        base_url: str = "ws://127.0.0.1:8080",
        *,
        relay_token: str = "",
        connect_fn=None,
    ) -> None:
        # /ws/phone-media lives on the same host as the app; accept http(s) too.
        self.base_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        self.relay_token = relay_token
        self._connect_fn = connect_fn
        self._ws = None
        self.ended = False

    @property
    def url(self) -> str:
        return f"{self.base_url}/ws/phone-media"

    async def connect(
        self, *, did: str, caller_id: str = "", call_uuid: str | None = None
    ) -> None:
        """Open the relay leg and send the mandatory ``start`` frame."""
        if self._connect_fn is not None:
            self._ws = await self._connect_fn()
        else:  # pragma: no cover - needs a running CROSS3
            try:
                import websockets
            except ImportError as exc:
                raise RuntimeError(
                    "voice client requires websockets: pip install 'phonebot-qa[voice]'"
                ) from exc
            self._ws = await websockets.connect(
                self.url, additional_headers={"x-relay-token": self.relay_token}
            )
        start = {
            "type": "start",
            "did": did,
            "callerId": caller_id,
            "callUuid": call_uuid or f"test-{_uuid.uuid4().hex[:12]}",
            "format": "slin",
        }
        await self._ws.send(json.dumps(start))

    async def send_audio(self, audio: bytes | AudioBuffer) -> None:
        """Stream caller audio to the bot as 20 ms SLIN binary frames."""
        if self._ws is None:
            raise RuntimeError("connect() first")
        pcm = audio.to_bytes() if isinstance(audio, AudioBuffer) else bytes(audio)
        for i in range(0, len(pcm), SLIN_FRAME_BYTES):
            await self._ws.send(pcm[i : i + SLIN_FRAME_BYTES])

    async def next_bot_turn(self, *, quiet_ms: int = 600, max_ms: int = 20000) -> BotTurn:
        """Collect bot audio until it goes quiet, or a control frame arrives.

        Full-duplex streams have no explicit "turn end", so a turn boundary is a
        gap of ``quiet_ms`` with no further audio (Azure's semantic-VAD pacing),
        a ``clear`` (barge-in), or a ``hangup``/``stop``/close. ``max_ms`` bounds
        the wait so a stuck stream cannot hang the test.
        """
        import asyncio

        turn = BotTurn()
        chunks: list[bytes] = []
        # Bytes of audio that == quiet_ms and max_ms, to bound collection.
        budget_recv_timeout = quiet_ms / 1000.0
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max_ms / 1000.0

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    self._ws.recv(), timeout=min(budget_recv_timeout, remaining)
                )
            except asyncio.TimeoutError:
                # Quiet gap: end of this bot turn (only if we already got audio;
                # otherwise keep waiting up to max_ms for the turn to start).
                if chunks:
                    break
                continue
            except Exception:
                turn.ended = True
                self.ended = True
                break

            if isinstance(msg, (bytes, bytearray)):
                chunks.append(bytes(msg))
                continue

            # Text control frame.
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            turn.control.append(data)
            kind = data.get("type")
            if kind == "clear":
                turn.barge_in = True
            elif kind in ("hangup", "stop"):
                turn.ended = True
                turn.hangup_reason = data.get("reason") or kind
                self.ended = True
                break

        turn.audio = b"".join(chunks)
        return turn

    async def send_stop(self) -> None:
        """Tell the app the caller hung up."""
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "stop"}))
            except Exception:  # pragma: no cover - socket already closed
                pass

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # pragma: no cover
                pass
            self._ws = None
