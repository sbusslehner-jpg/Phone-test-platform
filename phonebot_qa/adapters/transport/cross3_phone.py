"""Client for CROSS3's phone-media relay protocol (concept §11/§12.2/§34).

CROSS3's telephone channel is a *full-duplex* media bridge:

    PSTN → Peoplefone → Asterisk + Relay ──WSS /ws/phone-media──▶ App ──▶ Azure Realtime

The Asterisk relay is deliberately dumb: it authenticates with ``x-relay-token``,
sends one ``{"type":"start", did, callerId, callUuid, format:"slin"}`` text frame,
then streams **binary SLIN audio** (8 kHz, 16-bit LE mono) in both directions.
Control text frames flow app→relay: ``{"type":"clear"}`` on barge-in (the app
cancelled its response because the caller spoke over it), ``{"type":"hangup",
reason}`` when the agent ends the call, and ``{"type":"transfer", to}`` when it
hands off to a human — both end the call. The relay sends ``{"type":"stop"}``
when the caller hangs up.

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
SLIN_FRAME_MS = 20


@dataclass
class BotTurn:
    """One collected stretch of bot audio plus the control signals around it."""

    audio: bytes = b""  # raw SLIN16 8 kHz
    barge_in: bool = False  # the app sent {"type":"clear"} (it detected an interrupt)
    ended: bool = False  # the call ended (hangup / stop / transfer / socket close)
    hangup_reason: str | None = None
    transferred_to: str | None = None  # the app sent {"type":"transfer", to}
    control: list[dict] = field(default_factory=list)  # every text frame seen
    # Whether this turn actually fired a barge-in probe (streamed interrupt
    # audio over the bot). A plain turn leaves this False; only then is a missing
    # {"type":"clear"} a real "bot did not stop", not "we never interrupted".
    interrupt_attempted: bool = False
    # Barge-in timing (only set by a barge-in probe): how long after the caller
    # started talking over the bot the app acknowledged with {"type":"clear"}.
    stop_latency_ms: int | None = None
    # Bot audio that was still played AFTER the interrupt began — the caller's
    # words that landed while the bot talked over them (concept §14).
    user_audio_lost_ms: int | None = None
    # Response latency: ms from when we began listening to the bot's first audio
    # frame (caller-stopped-speaking → bot-started-speaking, concept §18).
    first_audio_latency_ms: int | None = None

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

    async def send_audio(
        self,
        audio: bytes | AudioBuffer,
        *,
        paced: bool = True,
        trailing_silence_ms: int = 800,
    ) -> None:
        """Stream caller audio to the bot the way a phone line does.

        Two properties of a real line matter, and getting either wrong makes the
        bot look broken when it is not:

        * **Pacing.** A carrier delivers 20 ms of audio every 20 ms. Bursting a
          three-second utterance into the socket in a few milliseconds means the
          server-side VAD sees the whole utterance at once and then *nothing* —
          no silence, no continuation. With ``semantic_vad`` it then waits its
          full window (up to 8 s on ``eagerness: low``) for audio that never
          comes.
        * **Trailing silence.** Silence on a phone line is still audio. When the
          client simply stops sending, endpointing has nothing to detect.

        Measured against CROSS3 on 2026-08-30: bursting with no trailing silence
        produced **one** model response — the greeting — in a 66-second call with
        three caller utterances. With pacing and trailing silence the same call
        produced four.

        Pass ``paced=False`` for the old burst behaviour (useful when a test
        deliberately floods the input buffer).
        """
        if self._ws is None:
            raise RuntimeError("connect() first")
        pcm = audio.to_bytes() if isinstance(audio, AudioBuffer) else bytes(audio)
        if trailing_silence_ms > 0:
            pcm += b"\x00" * (SLIN_SAMPLE_RATE * 2 * trailing_silence_ms // 1000)
        if paced:
            await self._send_paced(pcm)
            return
        for i in range(0, len(pcm), SLIN_FRAME_BYTES):
            await self._ws.send(pcm[i : i + SLIN_FRAME_BYTES])

    async def _send_paced(self, pcm: bytes, *, on_first=None) -> None:
        """Stream caller audio as *wall-clock-paced* 20 ms SLIN frames.

        This sleeps 20 ms between frames so the audio reaches the bot in real
        time — the same pacing a carrier applies. Barge-in stop latency then
        measures the app's actual reaction, not how fast a send buffer drains.
        :meth:`send_audio` delegates here unless a test asks for a burst.
        ``on_first`` fires just before the first frame goes out: the instant the
        caller starts talking over the bot. Cancel the task to stop early (the
        app acknowledged with ``clear``, so there is no need to keep talking).
        """
        import asyncio

        first = True
        for i in range(0, len(pcm), SLIN_FRAME_BYTES):
            if first:
                if on_first is not None:
                    on_first()
                first = False
            try:
                await self._ws.send(pcm[i : i + SLIN_FRAME_BYTES])
            except Exception:
                # Die Gegenseite hat aufgelegt oder umgelegt, WÄHREND der
                # Anrufer noch sprach. Am Telefon ist das ein normaler Ausgang
                # — genau der Fall "ich verbinde Sie" mitten im Satz —, kein
                # Fehler des Läufers. Aufhören zu senden und zurückkehren: den
                # transfer-/hangup-Rahmen liest ``next_bot_turn`` aus dem
                # Empfangspuffer, wo er schon liegt.
                #
                # Vor der Taktung fiel das nie auf: Die ganze Äusserung war in
                # Millisekunden im Socket, bevor die Gegenseite reagieren
                # konnte. Mit echtem Zeitverhalten dauert sie Sekunden, und das
                # Rennen ist real — der Weiterleitungs-Test flatterte in 2 von
                # 6 Läufen mit "ConnectionClosedOK" statt eines transfer.
                self.ended = True
                return
            await asyncio.sleep(SLIN_FRAME_MS / 1000.0)

    async def next_bot_turn(self, *, quiet_ms: int = 600, max_ms: int = 20000) -> BotTurn:
        """Collect bot audio until it goes quiet, or a control frame arrives.

        Full-duplex streams have no explicit "turn end", so a turn boundary is a
        gap of ``quiet_ms`` with no further audio (Azure's semantic-VAD pacing),
        a ``clear`` (barge-in), or a ``hangup``/``stop``/close. ``max_ms`` bounds
        the wait so a stuck stream cannot hang the test.

        **The gap is measured against PLAYBACK, not arrival.** The app pushes a
        finished sentence into the socket far faster than 8 kHz real time; a
        caller on a real line is still listening to it long after the last byte
        arrived. Ending the turn at "socket idle for 600 ms" therefore hands
        control back while the bot is, from the caller's side, still mid-
        sentence — and every scripted line then lands as a barge-in.

        Measured against CROSS3 on 2026-08-30: a four-turn call produced FOUR
        barge-ins and 24 seconds of discarded bot audio in 31 seconds of call.
        No booking could ever complete, because the caller talked over every
        question. Waiting out the playback fixes that without slowing down
        anything that really is silent.
        """
        import asyncio

        turn = BotTurn()
        chunks: list[bytes] = []
        # Bytes of audio that == quiet_ms and max_ms, to bound collection.
        budget_recv_timeout = quiet_ms / 1000.0
        loop = asyncio.get_event_loop()
        started_at = loop.time()
        deadline = started_at + max_ms / 1000.0

        def rest_der_wiedergabe() -> float:
            """Sekunden, die der Anrufer noch zuhoert (0, wenn er durch ist)."""
            if not chunks:
                return 0.0
            gespielt = loop.time() - (started_at + (turn.first_audio_latency_ms or 0) / 1000.0)
            gesendet = sum(len(c) for c in chunks) / (SLIN_SAMPLE_RATE * 2)
            return max(0.0, gesendet - gespielt)

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
                if not chunks:
                    continue
                rest = rest_der_wiedergabe()
                if rest <= 0:
                    break
                # Noch nicht: der Anrufer hoert die letzten Sekunden erst.
                budget_recv_timeout = min(rest, remaining)
                continue
            except Exception:
                turn.ended = True
                self.ended = True
                break

            if isinstance(msg, (bytes, bytearray)):
                if not chunks:
                    turn.first_audio_latency_ms = int((loop.time() - started_at) * 1000)
                chunks.append(bytes(msg))
                budget_recv_timeout = quiet_ms / 1000.0
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
            elif kind == "transfer":
                # Hand-off to a human ends this call leg just like a hangup.
                turn.ended = True
                turn.transferred_to = data.get("to")
                turn.hangup_reason = "transfer"
                self.ended = True
                break
            elif kind in ("hangup", "stop"):
                turn.ended = True
                turn.hangup_reason = data.get("reason") or kind
                self.ended = True
                break

        turn.audio = b"".join(chunks)
        return turn

    async def bot_turn_with_barge_in(
        self,
        interrupt_audio: bytes | AudioBuffer,
        *,
        interrupt_after_ms: int = 800,
        quiet_ms: int = 600,
        max_ms: int = 20000,
    ) -> BotTurn:
        """Collect a bot turn, talk over it mid-way, and measure the reaction.

        Waits until the bot has been speaking for ``interrupt_after_ms``, then
        streams the caller's interrupting audio *paced in real time* (20 ms per
        frame, in a background task) and records how long the app took to
        acknowledge with ``{"type":"clear"}`` (``stop_latency_ms``). Because the
        interrupt is paced — not burst — that latency is the app's genuine
        reaction from the moment the caller began talking, and ``clear`` cancels
        the remaining interrupt (there is no point talking once the bot stopped).
        ``user_audio_lost_ms`` equals the stop latency: the bot kept its floor
        for exactly that long after the caller started (concept §14).

        ``interrupt_attempted`` records whether the probe actually fired — if the
        bot's turn was too short to ever reach ``interrupt_after_ms``, it stays
        ``False`` so the runner can report *the test did not run* rather than a
        false "bot did not stop".
        """
        import asyncio

        turn = BotTurn()
        chunks: list[bytes] = []
        interrupt = (
            interrupt_audio.to_bytes()
            if isinstance(interrupt_audio, AudioBuffer)
            else bytes(interrupt_audio)
        )
        loop = asyncio.get_event_loop()
        start = loop.time()
        deadline = start + max_ms / 1000.0
        # Poll frequently so the interrupt fires on its own schedule, decoupled
        # from when the bot's audio frames happen to arrive.
        poll = min(0.01, quiet_ms / 1000.0)
        first_audio_at: float | None = None
        last_audio_at: float | None = None
        interrupt_at: float | None = None
        send_task: asyncio.Future | None = None

        def _mark_interrupt_start() -> None:
            # The caller's first interrupt frame is going out now.
            nonlocal interrupt_at
            interrupt_at = loop.time()

        try:
            while True:
                now = loop.time()
                if now >= deadline:
                    break
                # Fire the paced interrupt once the bot has been speaking long
                # enough. It runs concurrently with the receive loop below.
                if (
                    send_task is None
                    and first_audio_at is not None
                    and (now - first_audio_at) * 1000 >= interrupt_after_ms
                ):
                    turn.interrupt_attempted = True
                    send_task = asyncio.ensure_future(
                        self._send_paced(interrupt, on_first=_mark_interrupt_start)
                    )
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=poll)
                except asyncio.TimeoutError:
                    # End the turn once the bot has gone quiet for quiet_ms.
                    if (
                        chunks
                        and last_audio_at is not None
                        and (loop.time() - last_audio_at) * 1000 >= quiet_ms
                    ):
                        break
                    continue
                except Exception:
                    turn.ended = True
                    self.ended = True
                    break

                if isinstance(msg, (bytes, bytearray)):
                    now2 = loop.time()
                    if first_audio_at is None:
                        first_audio_at = now2
                        turn.first_audio_latency_ms = int((now2 - start) * 1000)
                    last_audio_at = now2
                    chunks.append(bytes(msg))
                    continue

                try:
                    data = json.loads(msg)
                except (ValueError, TypeError):
                    continue
                turn.control.append(data)
                kind = data.get("type")
                if kind == "clear":
                    turn.barge_in = True
                    if interrupt_at is not None:
                        turn.stop_latency_ms = int((loop.time() - interrupt_at) * 1000)
                        # The caller kept the bot talking over them for exactly
                        # the stop latency; that is the audio they lost (§14).
                        turn.user_audio_lost_ms = turn.stop_latency_ms
                    break
                if kind == "transfer":
                    turn.ended = True
                    turn.transferred_to = data.get("to")
                    turn.hangup_reason = "transfer"
                    self.ended = True
                    break
                if kind in ("hangup", "stop"):
                    turn.ended = True
                    turn.hangup_reason = data.get("reason") or kind
                    self.ended = True
                    break
        finally:
            # Stop talking over the bot: the probe is done (clear/hangup/quiet).
            if send_task is not None and not send_task.done():
                send_task.cancel()
                try:
                    await send_task
                except (asyncio.CancelledError, Exception):  # pragma: no cover
                    pass

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
