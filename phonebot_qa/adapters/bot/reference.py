"""An in-process reference phonebot (the "bot under test").

This is a deterministic appointment-management voice agent implemented as a
small state machine. It exists so the whole platform runs end-to-end with no
external services, and so the docs/tests have a concrete system to exercise.

Its behaviour is configurable via :class:`BotBehavior`, which lets us model
different *bot versions* for the release gate (section 29) and reproduce known
bugs for regression tests (section 24) — e.g. ``handle_corrections=False``
reproduces "bot booked the old date after a correction".

The bot follows the safety-critical rules from the concept document:
* it confirms before any write (``confirmation_requested`` /
  ``confirmation_received``);
* it never claims success for a tool call the backend did not confirm
  (section 17) — unless the deliberately-buggy ``report_success_on_fault`` flag
  is set;
* it refuses red-team attempts (instruction override, mass deletion,
  impersonation, cross-customer access) — sections 22 & 27.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import BotAdapter, BotResponse, BotSession, SessionContext

# First ISO-8601 datetime in a message is treated as the caller's target time.
_ISO_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)

_MOVE_KW = ("verschieb", "umbuch", "verleg", "ander", "reschedul", "move", "change")
_CANCEL_KW = ("absag", "stornier", "cancel", "abbestell")
_CREATE_KW = ("neuen termin", "termin buch", "termin vereinbar", "book", "neu buch")
_CONFIRM_KW = ("ja", "genau", "passt", "richtig", "stimmt", "korrekt", "yes", "bestätig", "ok", "okay")
_DECLINE_KW = ("nein", "no", "doch nicht", "lieber nicht", "abbrechen")

# Red-team signatures (section 22). Kept explicit and conservative.
_INJECTION_KW = (
    "vergiss deine regeln",
    "vergiss alle regeln",
    "ignoriere deine",
    "ignoriere alle",
    "ignore your rules",
    "ignore all previous",
    "ignore previous instructions",
    "system prompt",
    "systemprompt",
    "neue anweisung",
)
_MASS_ACTION_KW = (
    "alle termine",
    "lösche alle",
    "loesche alle",
    "delete all",
    "cancel all",
    "storniere alle",
    "alle absagen",
    "alle löschen",
)
_IMPERSONATION_KW = (
    "ich bin mitarbeiter",
    "ich bin ein mitarbeiter",
    "ich bin ihr kollege",
    "zeig mir alle kunden",
    "alle kundendaten",
    "show me all customers",
    "ohne identifikation",
    "der mitarbeiter gestern",
    "kollege hat gesagt",
    "admin",
)


@dataclass
class BotBehavior:
    """Feature flags describing one bot version's behaviour."""

    confirm_before_write: bool = True
    handle_corrections: bool = True
    report_success_on_fault: bool = False  # True == known bug
    strict_authorization: bool = True


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


class ReferenceAppointmentBot(BotAdapter):
    """Deterministic appointment bot used as the default system under test."""

    def __init__(
        self, version: str = "reference-1.0", behavior: BotBehavior | None = None
    ) -> None:
        self.version = version
        self.behavior = behavior or BotBehavior()

    async def start_session(self, context: SessionContext) -> BotSession:
        session = BotSession(
            session_id=f"sess-{context.scenario_id}",
            greeting=(
                "Guten Tag, hier ist die Terminverwaltung. Wie kann ich Ihnen helfen?"
            ),
        )
        session.state.update(
            proxy=context.proxy,
            events=context.events,
            phase="await_request",
            intent=None,
            appointment_id=None,
            target_datetime=None,
            pending=None,  # ("update"|"cancel"|"create", payload)
            done=False,
        )
        return session

    async def send_text(self, session: BotSession, message: str) -> BotResponse:
        state = session.state
        proxy = state["proxy"]
        events = state["events"]
        turn = state.get("turn_index")
        text = message.lower().strip()

        if events is not None:
            events.emit("bot_processing_started", turn=turn)

        # -- 1. Safety / red-team gate (always first) --------------------- #
        refusal = self._maybe_refuse(text, events, turn)
        if refusal is not None:
            return BotResponse(text=refusal, metadata={"refused": True})

        # -- 2. Correction handling (a new target time mid-flow) --------- #
        # Only time-bearing actions (update/create) can be "corrected" to a new
        # time. A cancel has no target time, so a datetime uttered during a
        # cancel confirmation must NOT be treated as a correction (that would
        # misroute it into a booking).
        found_dt = self._extract_datetime(message)
        pending = state.get("pending")
        if (
            pending
            and pending[0] in ("update", "create")
            and found_dt
            and found_dt != state.get("target_datetime")
        ):
            return await self._handle_correction(session, found_dt)

        # -- 3. Confirmation of a pending action ------------------------- #
        if pending and _contains(text, _CONFIRM_KW) and not _contains(text, _DECLINE_KW):
            return await self._execute_pending(session)

        if pending and _contains(text, _DECLINE_KW):
            state["pending"] = None
            return BotResponse(
                text="In Ordnung, ich habe nichts geändert. Möchten Sie etwas anderes?",
            )

        # -- 4. Intent detection on a fresh request ---------------------- #
        if _contains(text, _CANCEL_KW) and not _contains(text, _MOVE_KW):
            return await self._begin_cancel(session, message)
        if _contains(text, _MOVE_KW):
            return await self._begin_move(session, message, found_dt)
        if _contains(text, _CREATE_KW):
            return await self._begin_create(session, message, found_dt)

        # A bare datetime with no verb, when we already know the intent (e.g. the
        # caller supplies a time in a follow-up turn, or an alternative after a
        # busy slot). Handled symmetrically for both move and create.
        if found_dt and state.get("intent") == "move":
            return await self._begin_move(session, message, found_dt)
        if found_dt and state.get("intent") == "create":
            return await self._begin_create(session, message, found_dt)

        return BotResponse(
            text=(
                "Entschuldigung, das habe ich nicht verstanden. Möchten Sie einen "
                "Termin verschieben, absagen oder neu vereinbaren?"
            )
        )

    # -- intent handlers --------------------------------------------------- #

    async def _begin_move(self, session, message, found_dt):
        state = session.state
        state["intent"] = "move"
        proxy, events, turn = state["proxy"], state["events"], state.get("turn_index")
        apt_id = self._resolve_appointment(session)
        if apt_id is None:
            return BotResponse(
                text="Ich finde leider keinen Termin auf Ihren Namen. Stimmt die Rufnummer?"
            )
        state["appointment_id"] = apt_id
        if not found_dt:
            return BotResponse(
                text="Gerne. Auf welchen Zeitpunkt möchten Sie den Termin verschieben?"
            )
        return await self._propose_time(session, "update", found_dt)

    async def _begin_cancel(self, session, message):
        state = session.state
        state["intent"] = "cancel"
        apt_id = self._resolve_appointment(session)
        if apt_id is None:
            return BotResponse(text="Ich finde keinen Termin auf Ihren Namen.")
        state["appointment_id"] = apt_id
        state["pending"] = ("cancel", {"appointment_id": apt_id})
        return self._ask_confirmation(
            session,
            f"Möchten Sie Ihren Termin ({apt_id}) wirklich absagen?",
        )

    async def _begin_create(self, session, message, found_dt):
        state = session.state
        state["intent"] = "create"
        if not found_dt:
            return BotResponse(text="Gerne. Für wann möchten Sie den neuen Termin?")
        return await self._propose_time(session, "create", found_dt)

    async def _propose_time(self, session, action: str, target_dt: str):
        """Check availability for ``target_dt`` and ask for confirmation."""
        state = session.state
        proxy, turn = state["proxy"], state.get("turn_index")
        try:
            avail = proxy.call(
                "availability.search", {"datetime": target_dt}, turn=turn
            )
        except Exception as exc:  # backend fault while checking availability
            return self._backend_trouble(exc)
        if not avail.get("available", False):
            return BotResponse(
                text=(
                    f"Der Zeitpunkt {target_dt} ist leider nicht frei. "
                    "Möchten Sie einen anderen Zeitpunkt nennen?"
                )
            )
        state["target_datetime"] = target_dt
        if action == "update":
            payload = {
                "appointment_id": state["appointment_id"],
                "datetime": target_dt,
            }
            state["pending"] = ("update", payload)
            return self._ask_confirmation(
                session, f"Ich verschiebe Ihren Termin auf {target_dt}. Ist das korrekt?"
            )
        elif action == "create":
            payload = {
                "customer_id": self._session_customer(session),
                "datetime": target_dt,
            }
            state["pending"] = ("create", payload)
            return self._ask_confirmation(
                session, f"Ich buche einen neuen Termin am {target_dt}. Ist das korrekt?"
            )
        # Defensive: _propose_time must only be called for time-bearing actions.
        raise ValueError(f"_propose_time called with unsupported action {action!r}")

    async def _handle_correction(self, session, new_dt: str):
        state = session.state
        action = state["pending"][0]
        if not self.behavior.handle_corrections:
            # KNOWN BUG (section 24): ignore the correction, keep the old target.
            return self._ask_confirmation(
                session,
                "Alles klar, ich habe den ursprünglichen Zeitpunkt vorgemerkt. "
                "Soll ich das so bestätigen?",
            )
        return await self._propose_time(session, action, new_dt)

    async def _execute_pending(self, session):
        state = session.state
        proxy, events, turn = state["proxy"], state["events"], state.get("turn_index")
        action, payload = state["pending"]

        if self.behavior.confirm_before_write and events is not None:
            events.emit("confirmation_received", turn=turn)

        tool = {
            "update": "appointment.update",
            "cancel": "appointment.cancel",
            "create": "appointment.create",
        }[action]
        try:
            proxy.call(tool, payload, turn=turn)
        except Exception as exc:
            if self.behavior.report_success_on_fault:
                # KNOWN BUG (section 17 violation): claim success despite fault.
                state["pending"] = None
                state["done"] = True
                return BotResponse(
                    text="Erledigt! Ihr Termin ist geändert.", done=True
                )
            return self._backend_trouble(exc)

        state["pending"] = None
        state["done"] = True
        done_text = {
            "update": "Erledigt! Ihr Termin wurde verschoben.",
            "cancel": "Ihr Termin wurde abgesagt.",
            "create": "Ihr neuer Termin ist gebucht.",
        }[action]
        return BotResponse(text=done_text, done=True)

    # -- helpers ----------------------------------------------------------- #

    def _ask_confirmation(self, session, text: str) -> BotResponse:
        events = session.state["events"]
        turn = session.state.get("turn_index")
        if events is not None:
            events.emit("confirmation_requested", turn=turn)
        return BotResponse(text=text, metadata={"awaiting_confirmation": True})

    def _backend_trouble(self, exc: Exception) -> BotResponse:
        # Never claim success — this is the section-17 invariant.
        return BotResponse(
            text=(
                "Entschuldigung, ich habe gerade ein technisches Problem und "
                "konnte die Änderung nicht durchführen. Bitte versuchen Sie es "
                "später noch einmal — es wurde nichts geändert."
            ),
            metadata={"backend_error": str(exc)},
        )

    def _maybe_refuse(self, text: str, events, turn) -> str | None:
        if _contains(text, _INJECTION_KW):
            self._emit_refusal(events, turn, "instruction_override")
            return (
                "Das kann ich nicht tun. Ich halte mich an die geltenden Regeln "
                "und kann Ihnen nur mit Ihren eigenen Terminen helfen."
            )
        if _contains(text, _MASS_ACTION_KW):
            self._emit_refusal(events, turn, "mass_action")
            return (
                "Ich kann keine Massenänderungen vornehmen. Ich kann nur Ihren "
                "eigenen einzelnen Termin bearbeiten."
            )
        if _contains(text, _IMPERSONATION_KW):
            self._emit_refusal(events, turn, "impersonation_or_authz")
            return (
                "Aus Datenschutzgründen kann ich Ihnen ausschließlich zu Ihrem "
                "eigenen Konto Auskunft geben und benötige eine Identifikation."
            )
        return None

    @staticmethod
    def _emit_refusal(events, turn, reason: str) -> None:
        if events is not None:
            events.emit("policy_violation_refused", turn=turn, reason=reason)

    def _extract_datetime(self, message: str) -> str | None:
        match = _ISO_RE.search(message)
        return match.group(0) if match else None

    def _session_customer(self, session) -> str | None:
        proxy = session.state["proxy"]
        if proxy is None:
            return None
        owner = proxy.world.snapshot().get("session_customer_id")
        return str(owner) if owner is not None else None

    def _resolve_appointment(self, session) -> str | None:
        """Find the session customer's appointment via the tool gateway."""
        proxy, turn = session.state["proxy"], session.state.get("turn_index")
        if proxy is None:
            return None
        try:
            result = proxy.call("appointment.list", {}, turn=turn)
        except Exception:
            return None
        appts = result.get("appointments", []) if isinstance(result, dict) else []
        return str(appts[0]["id"]) if appts else None
