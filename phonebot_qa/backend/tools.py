"""Tool registry & default appointment-domain tools (concept sections 15-16).

Tools are the backend actions a phonebot can invoke. Each tool is a pure-ish
handler that reads/writes the :class:`World` and may emit domain events (e.g.
``availability_checked``, ``appointment_updated``) that scenarios assert on
(section 20). Every invocation flows through the :class:`~.proxy.ToolProxy`,
which adds ``tool_called`` / ``tool_result`` events, latency and fault injection.

The default registry implements a calendar + CRM domain rich enough to drive the
appointment-management scenarios in the concept document (sections 5 & 26).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..observability import EventLog
from .world import World


@dataclass
class ToolContext:
    """Everything a tool handler needs: world state + event sink + turn no."""

    world: World
    events: EventLog
    turn: int | None = None


ToolHandler = Callable[[ToolContext, dict[str, Any]], Any]


class ToolError(Exception):
    """Raised by a tool handler for a *business* error (bad args, not found).

    Distinct from an injected technical fault: a ``ToolError`` is a legitimate
    negative result the bot should handle, not backend flakiness.
    """


@dataclass
class Tool:
    """A registered backend tool."""

    name: str
    handler: ToolHandler
    # Simulated base latency for latency decomposition (section 18).
    base_latency_ms: int = 150
    # Event emitted on a successful, non-faulted call (section 5 required_events).
    success_event: str | None = None
    description: str = ""

    def __call__(self, ctx: ToolContext, arguments: dict[str, Any]) -> Any:
        return self.handler(ctx, arguments)


class ToolRegistry:
    """A name -> :class:`Tool` map."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def add(
        self,
        name: str,
        handler: ToolHandler,
        *,
        base_latency_ms: int = 150,
        success_event: str | None = None,
        description: str = "",
    ) -> Tool:
        return self.register(
            Tool(
                name=name,
                handler=handler,
                base_latency_ms=base_latency_ms,
                success_event=success_event,
                description=description,
            )
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


# --------------------------------------------------------------------------- #
# Default appointment / CRM domain tools                                       #
# --------------------------------------------------------------------------- #


def _require(arguments: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if arguments.get(k) in (None, "")]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")


def _authorize(ctx: ToolContext, customer_id: Any) -> None:
    """Ensure the given customer owns the session; else it is unauthorized.

    Applies to writes too, not just reads: a bot must not modify another
    customer's records (sections 22 & 27). No-op when the session is unscoped
    (no ``session_customer_id``) or the record has no owner to check.
    """
    session_owner = ctx.world.snapshot().get("session_customer_id")
    if session_owner is None or customer_id is None:
        return
    if str(customer_id) != str(session_owner):
        ctx.events.emit(
            "unauthorized_data_access_attempt",
            turn=ctx.turn,
            requested_customer=str(customer_id),
            session_customer=str(session_owner),
        )
        raise ToolError("unauthorized: record does not belong to this session")


def _availability_search(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Check whether a requested time is available.

    In the mock backend every requested slot is available unless the world
    explicitly marks it busy via a ``busy_slots`` scalar. Emits
    ``availability_checked`` (section 5).
    """
    requested = args.get("datetime") or args.get("date")
    busy = set(ctx.world.snapshot().get("busy_slots", []) or [])
    available = requested not in busy
    ctx.events.emit("availability_checked", turn=ctx.turn, requested=requested)
    return {
        "requested": requested,
        "available": available,
        "alternatives": [] if available else ["propose_other_time"],
    }


def _appointment_update(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Move an existing appointment. Emits ``appointment_updated``."""
    _require(args, "appointment_id", "datetime")
    apt_id = str(args["appointment_id"])
    record = ctx.world.get("appointments", apt_id)
    if record is None:
        raise ToolError(f"appointment {apt_id!r} not found")
    _authorize(ctx, record.get("customer_id"))
    ctx.world.update("appointments", apt_id, datetime=args["datetime"])
    ctx.events.emit(
        "appointment_updated",
        turn=ctx.turn,
        appointment_id=apt_id,
        datetime=args["datetime"],
    )
    return {"status": "updated", "appointment_id": apt_id, "datetime": args["datetime"]}


def _appointment_create(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Create a new appointment. Emits ``appointment_created``."""
    _require(args, "customer_id", "datetime")
    _authorize(ctx, args.get("customer_id"))
    coll = ctx.world.collection("appointments")
    new_id = args.get("id") or f"apt_new_{len(coll) + 1}"
    record = {
        "id": str(new_id),
        "customer_id": str(args["customer_id"]),
        "datetime": args["datetime"],
    }
    ctx.world.put("appointments", record)
    ctx.events.emit(
        "appointment_created", turn=ctx.turn, appointment_id=str(new_id)
    )
    return {"status": "created", "appointment_id": str(new_id)}


def _appointment_cancel(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Cancel an existing appointment. Emits ``appointment_cancelled``."""
    _require(args, "appointment_id")
    apt_id = str(args["appointment_id"])
    record = ctx.world.get("appointments", apt_id)
    if record is None:
        raise ToolError(f"appointment {apt_id!r} not found")
    _authorize(ctx, record.get("customer_id"))
    ctx.world.update("appointments", apt_id, status="cancelled")
    ctx.events.emit("appointment_cancelled", turn=ctx.turn, appointment_id=apt_id)
    return {"status": "cancelled", "appointment_id": apt_id}


def _appointment_list(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """List the session customer's active appointments. Emits ``appointments_listed``.

    Authorization-scoped: only the customer that owns the session (or the
    explicitly session-owned customer) is returned. Cancelled appointments are
    excluded.
    """
    snapshot = ctx.world.snapshot()
    session_owner = snapshot.get("session_customer_id")
    requested = args.get("customer_id")
    if requested is not None and session_owner is not None and str(requested) != str(session_owner):
        ctx.events.emit(
            "unauthorized_data_access_attempt",
            turn=ctx.turn,
            requested_customer=str(requested),
            session_customer=str(session_owner),
        )
        raise ToolError("unauthorized: cannot list another customer's appointments")
    owner = str(requested) if requested is not None else (
        str(session_owner) if session_owner is not None else None
    )
    appointments = []
    for record in ctx.world.collection("appointments").values():
        if record.get("status") == "cancelled":
            continue
        if owner is not None:
            # Default-deny: a record whose owner cannot be established (no
            # customer_id) is NOT attributed to the caller.
            record_owner = record.get("customer_id")
            if record_owner is None or str(record_owner) != owner:
                continue
        appointments.append(dict(record))
    appointments.sort(key=lambda r: str(r.get("id")))
    ctx.events.emit("appointments_listed", turn=ctx.turn, count=len(appointments))
    return {"appointments": appointments}


def _customer_lookup(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Look up a customer record. Emits ``customer_looked_up``.

    Enforces a basic authorization invariant: a lookup must reference the
    customer that owns the session (``session_customer_id`` scalar), if present.
    Cross-customer access is a safety violation the red-team suite targets
    (sections 22 & 27).
    """
    _require(args, "customer_id")
    cust_id = str(args["customer_id"])
    session_owner = ctx.world.snapshot().get("session_customer_id")
    if session_owner is not None and str(session_owner) != cust_id:
        ctx.events.emit(
            "unauthorized_data_access_attempt",
            turn=ctx.turn,
            requested_customer=cust_id,
            session_customer=str(session_owner),
        )
        raise ToolError("unauthorized: customer does not own this session")
    record = ctx.world.get("customers", cust_id)
    if record is None:
        raise ToolError(f"customer {cust_id!r} not found")
    ctx.events.emit("customer_looked_up", turn=ctx.turn, customer_id=cust_id)
    return dict(record)


def default_registry() -> ToolRegistry:
    """Build a fresh registry with the default appointment/CRM tools."""
    reg = ToolRegistry()
    reg.add(
        "availability.search",
        _availability_search,
        base_latency_ms=200,
        description="Check whether a requested slot is free.",
    )
    reg.add(
        "appointment.update",
        _appointment_update,
        base_latency_ms=250,
        description="Move an existing appointment to a new time.",
    )
    reg.add(
        "appointment.create",
        _appointment_create,
        base_latency_ms=250,
        description="Create a new appointment.",
    )
    reg.add(
        "appointment.cancel",
        _appointment_cancel,
        base_latency_ms=200,
        description="Cancel an existing appointment.",
    )
    reg.add(
        "appointment.list",
        _appointment_list,
        base_latency_ms=150,
        description="List the session customer's active appointments.",
    )
    reg.add(
        "customer.lookup",
        _customer_lookup,
        base_latency_ms=150,
        description="Look up a customer record (authorization-checked).",
    )
    return reg
