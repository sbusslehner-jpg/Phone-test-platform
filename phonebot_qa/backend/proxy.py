"""Tool proxy — the test tool gateway (concept section 16).

Every tool call the bot makes flows through here. The proxy:

* records a structured :class:`~phonebot_qa.models.ToolCall` (for deterministic
  assertions — ``count_tool_calls`` etc., section 20);
* emits ``tool_called`` / ``tool_result`` events with timing (section 18);
* advances the logical clock by a simulated latency (base + injected);
* consults the :class:`~.faults.FaultInjector` and, on a fault, records the
  failure and raises :class:`ToolFaultError` so the bot must handle it — this is
  how we test the critical invariant that *the bot never reports success for an
  action the backend did not confirm* (section 17).
"""

from __future__ import annotations

from typing import Any

from ..models import ToolCall
from ..observability import EventLog
from .faults import FaultInjector
from .tools import ToolContext, ToolError, ToolRegistry
from .world import World


class ToolFaultError(Exception):
    """A technical backend fault surfaced to the bot (timeout, 500, ...)."""

    def __init__(self, message: str, *, kind: str | None = None, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


class ToolProxy:
    """Mediates all bot -> backend tool calls for one test case."""

    def __init__(
        self,
        registry: ToolRegistry,
        world: World,
        events: EventLog,
        faults: FaultInjector | None = None,
    ) -> None:
        self.registry = registry
        self.world = world
        self.events = events
        self.faults = faults or FaultInjector()
        self.calls: list[ToolCall] = []

    def call(
        self, tool_name: str, arguments: dict[str, Any] | None = None, *, turn: int | None = None
    ) -> Any:
        """Invoke ``tool_name`` through the gateway.

        Returns the tool result on success. Raises :class:`ToolFaultError` for
        injected technical faults and :class:`ToolError` for business errors.
        A :class:`ToolCall` record is appended in *all* cases.
        """
        arguments = dict(arguments or {})
        tool = self.registry.get(tool_name)
        self.events.emit("tool_called", turn=turn, tool=tool_name, arguments=arguments)

        if tool is None:
            record = ToolCall(
                tool=tool_name,
                arguments=arguments,
                status="error",
                turn=turn,
                error="unknown tool",
            )
            self.calls.append(record)
            self.events.emit("tool_result", turn=turn, tool=tool_name, status="error")
            raise ToolError(f"unknown tool {tool_name!r}")

        decision = self.faults.decide(tool_name)
        total_latency = tool.base_latency_ms + decision.extra_latency_ms
        self.events.clock.advance(total_latency)

        # -- injected technical fault ------------------------------------- #
        if decision.fail:
            self.events.emit(
                "tool_fault_injected",
                turn=turn,
                tool=tool_name,
                kind=decision.kind,
                status=decision.status,
            )
            record = ToolCall(
                tool=tool_name,
                arguments=arguments,
                status="fault",
                duration_ms=total_latency,
                turn=turn,
                error=decision.message,
            )
            self.calls.append(record)
            self.events.emit(
                "tool_result", turn=turn, tool=tool_name, status="fault"
            )
            raise ToolFaultError(
                decision.message or "backend fault",
                kind=decision.kind,
                status=decision.status,
            )

        # -- real handler ------------------------------------------------- #
        ctx = ToolContext(world=self.world, events=self.events, turn=turn)
        try:
            result = tool(ctx, arguments)
        except ToolError as exc:
            record = ToolCall(
                tool=tool_name,
                arguments=arguments,
                status="error",
                duration_ms=total_latency,
                turn=turn,
                error=str(exc),
            )
            self.calls.append(record)
            self.events.emit("tool_result", turn=turn, tool=tool_name, status="error")
            raise

        # Malformed/partial response override (still "succeeds" at transport
        # level but returns corrupted data — tests bot robustness, section 17).
        if decision.has_override:
            result = decision.override_result

        if tool.success_event:
            self.events.emit(tool.success_event, turn=turn)

        record = ToolCall(
            tool=tool_name,
            arguments=arguments,
            result=result,
            status="success",
            duration_ms=total_latency,
            turn=turn,
        )
        self.calls.append(record)
        self.events.emit("tool_result", turn=turn, tool=tool_name, status="success")
        return result

    def count(self, tool_name: str) -> int:
        return sum(1 for c in self.calls if c.tool == tool_name)
