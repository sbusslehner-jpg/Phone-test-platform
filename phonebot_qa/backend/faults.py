"""Fault injection for the tool proxy (concept section 17).

The test tool gateway can deliberately inject technical failures — timeouts,
HTTP 500/429, connection resets, malformed/partial responses, high latency,
service-unavailable — to test whether the bot degrades gracefully. The most
important invariant this enables (section 17): *the bot must never claim an
action succeeded when the backend did not confirm it.*

Fault decisions are **deterministic** given the case seed and the order of tool
calls, so a failing run can be replayed exactly (section 24, regression).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# Fault "kinds" mirror the failure list in section 17.
FAULT_KINDS = {
    "timeout",
    "http_500",
    "http_429",
    "connection_reset",
    "malformed_response",
    "partial_response",
    "service_unavailable",
}


@dataclass
class FaultResult:
    """The decision for a single tool call."""

    extra_latency_ms: int = 0
    fail: bool = False
    kind: str | None = None
    status: int | None = None
    message: str | None = None
    # When set, replaces the tool's real result with a corrupted payload.
    override_result: Any = None
    has_override: bool = field(default=False, repr=False)

    @property
    def injected(self) -> bool:
        return self.fail or self.extra_latency_ms > 0 or self.has_override


class FaultInjector:
    """Applies a scenario's ``faults`` config to individual tool calls.

    Config shape (service is matched against the tool name's first segment,
    e.g. tool ``calendar.update`` -> service ``calendar``)::

        faults:
          calendar:
            latency_ms: 4000
          crm:
            first_request:
              status: 500
          booking:
            failure_probability: 0.1
            kind: service_unavailable
    """

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = 0) -> None:
        self.config = config or {}
        self._rng = random.Random(seed)
        # Per-service call counter (drives ``first_request`` / ``every_n``).
        self._calls: dict[str, int] = {}

    @staticmethod
    def _service_of(tool: str) -> str:
        return tool.split(".", 1)[0]

    def decide(self, tool: str) -> FaultResult:
        """Decide whether/how to fault the given tool call."""
        service = self._service_of(tool)
        # Match either the exact tool name or its service prefix; exact wins.
        cfg = self.config.get(tool) or self.config.get(service)
        result = FaultResult()
        if not isinstance(cfg, dict):
            return result

        call_index = self._calls.get(service, 0)
        self._calls[service] = call_index + 1

        # ``first_request`` overrides apply only on the first call to a service.
        active = dict(cfg)
        first = cfg.get("first_request")
        if isinstance(first, dict):
            if call_index == 0:
                active = {**{k: v for k, v in cfg.items() if k != "first_request"}, **first}
            else:
                active = {k: v for k, v in cfg.items() if k != "first_request"}

        # Latency is always applied when configured.
        latency = active.get("latency_ms")
        if isinstance(latency, (int, float)) and latency > 0:
            result.extra_latency_ms = int(latency)

        # Probabilistic vs deterministic failure.
        prob = active.get("failure_probability")
        forced = bool(active.get("fail"))
        status = active.get("status")
        kind = active.get("kind")

        will_fail = forced or status is not None or kind is not None
        if isinstance(prob, (int, float)) and prob > 0:
            will_fail = will_fail or (self._rng.random() < float(prob))

        if will_fail:
            result.fail = True
            result.status = int(status) if status is not None else None
            result.kind = str(kind) if kind else self._kind_for_status(status)
            result.message = self._message_for(result.kind, result.status)

        # Malformed / partial responses corrupt the payload instead of failing.
        malformed = active.get("malformed_response")
        if malformed:
            result.has_override = True
            result.override_result = {"__malformed__": True}
            result.kind = result.kind or "malformed_response"

        return result

    @staticmethod
    def _kind_for_status(status: Any) -> str:
        mapping = {500: "http_500", 429: "http_429", 503: "service_unavailable"}
        return mapping.get(status, "service_unavailable") if status else "service_unavailable"

    @staticmethod
    def _message_for(kind: str | None, status: int | None) -> str:
        if status:
            return f"backend returned HTTP {status} ({kind})"
        return f"backend fault: {kind}"
