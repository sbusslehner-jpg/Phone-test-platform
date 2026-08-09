"""Deterministic clock + structured event log.

Every component emits structured events (section 18). To keep runs perfectly
reproducible (same scenario + persona + seed => same trace), the platform does
**not** read wall-clock time during a run. Instead a logical :class:`Clock`
advances in milliseconds only when a component explicitly reports elapsed time
(e.g. the tool proxy adds a simulated tool latency). This makes latency
decomposition (section 18) exact and comparable across runs.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..models import Event


class Clock:
    """A monotonic logical clock measured in milliseconds from session start."""

    __slots__ = ("_now_ms",)

    def __init__(self) -> None:
        self._now_ms = 0

    @property
    def now_ms(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Advance the clock by ``delta_ms`` (>= 0) and return the new time."""
        if delta_ms < 0:
            raise ValueError("clock cannot move backwards")
        self._now_ms += delta_ms
        return self._now_ms


class EventLog:
    """Append-only, ordered store of structured events for one conversation.

    The event log is the *technical* half of the platform's source of truth
    (section 3): deterministic assertions query it for required/forbidden events
    and latency metrics.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self._events: list[Event] = []

    def emit(
        self,
        type: str,
        *,
        turn: int | None = None,
        **payload: object,
    ) -> Event:
        """Record an event stamped with the current logical time."""
        event = Event(
            type=str(type),
            t_ms=self.clock.now_ms,
            turn=turn,
            payload=dict(payload),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def names(self) -> list[str]:
        return [e.type for e in self._events]

    def has(self, event_type: str) -> bool:
        return any(e.type == event_type for e in self._events)

    def count(self, event_type: str) -> int:
        return sum(1 for e in self._events if e.type == event_type)

    def filter(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.type == event_type]

    def extend(self, events: Iterable[Event]) -> None:
        self._events.extend(events)

    def __len__(self) -> int:
        return len(self._events)
