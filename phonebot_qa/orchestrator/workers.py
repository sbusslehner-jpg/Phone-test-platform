"""Worker / queue abstraction (concept sections 6 & 31).

The concept's target topology is FastAPI + Redis + Celery/Dramatiq workers so a
suite of thousands of conversations (section 28) fans out across machines. The
platform keeps that behind a small :class:`CaseQueue` interface:

``InProcessQueue``
    The default. Runs cases concurrently with asyncio in the current process —
    no broker, no extra services, fully deterministic. This is what CI uses.

``CeleryQueue`` / ``DramatiqQueue``
    Thin adapters that submit each case to a real broker and collect the
    results. Both brokers are optional dependencies; constructing an adapter
    without its package raises a clear error rather than failing at import.

Because a test case is fully described by ``(scenario, persona, seed, mode,
bot_version)`` and every run is deterministic (section 24), distributing cases
is safe: a case computed on a worker is identical to one computed locally.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..models import CaseResult
from .generator import TestCase


class CaseQueue(ABC):
    """Distributes test cases to workers and collects their results."""

    @abstractmethod
    async def map(
        self,
        cases: list[TestCase],
        run_one: Callable[[TestCase], Any],
    ) -> list[CaseResult]:
        """Execute every case and return results in input order."""


class InProcessQueue(CaseQueue):
    """Bounded-concurrency asyncio execution (the default engine backend)."""

    def __init__(self, concurrency: int = 8) -> None:
        self.concurrency = max(1, concurrency)

    async def map(self, cases, run_one) -> list[CaseResult]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _guarded(case: TestCase) -> CaseResult:
            async with semaphore:
                return await run_one(case)

        return list(await asyncio.gather(*(_guarded(c) for c in cases)))


def serialize_case(case: TestCase) -> dict[str, Any]:
    """Broker-safe payload for one test case."""
    return {
        "case_id": case.case_id,
        "scenario": case.scenario.model_dump(mode="json"),
        "persona_id": case.persona_id,
        "seed": case.seed,
        "mode": case.mode,
        "bot_version": case.bot_version,
        "metadata": dict(case.metadata),
    }


def deserialize_case(payload: dict[str, Any]) -> TestCase:
    """Rebuild a test case on the worker side."""
    from ..models import Scenario

    return TestCase(
        case_id=payload["case_id"],
        scenario=Scenario.model_validate(payload["scenario"]),
        persona_id=payload.get("persona_id"),
        seed=int(payload.get("seed", 0)),
        mode=payload.get("mode", "text"),
        bot_version=payload.get("bot_version", "unknown"),
        metadata=payload.get("metadata", {}) or {},
    )


class _BrokerQueue(CaseQueue):
    """Shared logic for broker-backed queues."""

    #: name of the package that must be importable
    package = ""

    def __init__(self, task, *, timeout: float = 600.0) -> None:
        self._require()
        self.task = task
        self.timeout = timeout

    @classmethod
    def _require(cls) -> None:
        try:
            __import__(cls.package)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"{cls.__name__} requires {cls.package}: pip install {cls.package}"
            ) from exc

    async def map(self, cases, run_one) -> list[CaseResult]:  # pragma: no cover - needs broker
        payloads = [serialize_case(c) for c in cases]
        handles = [self._submit(p) for p in payloads]
        raw = await asyncio.gather(
            *(asyncio.to_thread(self._collect, h) for h in handles)
        )
        return [CaseResult.model_validate(r) for r in raw]

    def _submit(self, payload: dict[str, Any]):  # pragma: no cover
        raise NotImplementedError

    def _collect(self, handle):  # pragma: no cover
        raise NotImplementedError


class CeleryQueue(_BrokerQueue):
    """Submit cases to a Celery task (concept §31)."""

    package = "celery"

    def _submit(self, payload):  # pragma: no cover - needs a broker
        return self.task.delay(payload)

    def _collect(self, handle):  # pragma: no cover - needs a broker
        return handle.get(timeout=self.timeout)


class DramatiqQueue(_BrokerQueue):
    """Submit cases to a Dramatiq actor (concept §31)."""

    package = "dramatiq"

    def _submit(self, payload):  # pragma: no cover - needs a broker
        return self.task.send(payload)

    def _collect(self, handle):  # pragma: no cover - needs a broker
        return handle.get_result(block=True, timeout=int(self.timeout * 1000))


def run_case_payload(payload: dict[str, Any], *, bot_factory=None) -> dict[str, Any]:
    """Worker-side entry point: run one serialized case, return a serialized result.

    Register this from your Celery/Dramatiq app::

        @app.task(name="phonebot_qa.run_case")
        def run_case(payload):
            return run_case_payload(payload)
    """
    from ..adapters.bot.reference import ReferenceAppointmentBot
    from .engine import RunEngine

    case = deserialize_case(payload)
    bot = (bot_factory or (lambda v: ReferenceAppointmentBot(version=v)))(
        case.bot_version
    )
    engine = RunEngine(bot)
    result = asyncio.run(engine.run_case(case))
    return result.model_dump(mode="json")
