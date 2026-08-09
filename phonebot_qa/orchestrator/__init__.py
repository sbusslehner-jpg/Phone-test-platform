"""Test orchestration (concept sections 6 & 28-29).

Turns a suite request into concrete test cases (scenario × persona × seed × mode
× bot version), runs them, aggregates the results, and applies the CI release
gate (baseline vs candidate). The MVP runs cases in-process with asyncio; the
same seams (case generation, a run engine) map onto Celery/Dramatiq workers for
scale (section 31) without changing the interfaces.
"""

from __future__ import annotations

from .engine import RunEngine, RunSummary, run_suite
from .gate import GateResult, SuiteMetrics, release_gate
from .generator import TestCase, generate_cases

__all__ = [
    "GateResult",
    "RunEngine",
    "RunSummary",
    "SuiteMetrics",
    "TestCase",
    "generate_cases",
    "release_gate",
    "run_suite",
]
