"""Production failure ingestion (concept sections 23 & 35).

The long-term target picture is a self-growing test system::

    Production -> auffälliger Call -> Failure Analysis -> Scenario Generation
        -> Regression Case -> Test Suite -> Bot change -> Candidate Evaluation

This package implements the left half: take a real production call trace and
synthesize a runnable scenario + regression case from it, so a bug that happened
once to a real caller can never happen again unnoticed.
"""

from __future__ import annotations

from .ingest import ProductionTrace, scenario_from_trace, regression_case_from_trace

__all__ = ["ProductionTrace", "regression_case_from_trace", "scenario_from_trace"]
