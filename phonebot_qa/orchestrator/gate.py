"""CI release gate (concept section 29).

A candidate bot version is always compared against the current production
baseline. The deployment rule is::

    candidate_success >= baseline_success
    AND critical_failures == 0
    AND regression_failures == 0

If all hold the candidate may deploy (PASS); otherwise deployment is blocked
(BLOCK), with explicit reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import RunSummary


@dataclass
class SuiteMetrics:
    """The comparable headline metrics for one bot version (section 29)."""

    bot_version: str
    task_success: float
    critical_errors: int
    avg_turns: float
    p95_latency_ms: float
    voice_success: float | None

    @classmethod
    def from_summary(cls, summary: RunSummary) -> "SuiteMetrics":
        return cls(
            bot_version=summary.bot_version,
            task_success=summary.pass_rate,
            critical_errors=summary.critical_failures,
            avg_turns=summary.avg_turns,
            p95_latency_ms=summary.p95_latency_ms,
            voice_success=summary.voice_success,
        )


@dataclass
class GateResult:
    """The gate decision plus the reasons and the comparison table."""

    decision: str  # "PASS" | "BLOCK"
    reasons: list[str] = field(default_factory=list)
    baseline: SuiteMetrics | None = None
    candidate: SuiteMetrics | None = None
    regression_failures: int = 0

    @property
    def deployable(self) -> bool:
        return self.decision == "PASS"

    def to_report(self) -> dict:
        def _m(m: SuiteMetrics | None) -> dict | None:
            return None if m is None else vars(m)

        return {
            "decision": self.decision,
            "deployable": self.deployable,
            "reasons": self.reasons,
            "regression_failures": self.regression_failures,
            "baseline": _m(self.baseline),
            "candidate": _m(self.candidate),
        }


def _count_regression_failures(summary: RunSummary) -> int:
    """Failing cases that are regression cases.

    A regression case is identified by the ``regression`` scenario tag (added by
    the regression store on replay) or, for belt-and-braces, a ``regr_``
    scenario-id prefix.
    """
    return sum(
        1
        for r in summary.results
        if r.result != "PASS"
        and ("regression" in r.scenario_tags or r.scenario_id.startswith("regr_"))
    )


def release_gate(
    candidate: RunSummary,
    baseline: RunSummary | None = None,
    *,
    regression_failures: int | None = None,
    min_success_delta: float = 0.0,
) -> GateResult:
    """Apply the deployment rule (section 29).

    ``regression_failures`` defaults to the number of failed cases whose
    scenario id marks them as regression cases (``regr_*``). ``min_success_delta``
    lets a team require the candidate to *beat* the baseline by a margin.
    """
    if regression_failures is None:
        regression_failures = _count_regression_failures(candidate)

    reasons: list[str] = []

    if candidate.critical_failures != 0:
        reasons.append(
            f"{candidate.critical_failures} critical failure(s) in candidate"
        )
    if regression_failures != 0:
        reasons.append(f"{regression_failures} regression case(s) failing")
    if baseline is not None:
        required = baseline.pass_rate + min_success_delta
        if candidate.pass_rate < required:
            reasons.append(
                f"candidate task success {candidate.pass_rate:.3f} < required "
                f"{required:.3f} (baseline {baseline.pass_rate:.3f})"
            )

    decision = "PASS" if not reasons else "BLOCK"
    return GateResult(
        decision=decision,
        reasons=reasons,
        baseline=SuiteMetrics.from_summary(baseline) if baseline else None,
        candidate=SuiteMetrics.from_summary(candidate),
        regression_failures=regression_failures,
    )
