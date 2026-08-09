"""The discovery loop (concept sections 23 & 35).

Runs generated probes and variants against the bot, converts every violation
into a :class:`~phonebot_qa.models.Finding`, and (optionally) freezes each
finding as a regression case — closing the self-growing loop::

    rules + seed scenarios -> generate -> run -> evaluate
        -> violation? -> Finding -> Regression Case -> permanent suite member
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.bot.base import BotAdapter
from ..models import CaseResult, Finding, Persona, Scenario
from ..orchestrator.engine import RunEngine, RunSummary, summarize
from ..orchestrator.generator import generate_cases
from ..regression.store import RegressionStore
from ..simulator.personas import resolve_persona
from .rules import BUSINESS_RULES, BusinessRule, probes_for_rules
from .variants import generate_variants


@dataclass
class DiscoveryReport:
    """What one discovery pass explored and found."""

    explored: int = 0
    findings: list[Finding] = field(default_factory=list)
    regression_case_ids: list[str] = field(default_factory=list)
    summary: RunSummary | None = None

    @property
    def violations(self) -> int:
        return len(self.findings)

    def to_report(self) -> dict:
        return {
            "explored": self.explored,
            "violations": self.violations,
            "regression_cases": self.regression_case_ids,
            "findings": [
                {
                    "id": f.id,
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "scenario_id": f.scenario_id,
                    "failed_assertions": f.failed_assertions,
                }
                for f in self.findings
            ],
        }


def _severity_for(result: CaseResult) -> str:
    failed = [a for a in result.assertions if a.critical and not a.passed]
    if any(a.category == "safety" for a in failed):
        return "critical"
    if any(a.category in ("business", "tool") for a in failed):
        return "high"
    return "medium"


def finding_from_result(
    result: CaseResult, *, source: str = "discovery", created_at: str | None = None
) -> Finding:
    """Turn a failing case into a structured finding."""
    failed = [a for a in result.assertions if a.critical and not a.passed]
    # Report the most serious category present, not whichever assertion the
    # evaluator happened to emit first: a safety violation must not be filed as
    # "business" merely because a business assertion was evaluated earlier.
    order = ("safety", "business", "tool", "technical")
    category = next(
        (c for c in order if any(a.category == c for a in failed)), "technical"
    )
    return Finding(
        id=f"finding_{result.case_id}",
        category=category if category in ("business", "safety", "tool", "technical") else "technical",
        severity=_severity_for(result),
        title=(result.critical_failure or result.error or "invariant violation")[:200],
        detail="; ".join(f"{a.name}: {a.detail}" for a in failed)[:2000],
        scenario_id=result.scenario_id,
        case_id=result.case_id,
        persona_id=result.persona_id,
        seed=result.seed,
        mode=result.mode,
        bot_version=result.bot_version,
        source=source,
        failed_assertions=[a.name for a in failed],
        created_at=created_at,
    )


class DiscoveryEngine:
    """Generates, runs and triages exploratory test cases."""

    def __init__(
        self,
        bot: BotAdapter,
        *,
        personas: dict[str, Persona] | None = None,
        concurrency: int = 8,
        store: RegressionStore | None = None,
    ) -> None:
        self.bot = bot
        self.personas = personas or {}
        self.concurrency = concurrency
        self.store = store

    def build_scenarios(
        self,
        seeds: list[Scenario] | None = None,
        rules: list[BusinessRule] | None = None,
        *,
        include_variants: bool = True,
        include_rules: bool = True,
    ) -> list[Scenario]:
        """Everything discovery will explore this pass."""
        scenarios: list[Scenario] = []
        if include_rules:
            scenarios += [scenario for _rule, scenario in probes_for_rules(rules or BUSINESS_RULES)]
        if include_variants:
            for seed in seeds or []:
                scenarios += generate_variants(seed)
        return scenarios

    async def discover(
        self,
        seeds: list[Scenario] | None = None,
        rules: list[BusinessRule] | None = None,
        *,
        case_seeds: list[int] | None = None,
        capture_regressions: bool = False,
        created_at: str | None = None,
        include_variants: bool = True,
        include_rules: bool = True,
    ) -> DiscoveryReport:
        """Run one discovery pass and return its findings."""
        scenarios = self.build_scenarios(
            seeds,
            rules,
            include_variants=include_variants,
            include_rules=include_rules,
        )
        report = DiscoveryReport(explored=0)
        if not scenarios:
            return report

        cases = generate_cases(
            scenarios,
            seeds=case_seeds or [0],
            bot_version=self.bot.version,
        )
        engine = RunEngine(
            self.bot, personas=self.personas, concurrency=self.concurrency
        )
        results = await engine.run_cases(cases)
        summary = summarize(results, self.bot.version)
        report.explored = len(results)
        report.summary = summary

        by_id = {s.id: s for s in scenarios}
        for result in summary.failures():
            finding = finding_from_result(result, created_at=created_at)
            scenario = by_id.get(result.scenario_id)
            if capture_regressions and self.store is not None and scenario is not None:
                persona = resolve_persona(result.persona_id, self.personas)
                case = self.store.capture(
                    result,
                    scenario,
                    reason=f"discovery: {finding.title}",
                    created_at=created_at,
                    persona=persona,
                )
                finding.regression_case_id = case.id
                report.regression_case_ids.append(case.id)
            report.findings.append(finding)
        return report
