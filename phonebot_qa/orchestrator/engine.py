"""In-process run engine + suite aggregation (concept sections 6-7 & 26).

Executes generated test cases concurrently with asyncio (a bounded semaphore
stands in for a worker pool; the same interface maps onto Celery/Dramatiq at
scale). Builds the right simulator per case — scripted for red-team, heuristic
for cooperative callers — and runs each through the conversation runner and the
evaluation pipeline, then aggregates a :class:`RunSummary`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.bot.base import BotAdapter
from ..adapters.bot.reference import ReferenceAppointmentBot
from ..evaluation.pipeline import EvaluationPipeline
from ..models import CaseResult, Persona, Scenario
from ..runner.conversation import ConversationRunner, RunnerConfig
from ..scenario.knowledge import isolate_user_knowledge
from ..simulator.base import UserSimulator
from ..simulator.heuristic import HeuristicSimulator
from ..simulator.personas import resolve_persona
from ..simulator.scripted import ScriptedSimulator
from .generator import TestCase, generate_cases


def _default_simulator(scenario: Scenario, knowledge, seed: int) -> UserSimulator:
    # Any scenario that ships an explicit script (red-team attacks, discovery
    # probes) is played verbatim; everything else gets the goal-driven caller.
    lines = scenario.user.user_visible.get("redteam_lines")
    if lines:
        return ScriptedSimulator(knowledge, lines=list(lines), seed=seed)
    return HeuristicSimulator(knowledge, seed=seed)


@dataclass
class RunSummary:
    """Aggregate view of a completed run (section 26)."""

    bot_version: str
    total: int
    passed: int
    failed: int
    errored: int
    critical_failures: int
    avg_score: float
    avg_turns: float
    p95_latency_ms: float
    voice_success: float | None
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if r.result != "PASS"]

    def to_report(self) -> dict:
        return {
            "bot_version": self.bot_version,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_rate": round(self.pass_rate, 4),
            "critical_failures": self.critical_failures,
            "avg_score": round(self.avg_score, 4),
            "avg_turns": round(self.avg_turns, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "voice_success": self.voice_success,
            "cases": [
                {
                    "case_id": r.case_id,
                    "scenario": r.scenario_id,
                    "persona": r.persona_id,
                    "seed": r.seed,
                    "mode": r.mode,
                    "result": r.result,
                    "score": r.score.total,
                    "critical_failure": r.critical_failure,
                }
                for r in self.results
            ],
        }


def summarize(results: list[CaseResult], bot_version: str) -> RunSummary:
    total = len(results)
    passed = sum(1 for r in results if r.result == "PASS")
    failed = sum(1 for r in results if r.result == "FAIL")
    errored = sum(1 for r in results if r.result == "ERROR")
    critical = sum(1 for r in results if r.critical_failure)
    avg_score = sum(r.score.total for r in results) / total if total else 0.0
    avg_turns = sum(r.latency.turns for r in results) / total if total else 0.0
    p95s = sorted(r.latency.p95_latency_ms for r in results)
    p95 = p95s[int(0.95 * (len(p95s) - 1))] if p95s else 0.0
    voice_results = [r for r in results if r.voice and r.voice.score is not None]
    voice_success = (
        sum(r.voice.score for r in voice_results) / len(voice_results)
        if voice_results
        else None
    )
    return RunSummary(
        bot_version=bot_version,
        total=total,
        passed=passed,
        failed=failed,
        errored=errored,
        critical_failures=critical,
        avg_score=avg_score,
        avg_turns=avg_turns,
        p95_latency_ms=p95,
        voice_success=voice_success,
        results=results,
    )


class RunEngine:
    """Runs test cases concurrently and aggregates results."""

    def __init__(
        self,
        bot: BotAdapter | None = None,
        *,
        pipeline: EvaluationPipeline | None = None,
        personas: dict[str, Persona] | None = None,
        runner_config: RunnerConfig | None = None,
        concurrency: int = 8,
        simulator_factory=_default_simulator,
        queue=None,
        voice_config=None,
    ) -> None:
        self.bot = bot or ReferenceAppointmentBot()
        self.pipeline = pipeline or EvaluationPipeline()
        self.personas = personas or {}
        self.runner_config = runner_config or RunnerConfig()
        self.concurrency = concurrency
        self.simulator_factory = simulator_factory
        # Distribution backend (concept §31). Defaults to in-process asyncio;
        # swap for CeleryQueue/DramatiqQueue to fan out across workers.
        from .workers import InProcessQueue

        self.queue = queue or InProcessQueue(concurrency)
        # Voice-mode configuration (concept §12.2/§34); None => text only.
        self.voice_config = voice_config

    def _build_runner(self, case: TestCase):
        """Text or voice runner depending on the case's mode (concept §12)."""
        if case.mode != "voice":
            return ConversationRunner(self.bot, config=self.runner_config)
        from ..runner.voice import VoiceConfig, VoiceConversationRunner

        # Start from the scenario's own ``audio:`` block, then let an explicit
        # engine-level config (a CLI noise sweep) override the profile/transport.
        # Scenario-specific semantics like barge-in are always preserved.
        config = VoiceConfig.from_scenario(case.scenario)
        override = self.voice_config
        if override is not None:
            config.profile = override.profile
            config.transport = override.transport
        return VoiceConversationRunner(self.bot, config=config)

    async def run_case(self, case: TestCase) -> CaseResult:
        persona = resolve_persona(case.persona_id, self.personas)
        knowledge = isolate_user_knowledge(case.scenario, persona)
        simulator = self.simulator_factory(case.scenario, knowledge, case.seed)
        runner = self._build_runner(case)
        artifacts = await runner.run(
            scenario=case.scenario, simulator=simulator, seed=case.seed
        )
        return await self.pipeline.evaluate(
            case.scenario,
            artifacts,
            case_id=case.case_id,
            persona_id=case.persona_id,
            bot_version=self.bot.version,
            mode=case.mode,
            seed=case.seed,
        )

    async def run_cases(self, cases: list[TestCase]) -> list[CaseResult]:
        """Run every case through the configured queue, preserving input order."""
        return await self.queue.map(cases, self.run_case)


async def run_suite(
    scenarios: list[Scenario],
    *,
    bot: BotAdapter | None = None,
    personas: dict[str, Persona] | None = None,
    persona_ids: list[str | None] | None = None,
    seeds: list[int] | None = None,
    modes: list[str] | None = None,
    pipeline: EvaluationPipeline | None = None,
    concurrency: int = 8,
) -> RunSummary:
    """Generate + run + summarize a suite in one call."""
    bot = bot or ReferenceAppointmentBot()
    engine = RunEngine(
        bot, pipeline=pipeline, personas=personas, concurrency=concurrency
    )
    cases = generate_cases(
        scenarios,
        personas=persona_ids,
        seeds=seeds,
        modes=modes,
        bot_version=bot.version,
    )
    results = await engine.run_cases(cases)
    return summarize(results, bot.version)
