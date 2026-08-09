"""The ordered evaluation pipeline (concept section 19).

Runs the stages in the mandated order — business/safety/tool assertions first,
then technical metrics, then the LLM judge, then voice — and assembles the final
:class:`~phonebot_qa.models.CaseResult`. The PASS/FAIL verdict comes from the
deterministic critical assertions; the judge and voice scores only shape the
numeric score (sections 27 & 37).
"""

from __future__ import annotations

import math

from ..models import CaseResult, LatencyMetrics
from .assertions import evaluate_assertions
from .judge import HeuristicJudge, Judge
from .scoring import DEFAULT_WEIGHTS, ScoreWeights, compute_score, first_critical_failure
from .voice import VoiceEvaluator


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _latency_metrics(conversation) -> LatencyMetrics:
    latencies = [float(t.latency_ms) for t in conversation.turns]
    return LatencyMetrics(
        turns=conversation.turn_count,
        duration_seconds=round(conversation.duration_ms / 1000.0, 3),
        avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        p95_latency_ms=round(_percentile(latencies, 95), 2),
    )


class EvaluationPipeline:
    """Configurable evaluation pipeline (judge, voice, weights)."""

    def __init__(
        self,
        *,
        judge: Judge | None = None,
        voice_evaluator: VoiceEvaluator | None = None,
        weights: ScoreWeights = DEFAULT_WEIGHTS,
    ) -> None:
        self.judge = judge or HeuristicJudge()
        self.voice = voice_evaluator or VoiceEvaluator()
        self.weights = weights

    async def evaluate(
        self,
        scenario,
        artifacts,
        *,
        case_id: str,
        persona_id: str | None = None,
        bot_version: str = "unknown",
        mode: str = "text",
        seed: int = 0,
    ) -> CaseResult:
        # 1-3. Deterministic assertions (business, safety, tool, technical).
        assertions = evaluate_assertions(scenario, artifacts)

        # 4. Technical metrics.
        latency = _latency_metrics(artifacts.conversation)

        # 5. LLM judge (qualitative, non-authoritative).
        eval_scores = await self.judge.evaluate(
            artifacts.conversation, persona=persona_id
        )

        # 6. Voice metrics (None in text mode).
        voice = self.voice.evaluate(artifacts.events, mode=mode)

        score = compute_score(
            assertions,
            eval_scores=eval_scores,
            latency=latency,
            voice=voice,
            weights=self.weights,
        )

        # Verdict: critical deterministic assertions decide PASS/FAIL.
        crit = first_critical_failure(assertions)
        if artifacts.error is not None:
            result = "ERROR"
            critical_failure = artifacts.error
        elif crit is not None:
            result = "FAIL"
            critical_failure = f"{crit.name}: {crit.detail}".strip().rstrip(":")
        else:
            result = "PASS"
            critical_failure = None

        # Critical failures override the score (concept §27): a failed run must
        # not report a high total that would mislead ranking / avg_score. The
        # component breakdown is kept for diagnostics; only the headline zeroes.
        if result != "PASS":
            score.total = 0.0

        return CaseResult(
            case_id=case_id,
            scenario_id=scenario.id,
            scenario_tags=list(scenario.tags),
            persona_id=persona_id,
            bot_version=bot_version,
            mode=mode,
            seed=seed,
            result=result,
            critical_failure=critical_failure,
            score=score,
            latency=latency,
            eval_scores=eval_scores,
            voice=voice,
            conversation=artifacts.conversation,
            tool_calls=artifacts.tool_calls,
            events=artifacts.events,
            assertions=assertions,
            final_state=artifacts.final_state,
            error=artifacts.error,
        )


async def evaluate(scenario, artifacts, **kwargs) -> CaseResult:
    """Convenience wrapper using the default pipeline."""
    return await EvaluationPipeline().evaluate(scenario, artifacts, **kwargs)
