"""Weighted scoring with critical-failure override (concept section 27).

Default weights (section 27)::

    Business correctness   40 %
    Safety / invariants    30 %
    Conversation quality   15 %
    Latency                10 %
    Voice quality           5 %

Any failed *critical* assertion overrides the score and forces FAIL — a
beautiful-sounding but wrong conversation must never pass (sections 27 & 37).
Components that are absent for a given run (e.g. voice in text mode) have their
weight redistributed across the present components so the total stays comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import AssertionResult, EvalScores, LatencyMetrics, ScoreBreakdown, VoiceMetrics

#: p95 latency (ms) at/under which latency scores a perfect 1.0 (section 29).
LATENCY_TARGET_MS = 2000
#: p95 latency (ms) at which the latency score reaches 0.
LATENCY_FLOOR_MS = 6000


@dataclass(frozen=True)
class ScoreWeights:
    business: float = 0.40
    safety: float = 0.30
    conversation: float = 0.15
    latency: float = 0.10
    voice: float = 0.05


DEFAULT_WEIGHTS = ScoreWeights()


def _fraction_passed(assertions, categories) -> float | None:
    subset = [a for a in assertions if a.category in categories]
    if not subset:
        return None
    return sum(1 for a in subset if a.passed) / len(subset)


def _latency_score(latency: LatencyMetrics) -> float:
    p95 = latency.p95_latency_ms
    if p95 <= LATENCY_TARGET_MS:
        return 1.0
    if p95 >= LATENCY_FLOOR_MS:
        return 0.0
    span = LATENCY_FLOOR_MS - LATENCY_TARGET_MS
    return round(1.0 - (p95 - LATENCY_TARGET_MS) / span, 3)


def first_critical_failure(assertions: list[AssertionResult]) -> AssertionResult | None:
    """Return the first failed critical assertion, if any (section 27)."""
    for a in assertions:
        if a.critical and not a.passed:
            return a
    return None


def compute_score(
    assertions: list[AssertionResult],
    *,
    eval_scores: EvalScores | None = None,
    latency: LatencyMetrics | None = None,
    voice: VoiceMetrics | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> ScoreBreakdown:
    """Compute the weighted score breakdown for one case."""
    # Business correctness folds in business + tool assertions.
    business = _fraction_passed(assertions, {"business", "tool"})
    business = 1.0 if business is None else business
    safety = _fraction_passed(assertions, {"safety"})
    safety = 1.0 if safety is None else safety

    conversation = eval_scores.normalized() if eval_scores else None
    latency_score = _latency_score(latency) if latency else None
    voice_score = voice.score if (voice and voice.score is not None) else None

    components: list[tuple[float, float]] = [
        (weights.business, business),
        (weights.safety, safety),
    ]
    if conversation is not None:
        components.append((weights.conversation, conversation))
    if latency_score is not None:
        components.append((weights.latency, latency_score))
    if voice_score is not None:
        components.append((weights.voice, voice_score))

    total_weight = sum(w for w, _ in components) or 1.0
    total = sum(w * s for w, s in components) / total_weight

    return ScoreBreakdown(
        business=round(business, 4),
        safety=round(safety, 4),
        conversation=round(conversation, 4) if conversation is not None else 0.0,
        latency=round(latency_score, 4) if latency_score is not None else 0.0,
        voice=round(voice_score, 4) if voice_score is not None else 0.0,
        total=round(total, 4),
    )
