"""DeepEval integration behind the :class:`Judge` interface (concept §21 & §30).

DeepEval provides LLM-as-a-judge conversation metrics. Per section 30 it is
abstracted behind our own interface so it can be swapped out; per section 37 its
verdicts only shape the *soft* conversation score and never decide business
correctness.

DeepEval is optional (and needs an LLM API key). When it is unavailable or a
metric errors, the judge falls back to the deterministic
:class:`~phonebot_qa.evaluation.judge.HeuristicJudge` so runs never break.
"""

from __future__ import annotations

from ..evaluation.judge import HeuristicJudge, Judge
from ..models import Conversation, EvalScores

try:  # optional dependency
    import deepeval  # noqa: F401

    HAS_DEEPEVAL = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_DEEPEVAL = False


class DeepEvalJudge(Judge):
    """Score conversation quality with DeepEval's conversational metrics.

    Maps DeepEval metric scores (0..1) onto the platform's 1..5 scale so results
    stay comparable with the built-in judge.
    """

    #: platform dimension -> DeepEval conversational metric class name
    METRIC_MAP = {
        "naturalness": "ConversationCompletenessMetric",
        "clarity": "ConversationRelevancyMetric",
        "efficiency": "ConversationCompletenessMetric",
        "correction_handling": "ConversationRelevancyMetric",
    }

    def __init__(self, *, model: str | None = None, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold
        self._fallback = HeuristicJudge()

    @property
    def available(self) -> bool:
        return HAS_DEEPEVAL

    @staticmethod
    def _to_five(score: float | None) -> float | None:
        if score is None:
            return None
        return round(max(0.0, min(1.0, float(score))) * 4.0 + 1.0, 2)

    def _build_test_case(self, conversation: Conversation):  # pragma: no cover - needs deepeval
        from deepeval.test_case import ConversationalTestCase, LLMTestCase

        turns = [
            LLMTestCase(input=t.user, actual_output=t.bot) for t in conversation.turns
        ]
        return ConversationalTestCase(turns=turns)

    async def evaluate(self, conversation: Conversation, *, persona=None) -> EvalScores:
        if not HAS_DEEPEVAL or not conversation.turns:
            return await self._fallback.evaluate(conversation, persona=persona)
        try:  # pragma: no cover - requires deepeval + an LLM key
            import deepeval.metrics as metrics_mod

            test_case = self._build_test_case(conversation)
            scores: dict[str, float | None] = {}
            cache: dict[str, float | None] = {}
            for dimension, metric_name in self.METRIC_MAP.items():
                if metric_name in cache:
                    scores[dimension] = cache[metric_name]
                    continue
                metric_cls = getattr(metrics_mod, metric_name, None)
                if metric_cls is None:
                    scores[dimension] = None
                    continue
                metric = (
                    metric_cls(threshold=self.threshold, model=self.model)
                    if self.model
                    else metric_cls(threshold=self.threshold)
                )
                metric.measure(test_case)
                cache[metric_name] = metric.score
                scores[dimension] = metric.score
            mapped = {k: self._to_five(v) for k, v in scores.items()}
            if not any(v is not None for v in mapped.values()):
                return await self._fallback.evaluate(conversation, persona=persona)
            return EvalScores(**mapped, rationale="deepeval")
        except Exception as exc:  # pragma: no cover - defensive
            fallback = await self._fallback.evaluate(conversation, persona=persona)
            fallback.rationale = f"deepeval unavailable ({exc}); heuristic fallback"
            return fallback
