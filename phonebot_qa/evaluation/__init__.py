"""Evaluation pipeline (concept sections 19-27).

Order matters (section 19): deterministic business, safety and tool assertions
run first and own the critical PASS/FAIL decision; technical metrics, the LLM
judge and voice metrics contribute to the *score* but never override a critical
deterministic failure (sections 27 & 37).
"""

from __future__ import annotations

from .assertions import evaluate_assertions
from .judge import HeuristicJudge, Judge
from .pipeline import EvaluationPipeline, evaluate
from .scoring import DEFAULT_WEIGHTS, ScoreWeights, compute_score
from .voice import VoiceEvaluator

__all__ = [
    "DEFAULT_WEIGHTS",
    "EvaluationPipeline",
    "HeuristicJudge",
    "Judge",
    "ScoreWeights",
    "VoiceEvaluator",
    "compute_score",
    "evaluate",
    "evaluate_assertions",
]
