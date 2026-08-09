"""Integrations with external OSS components (concept section 30).

Promptfoo (red teaming) and DeepEval (LLM evaluation) are integrated as
*replaceable* components behind the platform's own interfaces — per section 36,
the durable IP is the scenarios, assertions and regression store, not any single
third-party tool. Neither package is a hard dependency: the platform runs fully
without them and these modules degrade to clear errors or deterministic
fallbacks.
"""

from __future__ import annotations

from .deepeval import HAS_DEEPEVAL, DeepEvalJudge
from .promptfoo import (
    PromptfooConfig,
    build_promptfoo_config,
    findings_from_promptfoo,
    write_promptfoo_config,
)

__all__ = [
    "HAS_DEEPEVAL",
    "DeepEvalJudge",
    "PromptfooConfig",
    "build_promptfoo_config",
    "findings_from_promptfoo",
    "write_promptfoo_config",
]
