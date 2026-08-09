"""Automatic test discovery (concept section 23).

Beyond the hand-written scenarios, the platform explores on its own:

    Business Rules -> Generator -> Conversation -> Phonebot -> Evaluator
        -> Violation? -> Finding -> Regression Case

The generator produces scenario *variants* (mutations of a seed scenario) and
adversarial probes derived from declared business rules. Anything that breaks an
invariant becomes a :class:`~phonebot_qa.models.Finding` and can be frozen as a
regression case, so the suite grows itself (section 35).
"""

from __future__ import annotations

from .engine import DiscoveryEngine, DiscoveryReport
from .rules import BUSINESS_RULES, BusinessRule, probes_for_rules
from .variants import generate_variants

__all__ = [
    "BUSINESS_RULES",
    "BusinessRule",
    "DiscoveryEngine",
    "DiscoveryReport",
    "generate_variants",
    "probes_for_rules",
]
