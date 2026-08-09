"""Adversarial / red-team testing (concept sections 22-23).

Red-team cases try to make the bot violate an invariant: prompt injection,
authorization bypass, PII leakage, manipulation, policy violations. They are
expressed as ordinary scenarios whose expectations assert that the bot *refused*
and that no unauthorized state change or data access happened.
"""

from __future__ import annotations

from .attacks import ATTACKS, AttackVector, build_redteam_scenario, redteam_scenarios

__all__ = ["ATTACKS", "AttackVector", "build_redteam_scenario", "redteam_scenarios"]
