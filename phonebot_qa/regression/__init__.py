"""Regression management (concept sections 24 & 36).

Every real failure can become a regression case — a fully self-contained,
replayable record of the exact conditions that broke the bot. The regression
store is part of the platform's durable IP (section 36): the accumulated cases
are the institutional memory of every bug ever found.
"""

from __future__ import annotations

from .store import RegressionCase, RegressionStore

__all__ = ["RegressionCase", "RegressionStore"]
