"""Persistence layer (concept sections 25 & 31).

Implements the concept's data model on SQLAlchemy so runs, cases, transcripts,
tool calls, events, assertions, evaluations, findings and regression cases are
durable and queryable — the historical test results are part of the platform's
core IP (section 36).

SQLAlchemy is an optional dependency: the platform runs fully in-memory without
it. ``pip install 'phonebot-qa[db]'`` enables this module. The default URL is a
local SQLite file so nothing external is needed; set ``PHONEBOT_DATABASE_URL``
to a Postgres DSN for the production topology described in section 31.
"""

from __future__ import annotations

from .repository import HAS_SQLALCHEMY, ResultsRepository, default_database_url

__all__ = ["HAS_SQLALCHEMY", "ResultsRepository", "default_database_url"]
