"""SQLAlchemy schema — the concept's data model (section 25).

Tables mirror the entity list in the concept document::

    bots · bot_versions · scenarios · personas · test_suites · runs · run_cases
    conversations · turns · tool_calls · events · assertion_results
    eval_results · findings · regression_cases

The central relationship is ``run_case`` -> {conversation, tool_calls, events,
assertion_results, eval_results, findings}.

Import of this module requires SQLAlchemy; callers should go through
:mod:`phonebot_qa.persistence.repository`, which degrades gracefully.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all platform tables."""


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)

    versions: Mapped[list["BotVersion"]] = relationship(
        back_populates="bot", cascade="all, delete-orphan"
    )


class BotVersion(Base):
    __tablename__ = "bot_versions"
    __table_args__ = (UniqueConstraint("bot_id", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"))
    version: Mapped[str] = mapped_column(String(200))

    bot: Mapped[Bot] = relationship(back_populates="versions")


class ScenarioRow(Base):
    """A scenario *snapshot* — scenarios evolve, results must stay explainable."""

    __tablename__ = "scenarios"
    __table_args__ = (UniqueConstraint("scenario_id", "content_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(200), index=True)
    # Version = content hash, so a changed scenario is a new row (section 25's
    # scenario_versions) without needing a manual bump.
    content_hash: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    definition: Mapped[dict] = mapped_column(JSON, default=dict)


class PersonaRow(Base):
    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(primary_key=True)
    persona_id: Mapped[str] = mapped_column(String(200), unique=True)
    definition: Mapped[dict] = mapped_column(JSON, default=dict)


class TestSuite(Base):
    __tablename__ = "test_suites"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(200), index=True, default=None)
    suite: Mapped[str] = mapped_column(String(200), default="all")
    bot_version: Mapped[str] = mapped_column(String(200), default="unknown")
    status: Mapped[str] = mapped_column(String(40), default="completed")
    created_at: Mapped[str | None] = mapped_column(String(40), default=None)
    # Denormalised headline metrics so the dashboard/gate can query cheaply.
    total: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    errored: Mapped[int] = mapped_column(Integer, default=0)
    critical_failures: Mapped[int] = mapped_column(Integer, default=0)
    avg_score: Mapped[float] = mapped_column(Float, default=0.0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    cases: Mapped[list["RunCase"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RunCase(Base):
    __tablename__ = "run_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(400), index=True)
    scenario_id: Mapped[str] = mapped_column(String(200), index=True)
    persona_id: Mapped[str | None] = mapped_column(String(200), default=None)
    bot_version: Mapped[str] = mapped_column(String(200), default="unknown")
    mode: Mapped[str] = mapped_column(String(20), default="text")
    seed: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[str] = mapped_column(String(20), default="FAIL")
    critical_failure: Mapped[str | None] = mapped_column(Text, default=None)
    score_total: Mapped[float] = mapped_column(Float, default=0.0)
    score_business: Mapped[float] = mapped_column(Float, default=0.0)
    score_safety: Mapped[float] = mapped_column(Float, default=0.0)
    score_conversation: Mapped[float] = mapped_column(Float, default=0.0)
    score_latency: Mapped[float] = mapped_column(Float, default=0.0)
    score_voice: Mapped[float] = mapped_column(Float, default=0.0)
    turns: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    final_state: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped[Run] = relationship(back_populates="cases")
    conversation: Mapped["ConversationRow | None"] = relationship(
        back_populates="run_case", cascade="all, delete-orphan", uselist=False
    )
    tool_calls: Mapped[list["ToolCallRow"]] = relationship(
        back_populates="run_case", cascade="all, delete-orphan"
    )
    events: Mapped[list["EventRow"]] = relationship(
        back_populates="run_case", cascade="all, delete-orphan"
    )
    assertion_results: Mapped[list["AssertionResultRow"]] = relationship(
        back_populates="run_case", cascade="all, delete-orphan"
    )
    eval_results: Mapped[list["EvalResultRow"]] = relationship(
        back_populates="run_case", cascade="all, delete-orphan"
    )
    findings: Mapped[list["FindingRow"]] = relationship(
        back_populates="run_case", cascade="all, delete-orphan"
    )


class ConversationRow(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_case_id: Mapped[int] = mapped_column(ForeignKey("run_cases.id"), index=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    run_case: Mapped[RunCase] = relationship(back_populates="conversation")
    turns: Mapped[list["TurnRow"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    user_text: Mapped[str] = mapped_column(Text, default="")
    bot_text: Mapped[str] = mapped_column(Text, default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    conversation: Mapped[ConversationRow] = relationship(back_populates="turns")


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_case_id: Mapped[int] = mapped_column(ForeignKey("run_cases.id"), index=True)
    tool: Mapped[str] = mapped_column(String(200), index=True)
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, default=None)
    status: Mapped[str] = mapped_column(String(20), default="success")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    turn: Mapped[int | None] = mapped_column(Integer, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    run_case: Mapped[RunCase] = relationship(back_populates="tool_calls")


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_case_id: Mapped[int] = mapped_column(ForeignKey("run_cases.id"), index=True)
    type: Mapped[str] = mapped_column(String(100), index=True)
    t_ms: Mapped[int] = mapped_column(Integer, default=0)
    turn: Mapped[int | None] = mapped_column(Integer, default=None)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    run_case: Mapped[RunCase] = relationship(back_populates="events")


class AssertionResultRow(Base):
    __tablename__ = "assertion_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_case_id: Mapped[int] = mapped_column(ForeignKey("run_cases.id"), index=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(40), default="business")
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    critical: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[str] = mapped_column(Text, default="")

    run_case: Mapped[RunCase] = relationship(back_populates="assertion_results")


class EvalResultRow(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_case_id: Mapped[int] = mapped_column(ForeignKey("run_cases.id"), index=True)
    metric: Mapped[str] = mapped_column(String(100), index=True)
    score: Mapped[float | None] = mapped_column(Float, default=None)
    rationale: Mapped[str] = mapped_column(Text, default="")

    run_case: Mapped[RunCase] = relationship(back_populates="eval_results")


class FindingRow(Base):
    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_case_id: Mapped[int | None] = mapped_column(
        ForeignKey("run_cases.id"), index=True, default=None
    )
    finding_id: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(40), default="safety")
    severity: Mapped[str] = mapped_column(String(20), default="high")
    title: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    scenario_id: Mapped[str] = mapped_column(String(200), default="")
    source: Mapped[str] = mapped_column(String(40), default="discovery")
    failed_assertions: Mapped[list] = mapped_column(JSON, default=list)
    regression_case_id: Mapped[str | None] = mapped_column(String(400), default=None)
    created_at: Mapped[str | None] = mapped_column(String(40), default=None)

    run_case: Mapped[RunCase | None] = relationship(back_populates="findings")


class RegressionCaseRow(Base):
    __tablename__ = "regression_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str] = mapped_column(String(400), unique=True)
    scenario_id: Mapped[str] = mapped_column(String(200), index=True)
    persona_id: Mapped[str | None] = mapped_column(String(200), default=None)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    mode: Mapped[str] = mapped_column(String(20), default="text")
    bot_version: Mapped[str] = mapped_column(String(200), default="unknown")
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[str | None] = mapped_column(String(40), default=None)
