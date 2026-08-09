"""Repository API over the persistence schema (concept sections 25 & 29).

``ResultsRepository`` is the only thing the rest of the platform touches. It
persists a whole run (summary + every case with its transcript, tool calls,
events, assertions, evaluations and findings) and answers the queries the
release gate and dashboard need — notably "what was the baseline's task success
for this suite?", which is what makes candidate-vs-production comparison
(section 29) possible across runs rather than only within one invocation.

SQLAlchemy is optional. When it is missing, importing this module still works
and ``HAS_SQLALCHEMY`` is ``False``; constructing a repository then raises a
clear error instead of failing at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

try:  # optional dependency
    from sqlalchemy import case, create_engine, func, select
    from sqlalchemy.orm import Session, sessionmaker

    from .schema import (
        AssertionResultRow,
        Base,
        Bot,
        BotVersion,
        ConversationRow,
        EvalResultRow,
        EventRow,
        FindingRow,
        PersonaRow,
        RegressionCaseRow,
        Run,
        RunCase,
        ScenarioRow,
        ToolCallRow,
        TurnRow,
    )

    HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_SQLALCHEMY = False


DEFAULT_SQLITE_URL = "sqlite+pysqlite:///phonebot_qa.db"


def default_database_url() -> str:
    """The configured DB URL (``PHONEBOT_DATABASE_URL``) or a local SQLite file."""
    return os.environ.get("PHONEBOT_DATABASE_URL", DEFAULT_SQLITE_URL)


def _hash_definition(definition: dict[str, Any]) -> str:
    blob = json.dumps(definition, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class ResultsRepository:
    """Durable store for runs, cases and findings."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        if not HAS_SQLALCHEMY:  # pragma: no cover
            raise RuntimeError(
                "persistence requires SQLAlchemy: pip install 'phonebot-qa[db]'"
            )
        self.url = url or default_database_url()
        self.engine = create_engine(self.url, echo=echo, future=True)
        self._session_factory = sessionmaker(bind=self.engine, future=True)
        Base.metadata.create_all(self.engine)

    def session(self) -> "Session":
        return self._session_factory()

    # -- writing ----------------------------------------------------------- #

    def save_run(
        self,
        summary,
        *,
        suite: str = "all",
        external_id: str | None = None,
        created_at: str | None = None,
        scenarios: list | None = None,
        findings: list | None = None,
    ) -> int:
        """Persist a :class:`RunSummary` and all of its cases. Returns the run id."""
        with self.session() as session:
            run = Run(
                external_id=external_id,
                suite=suite,
                bot_version=summary.bot_version,
                status="completed",
                created_at=created_at,
                total=summary.total,
                passed=summary.passed,
                failed=summary.failed,
                errored=summary.errored,
                critical_failures=summary.critical_failures,
                avg_score=summary.avg_score,
                p95_latency_ms=summary.p95_latency_ms,
            )
            session.add(run)
            session.flush()

            self._ensure_bot_version(session, summary.bot_version)
            for scenario in scenarios or []:
                self._ensure_scenario(session, scenario)

            findings_by_case: dict[str, list] = {}
            for finding in findings or []:
                findings_by_case.setdefault(finding.case_id, []).append(finding)

            for result in summary.results:
                case = self._build_case(run.id, result)
                session.add(case)
                session.flush()
                self._attach_children(session, case, result)
                for finding in findings_by_case.get(result.case_id, []):
                    session.add(self._build_finding(finding, run_case_id=case.id))

            session.commit()
            return run.id

    @staticmethod
    def _build_case(run_id: int, result) -> "RunCase":
        return RunCase(
            run_id=run_id,
            case_id=result.case_id,
            scenario_id=result.scenario_id,
            persona_id=result.persona_id,
            bot_version=result.bot_version,
            mode=result.mode,
            seed=result.seed,
            result=result.result,
            critical_failure=result.critical_failure,
            score_total=result.score.total,
            score_business=result.score.business,
            score_safety=result.score.safety,
            score_conversation=result.score.conversation,
            score_latency=result.score.latency,
            score_voice=result.score.voice,
            turns=result.latency.turns,
            duration_seconds=result.latency.duration_seconds,
            avg_latency_ms=result.latency.avg_latency_ms,
            p95_latency_ms=result.latency.p95_latency_ms,
            final_state=result.final_state,
        )

    def _attach_children(self, session: "Session", case: "RunCase", result) -> None:
        conversation = ConversationRow(
            run_case_id=case.id, duration_ms=result.conversation.duration_ms
        )
        session.add(conversation)
        session.flush()
        for turn in result.conversation.turns:
            session.add(
                TurnRow(
                    conversation_id=conversation.id,
                    idx=turn.index,
                    user_text=turn.user,
                    bot_text=turn.bot,
                    latency_ms=turn.latency_ms,
                )
            )
        for call in result.tool_calls:
            session.add(
                ToolCallRow(
                    run_case_id=case.id,
                    tool=call.tool,
                    arguments=call.arguments,
                    result=call.result if isinstance(call.result, dict) else {"value": call.result},
                    status=call.status,
                    duration_ms=call.duration_ms,
                    turn=call.turn,
                    error=call.error,
                )
            )
        for event in result.events:
            session.add(
                EventRow(
                    run_case_id=case.id,
                    type=event.type,
                    t_ms=event.t_ms,
                    turn=event.turn,
                    payload=event.payload,
                )
            )
        for assertion in result.assertions:
            session.add(
                AssertionResultRow(
                    run_case_id=case.id,
                    name=assertion.name,
                    category=assertion.category,
                    passed=assertion.passed,
                    critical=assertion.critical,
                    detail=assertion.detail,
                )
            )
        if result.eval_scores is not None:
            scores = result.eval_scores.model_dump()
            rationale = scores.pop("rationale", "")
            for metric, score in scores.items():
                session.add(
                    EvalResultRow(
                        run_case_id=case.id,
                        metric=metric,
                        score=score,
                        rationale=rationale,
                    )
                )
        if result.voice is not None and result.voice.score is not None:
            session.add(
                EvalResultRow(
                    run_case_id=case.id, metric="voice", score=result.voice.score
                )
            )

    @staticmethod
    def _build_finding(finding, *, run_case_id: int | None = None) -> "FindingRow":
        return FindingRow(
            run_case_id=run_case_id,
            finding_id=finding.id,
            category=finding.category,
            severity=finding.severity,
            title=finding.title,
            detail=finding.detail,
            scenario_id=finding.scenario_id,
            source=finding.source,
            failed_assertions=finding.failed_assertions,
            regression_case_id=finding.regression_case_id,
            created_at=finding.created_at,
        )

    def save_findings(self, findings: list) -> int:
        with self.session() as session:
            for finding in findings:
                session.add(self._build_finding(finding))
            session.commit()
            return len(findings)

    def save_regression_case(self, case, *, created_at: str | None = None) -> None:
        with self.session() as session:
            existing = session.scalar(
                select(RegressionCaseRow).where(RegressionCaseRow.case_id == case.id)
            )
            payload = case.model_dump(mode="json")
            if existing is not None:
                existing.payload = payload
                existing.reason = case.reason
            else:
                session.add(
                    RegressionCaseRow(
                        case_id=case.id,
                        scenario_id=str(case.scenario.get("id", "")),
                        persona_id=case.persona_id,
                        seed=case.seed,
                        mode=case.mode,
                        bot_version=case.bot_version,
                        reason=case.reason,
                        payload=payload,
                        created_at=created_at or case.created_at,
                    )
                )
            session.commit()

    def _ensure_bot_version(self, session: "Session", version: str) -> None:
        name = version.split("-")[0] or "bot"
        bot = session.scalar(select(Bot).where(Bot.name == name))
        if bot is None:
            bot = Bot(name=name)
            session.add(bot)
            session.flush()
        exists = session.scalar(
            select(BotVersion).where(
                BotVersion.bot_id == bot.id, BotVersion.version == version
            )
        )
        if exists is None:
            session.add(BotVersion(bot_id=bot.id, version=version))

    def _ensure_scenario(self, session: "Session", scenario) -> None:
        definition = scenario.model_dump(mode="json")
        content_hash = _hash_definition(definition)
        exists = session.scalar(
            select(ScenarioRow).where(
                ScenarioRow.scenario_id == scenario.id,
                ScenarioRow.content_hash == content_hash,
            )
        )
        if exists is None:
            session.add(
                ScenarioRow(
                    scenario_id=scenario.id,
                    content_hash=content_hash,
                    description=scenario.description,
                    tags=list(scenario.tags),
                    definition=definition,
                )
            )

    def save_persona(self, persona) -> None:
        with self.session() as session:
            exists = session.scalar(
                select(PersonaRow).where(PersonaRow.persona_id == persona.id)
            )
            definition = persona.model_dump(mode="json")
            if exists is None:
                session.add(PersonaRow(persona_id=persona.id, definition=definition))
            else:
                exists.definition = definition
            session.commit()

    # -- reading ----------------------------------------------------------- #

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self.session() as session:
            run = session.get(Run, run_id)
            if run is None:
                return None
            return {
                "id": run.id,
                "suite": run.suite,
                "bot_version": run.bot_version,
                "status": run.status,
                "total": run.total,
                "passed": run.passed,
                "failed": run.failed,
                "errored": run.errored,
                "critical_failures": run.critical_failures,
                "avg_score": run.avg_score,
                "pass_rate": (run.passed / run.total) if run.total else 0.0,
                "p95_latency_ms": run.p95_latency_ms,
            }

    def latest_run_for(self, bot_version: str, *, suite: str | None = None) -> dict[str, Any] | None:
        """Most recent run for a bot version — the baseline for the gate (§29)."""
        with self.session() as session:
            stmt = select(Run).where(Run.bot_version == bot_version)
            if suite is not None:
                stmt = stmt.where(Run.suite == suite)
            run = session.scalars(stmt.order_by(Run.id.desc()).limit(1)).first()
            return self.get_run(run.id) if run else None

    def case_results(self, run_id: int) -> list[dict[str, Any]]:
        with self.session() as session:
            rows = session.scalars(
                select(RunCase).where(RunCase.run_id == run_id).order_by(RunCase.id)
            ).all()
            return [
                {
                    "case_id": r.case_id,
                    "scenario_id": r.scenario_id,
                    "persona_id": r.persona_id,
                    "mode": r.mode,
                    "seed": r.seed,
                    "result": r.result,
                    "score": r.score_total,
                    "critical_failure": r.critical_failure,
                }
                for r in rows
            ]

    def flaky_scenarios(self, *, bot_version: str | None = None) -> list[dict[str, Any]]:
        """Scenarios with mixed PASS/FAIL history — flakiness signal for triage."""
        with self.session() as session:
            stmt = select(
                RunCase.scenario_id,
                func.count(RunCase.id),
                func.sum(case((RunCase.result == "PASS", 1), else_=0)),
            ).group_by(RunCase.scenario_id)
            if bot_version is not None:
                stmt = stmt.where(RunCase.bot_version == bot_version)
            out = []
            for scenario_id, total, passed in session.execute(stmt):
                passed = int(passed or 0)
                if 0 < passed < total:
                    out.append(
                        {
                            "scenario_id": scenario_id,
                            "runs": total,
                            "passed": passed,
                            "pass_rate": passed / total,
                        }
                    )
            return sorted(out, key=lambda r: r["pass_rate"])

    def open_findings(self) -> list[dict[str, Any]]:
        with self.session() as session:
            rows = session.scalars(select(FindingRow).order_by(FindingRow.id)).all()
            return [
                {
                    "id": r.finding_id,
                    "severity": r.severity,
                    "category": r.category,
                    "title": r.title,
                    "scenario_id": r.scenario_id,
                    "source": r.source,
                    "regression_case_id": r.regression_case_id,
                }
                for r in rows
            ]
