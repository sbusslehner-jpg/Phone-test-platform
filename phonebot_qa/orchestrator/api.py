"""FastAPI orchestrator API (concept section 6).

Implements the run-management surface from the concept document::

    POST /runs            -> create & start a run
    GET  /runs            -> list runs
    GET  /runs/{id}       -> run status + summary
    GET  /runs/{id}/cases -> per-case results
    POST /runs/{id}/cancel

Runs execute as background asyncio tasks against the in-process reference bot;
the same handler shape maps onto Celery/Dramatiq workers for scale (section 31).
FastAPI/uvicorn are optional dependencies (``pip install 'phonebot-qa[api]'``);
importing this module without them yields ``app = None``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

try:  # optional dependency
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_FASTAPI = False

from ..adapters.bot.reference import ReferenceAppointmentBot
from ..redteam.attacks import redteam_scenarios
from ..scenario.loader import load_personas, load_scenarios
from .engine import RunEngine, summarize
from .generator import generate_cases


def _scenarios_root() -> Path:
    return Path(os.environ.get("PHONEBOT_SCENARIOS_DIR", "scenarios"))


def _personas_root() -> Path:
    return Path(os.environ.get("PHONEBOT_PERSONAS_DIR", "personas"))


def _load_suite(suite: str):
    """Resolve a suite name to a list of scenarios."""
    if suite == "redteam":
        return redteam_scenarios()
    root = _scenarios_root()
    if suite in ("all", "", None):
        scenarios = load_scenarios(root)
    else:
        sub = root / suite
        scenarios = load_scenarios(sub) if sub.exists() else []
    if suite in ("all", "staging"):
        scenarios = scenarios + redteam_scenarios()
    return scenarios


if _HAS_FASTAPI:

    class RunRequest(BaseModel):
        suite: str = "all"
        bot_version: str = "reference-1.0"
        modes: list[str] = Field(default_factory=lambda: ["text"])
        personas: list[str] | None = None
        iterations: int = 1
        seeds: list[int] | None = None

    class _Run:
        def __init__(self, run_id: str, request: RunRequest) -> None:
            self.id = run_id
            self.request = request
            self.status = "pending"
            self.summary: dict[str, Any] | None = None
            self.error: str | None = None
            self.task: asyncio.Task | None = None

    def create_app() -> "FastAPI":
        app = FastAPI(
            title="Phonebot Test Platform — Orchestrator",
            version="0.1.0",
            description="CI/CD orchestrator for phonebot QA (concept section 6).",
        )
        runs: dict[str, _Run] = {}
        counter = {"n": 0}

        async def _execute(run: _Run) -> None:
            run.status = "running"
            try:
                scenarios = _load_suite(run.request.suite)
                if not scenarios:
                    run.status = "error"
                    run.error = f"no scenarios found for suite {run.request.suite!r}"
                    return
                personas = {}
                proot = _personas_root()
                if proot.exists():
                    personas = load_personas(proot)
                bot = ReferenceAppointmentBot(version=run.request.bot_version)
                seeds = run.request.seeds or list(range(run.request.iterations))
                cases = generate_cases(
                    scenarios,
                    personas=run.request.personas,
                    seeds=seeds,
                    modes=run.request.modes,
                    bot_version=bot.version,
                )
                engine = RunEngine(bot, personas=personas)
                results = await engine.run_cases(cases)
                run.summary = summarize(results, bot.version).to_report()
                run.status = "completed"
            except asyncio.CancelledError:
                run.status = "cancelled"
                raise
            except Exception as exc:  # pragma: no cover - defensive
                run.status = "error"
                run.error = f"{type(exc).__name__}: {exc}"

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        @app.post("/runs")
        async def create_run(request: RunRequest) -> dict:
            counter["n"] += 1
            run_id = f"run-{counter['n']}"
            run = _Run(run_id, request)
            runs[run_id] = run
            run.task = asyncio.create_task(_execute(run))
            return {"id": run_id, "status": run.status}

        @app.get("/runs")
        async def list_runs() -> dict:
            return {
                "runs": [
                    {"id": r.id, "status": r.status, "suite": r.request.suite}
                    for r in runs.values()
                ]
            }

        @app.get("/runs/{run_id}")
        async def get_run(run_id: str) -> dict:
            run = runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            summary = run.summary or {}
            return {
                "id": run.id,
                "status": run.status,
                "error": run.error,
                "summary": {k: v for k, v in summary.items() if k != "cases"},
            }

        @app.get("/runs/{run_id}/cases")
        async def get_run_cases(run_id: str) -> dict:
            run = runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            return {"id": run.id, "cases": (run.summary or {}).get("cases", [])}

        @app.post("/runs/{run_id}/cancel")
        async def cancel_run(run_id: str) -> dict:
            run = runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            if run.task and not run.task.done():
                run.task.cancel()
            return {"id": run.id, "status": "cancelling"}

        return app

    app = create_app()
else:  # pragma: no cover
    app = None

    def create_app():  # type: ignore[misc]
        raise RuntimeError(
            "FastAPI is required for the API: pip install 'phonebot-qa[api]'"
        )
