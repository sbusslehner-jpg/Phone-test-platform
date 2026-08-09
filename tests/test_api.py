"""FastAPI orchestrator API smoke test (run lifecycle)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from phonebot_qa.orchestrator.api import create_app  # noqa: E402
from tests.conftest import SCENARIOS_DIR  # noqa: E402


async def test_run_lifecycle(monkeypatch):
    monkeypatch.setenv("PHONEBOT_SCENARIOS_DIR", str(SCENARIOS_DIR))
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.json()["status"] == "ok"

        created = await client.post("/runs", json={"suite": "booking", "iterations": 1})
        run_id = created.json()["id"]

        # Let the background task finish (yield to the loop).
        summary = None
        for _ in range(200):
            await asyncio.sleep(0.01)
            resp = await client.get(f"/runs/{run_id}")
            data = resp.json()
            if data["status"] in ("completed", "error"):
                summary = data
                break
        assert summary is not None and summary["status"] == "completed", summary
        assert summary["summary"]["total"] >= 1

        cases = await client.get(f"/runs/{run_id}/cases")
        assert len(cases.json()["cases"]) == summary["summary"]["total"]


async def test_unknown_run_404(monkeypatch):
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/runs/does-not-exist")
        assert resp.status_code == 404
