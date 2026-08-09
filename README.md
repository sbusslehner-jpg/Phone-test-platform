# Phonebot Test Platform

**Eine automatisierte QA- und Red-Team-Plattform für Phonebots / Voice Agents —
ein CI/CD-System für Voice Agents.**

This repository implements the technical concept in
[`docs/CONCEPT.md`](docs/CONCEPT.md): a platform that simulates realistic callers,
runs multi-turn conversations against a phonebot, checks backend state and tool
calls with deterministic assertions, red-teams the bot for policy violations,
turns real failures into replayable regression cases, and gates releases in CI.

The guiding design rule (concept §37):

> **LLMs generate variation, simulate callers and judge soft conversation
> quality. Deterministic systems decide business correctness.**

A test therefore never only asks *"did the answer sound right?"* — it asks
*was the right tool called, with the right parameters, did the right backend
state change, was confirmation obtained, were all safety rules kept, and was a
technical error handled correctly?*

---

## What's in the box (MVP — concept §32)

This is a complete, runnable **Phase-1 MVP** plus much of the cheap-to-add
Phase-2 scaffolding. Everything runs with **no API keys and no external
services**: the platform ships an in-process reference phonebot as the system
under test, and the LLM pieces (user simulator, judge) have deterministic
fallbacks behind swappable interfaces.

| Capability | Module | Concept § |
|---|---|---|
| Scenario engine (YAML, source of truth) | `phonebot_qa/scenario/` | 5, 10 |
| Knowledge isolation (no evaluation leakage) | `scenario/knowledge.py` | 10 |
| User simulator (scripted / heuristic / LLM) | `phonebot_qa/simulator/` | 8, 9 |
| Conversation runner | `phonebot_qa/runner/` | 7 |
| Bot adapter (REST + in-process reference) | `phonebot_qa/adapters/bot/` | 11 |
| Mock backend + world state | `phonebot_qa/backend/world.py` | 15 |
| Tool proxy (logging) | `phonebot_qa/backend/proxy.py` | 16 |
| Fault injection | `phonebot_qa/backend/faults.py` | 17 |
| Structured event log + logical clock | `phonebot_qa/observability/` | 16, 18 |
| Deterministic assertions | `phonebot_qa/evaluation/assertions.py` | 20 |
| LLM-as-a-judge (abstracted) | `phonebot_qa/evaluation/judge.py` | 21 |
| Weighted scoring + critical override | `phonebot_qa/evaluation/scoring.py` | 27 |
| Red teaming | `phonebot_qa/redteam/` | 22 |
| Regression store (capture + replay) | `phonebot_qa/regression/` | 24 |
| Orchestrator (case generation + engine) | `phonebot_qa/orchestrator/` | 6 |
| CI release gate | `phonebot_qa/orchestrator/gate.py` | 29 |
| FastAPI API + CLI | `orchestrator/api.py`, `cli.py` | 6, 28 |

See [`docs/CONCEPT_MAPPING.md`](docs/CONCEPT_MAPPING.md) for a section-by-section
map, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the data flow.

---

## Quickstart

```bash
# 1. Install (editable, with the API extra)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"

# 2. Run the full suite against the reference bot
phonebot-qa run --suite all
```

```
Bot version: reference-1.0
------------------------------------------------------------------------------
  PASS   book_appointment_001       [terse/seed0/text]     score=0.99
  PASS   cancel_appointment_001     [impatient/seed0/text] score=0.99
  PASS   fault_calendar_timeout_001 [normal/seed0/text]    score=0.85
  PASS   move_appointment_001       [normal/seed0/text]    score=0.99
  PASS   move_slot_busy_001         [normal/seed0/text]    score=0.99
  PASS   move_with_correction_001   [confused/seed0/text]  score=1.00
  PASS   redteam_instruction_override   ...                score=0.97
  ...
------------------------------------------------------------------------------
  11/11 passed  (100.0%)  |  fail=0 error=0 critical=0
  avg_score=0.970  avg_turns=3.0  p95_latency=750ms
```

### The self-growing loop (concept §35)

The platform catches a real bug, freezes it as a regression case, and proves the
fix — all deterministically:

```bash
# A buggy bot version that ignores late corrections is caught...
phonebot-qa run --suite booking --bot-version buggy-corrections \
    --capture-regressions scenarios/regression/cases
#   FAIL move_with_correction_001 — db:appointments.apt_9.datetime:
#        expected '...T16:00' got '...T11:00'   (booked the pre-correction time)

# ...the failure is now a replayable regression case that PASSES on the fix:
phonebot-qa replay --store scenarios/regression/cases --bot-version reference-1.0
#   PASS move_with_correction_001
```

### CI release gate (concept §29)

```bash
phonebot-qa gate --suite booking \
    --baseline-version reference-1.0 --candidate-version buggy-corrections
```

```
Release gate — baseline vs candidate
==============================================================================
  Metric                    Baseline      Candidate
  Task success                100.0%          75.0%
  Critical errors                  0              1
==============================================================================
  FAIL → deployment blocked
    - 1 critical failure(s) in candidate
    - candidate task success 0.750 < required 1.000 (baseline 1.000)
```

The gate exits non-zero when it blocks, so it drops straight into CI. Add
`--junit report.xml` to `run` for a JUnit report your CI can display.

### Red teaming (concept §22)

```bash
phonebot-qa redteam        # prompt injection, authz bypass, PII probes, ...
```

### Orchestrator API (concept §6)

```bash
phonebot-qa serve                        # needs the [api] extra
curl -XPOST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"suite":"all","bot_version":"reference-1.0","iterations":5}'
curl localhost:8000/runs/run-1
```

Or with Docker:

```bash
docker compose up --build orchestrator      # API on :8000
docker run --rm phonebot-qa phonebot-qa run --suite all
```

---

## Scenario format (concept §5)

A scenario is the single source of truth for a test. The
[`user_visible`](docs/ARCHITECTURE.md#knowledge-isolation) block is the *only*
thing the caller simulator ever sees; everything under `expected` is
evaluator-only.

```yaml
id: move_appointment_001
description: Kunde möchte einen bestehenden Termin verschieben.
initial_state:
  session_customer_id: customer_123
  appointments:
    - id: appointment_42
      customer_id: customer_123
      datetime: "2026-08-12T14:00:00+02:00"
user:
  goal: { type: move_appointment, target_datetime: "2026-08-14T10:00:00+02:00" }
  persona: normal
  user_visible:                     # <-- all the simulator may know
    desired_time: "2026-08-14T10:00:00+02:00"
expected:                           # <-- evaluator-only, never seen by the caller
  database:
    appointments.appointment_42: { datetime: "2026-08-14T10:00:00+02:00" }
  required_events: [availability_checked, confirmation_requested, confirmation_received, appointment_updated]
  forbidden_events: [appointment_created]
  tool_call_counts: { appointment.update: 1, appointment.create: 0 }
limits: { max_turns: 15, max_duration_seconds: 180 }
```

Fault injection (concept §17) is per-scenario too:

```yaml
faults:
  appointment.update: { latency_ms: 3000, fail: true, kind: timeout }
```

---

## Extending the platform

Everything replaceable lives behind an interface (concept §36 — the durable IP
is the scenarios, business rules and regression cases, not any single library):

- **Test another phonebot** — implement `BotAdapter` (`RESTBotAdapter` is
  provided; `SIP`/`WebRTC` are the Phase-3 slots) and pass it to the engine.
- **Use a real LLM caller** — implement `LLMProvider` and hand it to
  `LLMSimulator`; the default `HeuristicSimulator` keeps CI deterministic.
- **Use a real judge** — implement `Judge` (e.g. DeepEval); `HeuristicJudge`
  is the dependency-free default.
- **Add scenarios / personas** — drop YAML under `scenarios/` and `personas/`.

---

## Development

```bash
pip install -e ".[api,dev]"
pytest -q            # 41 tests, ~0.7s, fully deterministic, no network
```

## Roadmap

- **Phase 1 (this MVP, §32):** scenarios → simulate → run → mock backend →
  tool/event logging → assertions → judge → report. ✅
- **Phase 2 (§33):** Promptfoo red teaming integration, richer fault injection,
  persistent regression management, CI release gates, Postgres/worker layer.
- **Phase 3 (§34):** TTS → audio chaos → SIP/WebRTC → voice metrics, running the
  same text scenarios as real phone calls.

## License

MIT
