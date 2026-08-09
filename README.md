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

## What's in the box

**All 37 concept sections are implemented** — Phase 1 (MVP), Phase 2 and
Phase 3. Everything runs with **no API keys, no brokers and no audio hardware**:
the platform ships an in-process reference phonebot as the system under test,
and every external piece (LLM simulator, judge, TTS, STT, telephony, broker,
database) sits behind a swappable interface with a deterministic default.

**Phase 1 — the core loop (§32)**

| Capability | Module | Concept § |
|---|---|---|
| Scenario engine (YAML, source of truth) | `phonebot_qa/scenario/` | 5, 10 |
| Knowledge isolation (no evaluation leakage) | `scenario/knowledge.py` | 10 |
| User simulator (scripted / heuristic / LLM) | `phonebot_qa/simulator/` | 8, 9 |
| Conversation runner | `phonebot_qa/runner/conversation.py` | 7 |
| Bot adapter (REST + in-process reference) | `phonebot_qa/adapters/bot/` | 11 |
| Mock backend, tool proxy, fault injection | `phonebot_qa/backend/` | 15–17 |
| Structured event log + logical clock | `phonebot_qa/observability/` | 16, 18 |
| Deterministic assertions | `evaluation/assertions.py` | 20 |
| LLM-as-a-judge (abstracted) | `evaluation/judge.py` | 21 |
| Weighted scoring + critical override | `evaluation/scoring.py` | 27 |
| Orchestrator, CI release gate, API, CLI | `phonebot_qa/orchestrator/`, `cli.py` | 6, 28, 29 |

**Phase 2 — the growing suite (§33)**

| Capability | Module | Concept § |
|---|---|---|
| Red teaming (built-in attacks) | `phonebot_qa/redteam/` | 22 |
| **Promptfoo integration** (config out, findings in) | `integrations/promptfoo.py` | 22, 30 |
| **DeepEval judge** behind the `Judge` interface | `integrations/deepeval.py` | 21, 30 |
| **Automatic test discovery** (rules + variants → findings) | `phonebot_qa/discovery/` | 23 |
| Regression store (capture + replay) | `phonebot_qa/regression/` | 24 |
| **Production call → regression case** | `phonebot_qa/production/` | 35 |
| **Persistence** (all 16 tables, SQLite/Postgres) | `phonebot_qa/persistence/` | 25, 31 |
| **Worker/queue layer** (Celery/Dramatiq) | `orchestrator/workers.py` | 31 |

**Phase 3 — voice end-to-end (§34)**

| Capability | Module | Concept § |
|---|---|---|
| **TTS / STT engines** (interfaces + deterministic simulators) | `audio/tts.py`, `audio/stt.py` | 12.2, 34 |
| **Audio chaos layer** (SNR, noise, speed, volume, packet loss) | `audio/chaos.py` | 13 |
| **11 named noise profiles** (street, car, restaurant, …) | `audio/profiles.py` | 13 |
| **Barge-in testing** + 300 ms SLA | `audio/bargein.py` | 14 |
| **SIP / WebRTC / loopback transports** | `adapters/transport/` | 11, 34 |
| **Voice runner** — the same scenarios as real calls | `runner/voice.py` | 12.2 |
| **Voice metrics**: WER, barge-in, latency decomposition | `evaluation/voice.py` | 14, 18 |

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

Promptfoo runs the same attack vectors as an external red-team engine, and its
findings flow back into the regression pipeline:

```bash
phonebot-qa promptfoo --out promptfoo/          # generate config + provider bridge
npx promptfoo@latest redteam run -c promptfoo/promptfooconfig.yaml -o results.json
phonebot-qa promptfoo --import-results results.json
```

### Automatic test discovery (concept §23)

Discovery goes beyond the written scenarios: it probes the declared business
rules and mutates each seed scenario along axes that break real bots — including
auto-generating a *backend-fault* variant of every happy path.

```bash
phonebot-qa discover --suite booking --capture-regressions scenarios/regression/cases
```

```
Discovery: explored 30 generated case(s)
------------------------------------------------------------------------------
  [critical] move_appointment_001__write_fault
             safety:no_false_success: bot reported success although the backend …
  [critical] book_appointment_001__write_fault
             safety:no_false_success: bot reported success although the backend …
------------------------------------------------------------------------------
  5 violation(s) found
  captured 5 regression case(s)
```

A healthy bot reports **0 violations** across the same 30 generated cases.

### Voice end-to-end (concept §12.2, §13, §14, §34)

The *same* scenarios run as real phone calls — TTS → noise → SIP/WebRTC → the
bot's STT — and the *same* business assertions decide PASS/FAIL:

```bash
phonebot-qa voice --suite voice
phonebot-qa voice --suite booking --profiles clean street car restaurant   # noise sweep
```

A noise sweep is the point: it turns "is the bot robust?" into a number.

| profile | passed | mean WER | avg turns |
|---|---|---|---|
| clean | 12/12 | 0.00 | 2.0 |
| office | 12/12 | 0.06 | 2.0 |
| street | 12/12 | 0.22 | 2.4 |
| car | 12/12 | 0.25 | 2.3 |
| restaurant | 8/12 | 0.47 | 2.8 |
| station | 9/12 | 0.52 | 3.3 |

Under moderate noise the bot recovers (turn count rises as the caller corrects a
misheard read-back); under heavy noise calls genuinely fail. Barge-in is a
first-class, *critical* assertion (concept §14):

```yaml
audio:
  profile: clean
  transport: webrtc
  barge_in: { interrupt_after_ms: 850 }
  barge_in_sla_ms: 300
```

```
voice: {barge_in_detected: true, stop_latency_ms: 180, user_audio_lost_ms: 180}
```

A bot that talks over its caller, or detects the interruption too late, fails.

### Production failure → regression case (concept §35)

```bash
phonebot-qa ingest production_call.json --store scenarios/regression/cases
phonebot-qa replay --store scenarios/regression/cases
```

The trace's pre-call backend state becomes the scenario's `initial_state` and the
caller's real utterances become the script, so the bug reproduces exactly and is
then guarded forever.

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
pip install -e ".[api,db,dev]"
pytest -q            # 98 tests, ~16s, fully deterministic, no network
```

Optional extras: `api` (FastAPI + httpx), `db` (SQLAlchemy/Postgres persistence),
`deepeval` (LLM judge), `llm` (Anthropic provider for the LLM simulator).

## Status

- **Phase 1 (§32)** — scenarios → simulate → run → mock backend → tool/event
  logging → assertions → judge → report. ✅
- **Phase 2 (§33)** — Promptfoo red teaming, fault injection, automatic test
  discovery, regression management, CI release gates, persistence, workers. ✅
- **Phase 3 (§34)** — TTS → audio chaos → SIP/WebRTC → voice metrics, running
  the same text scenarios as real phone calls. ✅

All 37 concept sections are implemented — see
[`docs/CONCEPT_MAPPING.md`](docs/CONCEPT_MAPPING.md).

Natural next steps beyond the concept: swapping the simulated TTS/STT for real
engines (the interfaces are already in place), a Grafana/Next.js dashboard over
the persisted results, and OpenTelemetry export of the event trace.

## License

MIT
