# Concept → Code mapping

How each section of [`CONCEPT.md`](CONCEPT.md) is realised in this repository.
Status: ✅ implemented · 🟡 partial / scaffolded · ⬜ future phase.

**All 37 concept sections are now implemented** (Phase 1 + Phase 2 + Phase 3).

| § | Concept topic | Status | Where |
|---|---|---|---|
| 1 | Goal: automated QA & red-team platform | ✅ | whole repo |
| 2 | Overall architecture | ✅ | `docs/ARCHITECTURE.md` |
| 3 | Strict component separation | ✅ | package layout under `phonebot_qa/` |
| 4 | Repository structure | ✅ | `phonebot_qa/{scenario,simulator,adapters,backend,evaluation,...}` |
| 5 | Scenario engine (source of truth) | ✅ | `scenario/loader.py`, `models.Scenario`, `scenarios/*.yaml` |
| 6 | Test orchestrator (API + case gen) | ✅ | `orchestrator/{api,generator,engine}.py` |
| 7 | Conversation runner | ✅ | `runner/conversation.py` |
| 8 | User simulator (behind interface) | ✅ | `simulator/{base,scripted,heuristic,llm}.py` |
| 9 | Personas | ✅ | `simulator/personas.py`, `personas/*.yaml` |
| 10 | Knowledge isolation | ✅ | `scenario/knowledge.py` |
| 11 | Bot adapter (REST/WS/SIP/WebRTC) | ✅ | `adapters/bot/{base,rest,reference,voice}.py`, `adapters/transport/{loopback,sip,webrtc}.py` |
| 12 | Two test levels (text / voice) | ✅ | `runner/conversation.py` + `runner/voice.py` — same scenarios, both modes |
| 13 | Audio chaos layer | ✅ | `audio/chaos.py`, `audio/profiles.py` (11 named profiles) |
| 14 | Barge-in testing | ✅ | `audio/bargein.py` + critical `voice:barge_in_*` assertions |
| 15 | Backend test environment | ✅ | `backend/world.py` (seed/state, no production) |
| 16 | Tool proxy (logging) | ✅ | `backend/proxy.py` |
| 17 | Fault injection | ✅ | `backend/faults.py` + always-on "no false success" invariant |
| 18 | Event logging + latency decomposition | ✅ | `observability/events.py`, `evaluation/pipeline.py` |
| 19 | Evaluation pipeline order | ✅ | `evaluation/pipeline.py` |
| 20 | Deterministic assertions | ✅ | `evaluation/assertions.py` |
| 21 | LLM evaluation (judge) | ✅ | `evaluation/judge.py` (interface + heuristic + LLM slot) |
| 22 | Red teaming | ✅ | `redteam/attacks.py` + `integrations/promptfoo.py` (config out, findings in) |
| 23 | Automatic test discovery | ✅ | `discovery/{rules,variants,engine}.py` — rules + variants → findings |
| 24 | Regression testing | ✅ | `regression/store.py`, `cli.py replay` |
| 25 | Data model | ✅ | `models.py` (runtime) + `persistence/schema.py` (all 16 tables) |
| 26 | Test result | ✅ | `models.CaseResult`, `--json` report |
| 27 | Scoring + critical override | ✅ | `evaluation/scoring.py` |
| 28 | CI/CD integration | ✅ | `cli.py` exit codes, `--junit`, suites |
| 29 | Release gate | ✅ | `orchestrator/gate.py`, `cli.py gate` |
| 30 | OSS components (Promptfoo/DeepEval/…) | ✅ | `integrations/{promptfoo,deepeval}.py`, both optional |
| 31 | Technology stack | ✅ | FastAPI/Pydantic/YAML/SQLAlchemy; `orchestrator/workers.py` (Celery/Dramatiq) |
| 32 | MVP | ✅ | end-to-end, `phonebot-qa run` |
| 33 | Phase 2 | ✅ | Promptfoo, fault injection, regression mgmt, gates, persistence, workers |
| 34 | Phase 3 (voice) | ✅ | `audio/*` + `runner/voice.py` + `adapters/transport/*` |
| 35 | Self-growing test system | ✅ | capture → replay demonstrated (`README` quickstart) |
| 36 | Core IP | ✅ | scenario engine, adapters, tool proxy, fixtures, assertions, regression store |
| 37 | Core design principle | ✅ | deterministic verdict + soft LLM score, enforced in scoring |
