# Concept → Code mapping

How each section of [`CONCEPT.md`](CONCEPT.md) is realised in this repository.
Status: ✅ implemented · 🟡 partial / scaffolded · ⬜ future phase.

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
| 11 | Bot adapter (REST/WS/SIP/WebRTC) | 🟡 | `adapters/bot/{base,rest,reference}.py` (REST + in-process; SIP/WebRTC = Phase 3) |
| 12 | Two test levels (text / voice) | 🟡 | text fully; voice mode plumbed, metrics stubbed |
| 13 | Audio chaos layer | ⬜ | Phase 3 (FFmpeg/SoX) |
| 14 | Barge-in testing | 🟡 | `evaluation/voice.py` (SLA check + metrics model) |
| 15 | Backend test environment | ✅ | `backend/world.py` (seed/state, no production) |
| 16 | Tool proxy (logging) | ✅ | `backend/proxy.py` |
| 17 | Fault injection | ✅ | `backend/faults.py` + always-on "no false success" invariant |
| 18 | Event logging + latency decomposition | ✅ | `observability/events.py`, `evaluation/pipeline.py` |
| 19 | Evaluation pipeline order | ✅ | `evaluation/pipeline.py` |
| 20 | Deterministic assertions | ✅ | `evaluation/assertions.py` |
| 21 | LLM evaluation (judge) | ✅ | `evaluation/judge.py` (interface + heuristic + LLM slot) |
| 22 | Red teaming | ✅ | `redteam/attacks.py` (Promptfoo = Phase-2 plug-in) |
| 23 | Automatic test discovery | 🟡 | red-team scenarios + regression capture form the loop |
| 24 | Regression testing | ✅ | `regression/store.py`, `cli.py replay` |
| 25 | Data model | ✅ | `models.py` (runtime), maps to the listed tables |
| 26 | Test result | ✅ | `models.CaseResult`, `--json` report |
| 27 | Scoring + critical override | ✅ | `evaluation/scoring.py` |
| 28 | CI/CD integration | ✅ | `cli.py` exit codes, `--junit`, suites |
| 29 | Release gate | ✅ | `orchestrator/gate.py`, `cli.py gate` |
| 30 | OSS components (Promptfoo/DeepEval/…) | 🟡 | abstracted behind `Judge`/`UserSimulator`/`LLMProvider` |
| 31 | Technology stack | 🟡 | FastAPI/Pydantic/YAML now; Postgres/Redis/Celery = `docker-compose.yml` `infra` profile |
| 32 | MVP | ✅ | end-to-end, `phonebot-qa run` |
| 33 | Phase 2 | 🟡 | fault injection, regression mgmt, gates done; Promptfoo/workers next |
| 34 | Phase 3 (voice) | ⬜ | model + event slots ready |
| 35 | Self-growing test system | ✅ | capture → replay demonstrated (`README` quickstart) |
| 36 | Core IP | ✅ | scenario engine, adapters, tool proxy, fixtures, assertions, regression store |
| 37 | Core design principle | ✅ | deterministic verdict + soft LLM score, enforced in scoring |
