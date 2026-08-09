# Architecture

This document describes how a single test case flows through the platform and
how the pieces fit together. It mirrors the concept document
([`CONCEPT.md`](CONCEPT.md)) and the code under `phonebot_qa/`.

## The data flow of one test case

```
                 Orchestrator (generate_cases)                 §6
                            │  Scenario × Persona × Seed × Mode × Bot Version
                            ▼
                    ┌───────────────┐
                    │ ConversationRunner │                     §7
                    └───────┬───────┘
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   UserSimulator      BotAdapter          EventLog + Clock     §8 / §11 / §18
 (KnowledgeView only) (reference bot)   (deterministic time)
          │                 │  tool calls
          │                 ▼
          │           ToolProxy ──► FaultInjector ──► World    §16 / §17 / §15
          │           (logs ToolCall,      (mock backend state)
          │            emits events)
          └────────────► transcript ◄───────────┘
                            │
                            ▼  RunArtifacts (conversation, events, tool_calls, final_state)
                    ┌───────────────┐
                    │ EvaluationPipeline │                     §19
                    └───────┬───────┘
   1. business  2. safety  3. tool   assertions (deterministic)  §20
   4. technical metrics (latency)
   5. LLM judge (soft)                                          §21
   6. voice metrics (voice mode)                                §14
                            │
                            ▼
                    compute_score  +  critical-failure override  §27
                            │
                            ▼
                     CaseResult (PASS / FAIL / ERROR + score)   §26
                            │
             ┌──────────────┴───────────────┐
             ▼                              ▼
      RunSummary → release_gate       RegressionStore.capture   §29 / §24
        (baseline vs candidate)        (on failure → replay)
```

## Key design decisions

### Deterministic truth first (§3, §37)

The PASS/FAIL verdict is decided **only** by deterministic assertions over the
backend `World` state, the recorded `ToolCall`s and the structured `EventLog`.
The LLM judge and voice metrics contribute to the numeric *score* but a failed
*critical* assertion overrides everything and forces FAIL
(`evaluation/scoring.py::first_critical_failure`). A beautiful conversation that
booked the wrong slot still fails.

### Knowledge isolation (§10)

The user simulator only ever receives a
`scenario.knowledge.KnowledgeView` — a frozen projection containing the caller's
goal, persona and `user_visible` knowledge. The scenario's `expected` block
(required tool calls, events, DB state) is *structurally absent* from that view,
so a mis-written or LLM-driven simulator cannot cheat by reading the answer.

### Determinism / reproducibility (§24)

No component reads wall-clock time during a run. A logical
`observability.Clock` advances only when a component reports elapsed time (tool
latency, bot "thinking" time), and all randomness is seeded from the case
`seed`. The result: the same `(scenario, persona, seed)` produces a byte-for-byte
identical transcript, event stream and score — which is exactly what makes a
regression case replayable.

### Everything swappable except the IP (§36)

The bot (`BotAdapter`), the caller (`UserSimulator` / `LLMProvider`) and the
judge (`Judge`) are all interfaces with dependency-free default implementations,
so the platform runs with no API keys and no external services. The durable
value — scenarios, business rules, safety invariants and the regression store —
lives in this repo as data and deterministic code, independent of any LLM vendor
or OSS tool.

## Voice mode (Phase 3, §12.2/§13/§14/§34)

Voice reuses the *entire* pipeline above and swaps only the runner. The scenario,
the assertions, the regression store and the release gate are unchanged — voice
simply adds failure modes on top of the business contract:

```
 UserSimulator (text)                                    §8
        │
        ▼
   TTS ──► AudioChaos ──► Transport ──► VoiceBotAdapter          §34 / §13 / §11
 (caller)  (SNR, noise,   (loopback/    ┌──────────────┐
           speed, volume,  SIP/WebRTC)  │ STT          │         §12.2
           packet loss)                 │  ↓           │
                                        │ text bot ────┼──► ToolProxy ──► World
                                        │  ↓           │        §16          §15
                                        │ TTS          │
                                        └──────┬───────┘
                                               │ bot audio
                                    BargeInController                     §14
                                    (interrupt, measure stop latency)
                                               │
                                               ▼
                              VoiceMetrics: WER · barge-in · SLA          §14/§18
                              latency_breakdown: stt/llm/tool/tts         §18
```

Two properties make this trustworthy:

**Determinism.** No wall-clock, no real randomness: TTS is synthesized from the
text, chaos and recognition are seeded from the case seed, and the transport's
jitter/loss RNG is seeded too. The same `(scenario, profile, seed)` reproduces
byte-for-byte — which is what lets a noisy voice failure become a replayable
regression case.

**Honest recognition.** The utterance rides with the audio, but the STT simulator
never returns it verbatim: it derives an error probability from the *measured*
degradation (SNR, packet loss, rate, gain, clipping) and corrupts words
accordingly, then reports a real edit-distance WER. So `clean` transcribes
cleanly and `restaurant` genuinely breaks calls. Swapping in Whisper or Deepgram
behind `STTEngine` changes nothing else.

**Response latency**, not wall-time, is what voice tests measure: from the caller
falling silent to the bot starting to speak. Caller speech and bot playback count
toward call *duration*, never toward responsiveness.

## Discovery & the self-growing loop (Phase 2, §23/§35)

```
BusinessRules ──► probes ──┐
                           ├──► run ──► evaluate ──► violation? ──► Finding
seed scenarios ──► variants┘                                          │
  · persona swap (expectations unchanged)                             ▼
  · slow backend (expectations unchanged)                     RegressionCase
  · write_fault  (expectations REWRITTEN to the no-change contract)   │
                                                                      ▼
Production call ──► ProductionTrace ──► scenario ──────────────► permanent suite
```

Generated variants must be **sound**: a generated case may never fail for a
reason the bot is not responsible for. Persona and latency mutations therefore
keep the original expectations, while the `write_fault` mutation rewrites them to
the unchanged-state contract (record keeps its original value, the write event
becomes forbidden, no false success). That single mutation gives every happy-path
scenario a matching backend-failure test for free.

## Module map

| Layer | Package |
|---|---|
| Domain models (the shared contract) | `phonebot_qa/models.py` |
| Scenario engine + knowledge isolation | `phonebot_qa/scenario/` |
| User simulation + personas | `phonebot_qa/simulator/` |
| Conversation execution | `phonebot_qa/runner/` |
| Bot adapters (REST + reference bot) | `phonebot_qa/adapters/bot/` |
| Mock backend, tool proxy, faults | `phonebot_qa/backend/` |
| Event log + logical clock | `phonebot_qa/observability/` |
| Evaluation (assertions, judge, voice, scoring) | `phonebot_qa/evaluation/` |
| Red teaming | `phonebot_qa/redteam/` |
| Automatic test discovery | `phonebot_qa/discovery/` |
| Regression store | `phonebot_qa/regression/` |
| Production trace ingestion | `phonebot_qa/production/` |
| Persistence (§25 data model) | `phonebot_qa/persistence/` |
| Promptfoo / DeepEval integrations | `phonebot_qa/integrations/` |
| Audio: buffer, TTS, STT, chaos, barge-in | `phonebot_qa/audio/` |
| Voice transports (loopback/SIP/WebRTC) | `phonebot_qa/adapters/transport/` |
| Voice runner | `phonebot_qa/runner/voice.py` |
| Orchestrator, gate, workers, API | `phonebot_qa/orchestrator/` |
| CLI | `phonebot_qa/cli.py` |
