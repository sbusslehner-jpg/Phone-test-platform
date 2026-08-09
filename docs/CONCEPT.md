# Technisches Konzept: Automatisierte Testplattform für Phonebots

> This is the original design concept this repository implements. The code maps
> to these sections as documented in [`CONCEPT_MAPPING.md`](CONCEPT_MAPPING.md).

## 1. Ziel

Ziel ist der Aufbau einer automatisierten QA- und Red-Team-Plattform für
Phonebots bzw. Voice Agents. Die Plattform soll reale Anrufer automatisiert
simulieren, Multi-Turn-Gespräche durchführen, unterschiedliche Personas und
Nutzerverhalten testen, Fehler- und Grenzfälle automatisch erzeugen, Backend-
und Tool-Aufrufe überprüfen, Voice-spezifische Probleme testen, Gesprächsqualität
bewerten, Regressionstests erzeugen, Releases automatisiert freigeben oder
blockieren und bestehende Open-Source-Komponenten verwenden.

Langfristig entsteht damit eine Art **CI/CD-System für Voice Agents**.

## 2. Grundprinzip

```
CI / Developer / Scheduler → Test Orchestrator
  → Scenario/Regression DB + Test Generator (User Simulator / Red-Team Agent)
  → Conversation Runner (TEXT MODE | VOICE MODE)
  → Phonebot under test (CRM / Calendar / APIs)
  → Event / Trace Log
  → Hard Assertions + LLM Evaluation + Voice Metrics
  → PASS / FAIL / SCORE → CI Release Gate + Dashboard
```

## 3. Architekturprinzipien

Strikte Trennung von: Szenariodefinition, User Simulation, Conversation
Execution, Bot-Anbindung, Backend-Zustand, Event Logging, deterministische
Assertions, qualitative LLM-Evaluation, Voice-Evaluation, Regression Testing.

Insbesondere sollte ein LLM **nicht** die alleinige Entscheidung über PASS oder
FAIL treffen. Die primäre Wahrheit sind Backend-Zustände, Tool Calls, Business
Rules und technische Events. LLMs werden hauptsächlich für Simulation, Variation,
Red Teaming und qualitative Bewertungen eingesetzt.

## 4. Repository-Struktur

```
phonebot-test-platform/
├── orchestrator/   ├── scenarios/   ├── simulator/   ├── adapters/
├── audio/          ├── evaluation/  ├── mocks/       ├── observability/
├── dashboard/      └── docker-compose.yml
```

## 5. Scenario Engine

Das Szenario ist die zentrale Abstraktion. Es definiert Ausgangszustand,
Nutzerziel, Persona, erlaubtes Wissen des Nutzers, erwarteten Backend-Zustand,
notwendige Aktionen, verbotene Aktionen und technische Limits. **Die
Scenario-Datei ist die Source of Truth.**

## 6. Test Orchestrator

Stack: Python, FastAPI, PostgreSQL, Redis, Celery/Dramatiq.
API: `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/cases`, `POST /runs/{id}/cancel`.
Der Orchestrator erzeugt Test Cases als Kreuzprodukt:
`Scenario × Persona × Noise Profile × User Behaviour × Seed × Bot Version`.

## 7. Conversation Runner

Führt das Gespräch aus und kontrolliert Session, Turn Count, Timeout, Transport,
Logging und Abbruchbedingungen.

## 8. User Simulator

Erhält Persona + Ziel + Wissen + Conversation History und erzeugt die nächste
Nutzerantwort. Hinter einem Interface abstrahiert (DeepEval / LLM / Local Model /
Scripted), damit die Plattform unabhängig von einem bestimmten LLM bleibt.

## 9. Personas

Beispiele: normal, ungeduldig, verwirrt, redselig, unsicher, älterer Nutzer,
sehr kurze Antworten, nicht technikaffin, schlechter Deutschsprecher,
widersprüchlicher Nutzer. Persona und fachliches Szenario sind unabhängig
kombinierbar.

## 10. Knowledge Isolation

Der Simulator darf nur Informationen kennen, die ein echter Nutzer kennen würde
(`user_visible_context`), niemals `evaluator_only_context` (erwartete Tool Calls,
Business Rules). Dadurch wird Evaluation Leakage verhindert.

## 11. Bot Adapter

Interface: `startSession`, `sendText`, `sendAudio`, `stopSession`.
Implementierungen: REST, WebSocket, SIP, WebRTC. Damit testet dieselbe Testsuite
verschiedene Phonebots.

## 12. Zwei Testebenen

**Text Tests** (schnell, günstig, parallelisierbar, reproduzierbar) — der
Großteil der Testfälle. **Voice End-to-End Tests** prüfen zusätzlich STT, TTS,
Turn Detection, Barge-in, Silence Detection, Latenz, Audioqualität,
Telefonieverhalten.

## 13. Audio Chaos Layer

Zwischen User-TTS und Phonebot: FFmpeg / SoX / WebRTC Audio Processing.
Testprofile: clean, street, office, car, restaurant, station, wind,
bad_connection, low_volume, fast_speaker, slow_speaker.

## 14. Barge-in Testing

Gemessen: interrupt_start, barge_in_detected, bot_audio_stop, user_stt_start,
bot_response_start. Mögliche SLA: Barge-in Detection < 300 ms.

## 15. Backend Test Environment

Tests laufen nicht gegen Produktion. Vor dem Test: `POST /test/setup`, danach
Prüfung via `GET /test/state`. Mocks für CRM, Calendar, Customer, Payment.

## 16. Tool Proxy

Alle Tool Calls laufen über ein Test Gateway (Logging + Fault Injection), was
deterministische Assertions ermöglicht.

## 17. Fault Injection

Timeout, HTTP 500, HTTP 429, Connection Reset, Malformed Response, Partial
Response, hohe Latenz, Service unavailable. **Wichtige Invariante: Der Bot darf
niemals behaupten, eine Aktion sei erfolgreich gewesen, wenn das Backend diese
Aktion nicht bestätigt hat.**

## 18. Event Logging

Strukturierte Events (user_audio_started … bot_audio_finished) erlauben exakte
Latenz-Zerlegung (STT / LLM / Tool / TTS). Tracing via OpenTelemetry.

## 19. Evaluation Pipeline

Reihenfolge: 1. Business Assertions, 2. Safety Assertions, 3. Tool Assertions,
4. Technical Metrics, 5. LLM Evaluation, 6. Voice Evaluation.

## 20. Deterministische Assertions

z.B. `appointment.date == expected_date`, `confirmation_received is True`,
`count_tool_calls("calendar.update") == 1`, `unauthorized_data_access is False`.
Diese entscheiden über kritische PASS/FAIL-Kriterien.

## 21. LLM Evaluation

DeepEval o.ä. bewertet Natürlichkeit, Verständlichkeit, Effizienz,
Korrektur-Umgang usw. — aber nicht allein über Business-Korrektheit.

## 22. Red Teaming

Promptfoo für Prompt Injection, Authorization, PII Leakage, Unexpected Input,
Security, Manipulation, Policy Violations.

## 23. Automatische Test Discovery

Business Rules → Red-Team Agent → Conversation Generation → Phonebot →
Evaluator → Violation? → Finding → Regression Case.

## 24. Regression Testing

Jeder echte Fehler wird zum Regression Case (Scenario, Persona, Seed, Audio-,
Fault- und Bot-Config, Expected State/Events, Forbidden Events).

## 25. Datenmodell

bots, bot_versions, scenarios, scenario_versions, personas, test_suites, runs,
run_cases, conversations, turns, tool_calls, events, assertion_results,
eval_results, findings, regression_cases.

## 26. Test Result

JSON mit scenario, bot_version, result, business/safety/conversation/voice
scores, metrics (turns, duration, latencies) und assertions.

## 27. Scoring

Business 40 %, Safety 30 %, Conversation 15 %, Latency 10 %, Voice 5 %.
**Critical Failures überschreiben den Score ⇒ FAIL** (z.B. Termin ohne
Bestätigung gebucht, Kundendaten geleakt, falscher Kunde verändert,
Backend-Fehler als Erfolg kommuniziert).

## 28. CI/CD Integration

PR: schnelle Texttests (Core + Regression, 500–2.000 Gespräche). Staging: +
Personas + Fault Injection + Red Team (5.000–20.000). Nightly: Full Regression +
Voice + Noise + Fuzzing + Red Team.

## 29. Release Gate

`candidate_success >= baseline_success AND critical_failures == 0 AND
regression_failures == 0` ⇒ Deployment möglich, sonst blockiert.

## 30. Open-Source-Komponenten

Promptfoo (Red Teaming), DeepEval (Evaluation / Simulation), EVA-Bench
(Bot-to-Bot Voice), VAmoS Bench (Scenario → Seed DB → Simulated Caller → Voice
Agent → Tools → DB → Trace + Assertions → PASS/FAIL). Austauschbar hinter
eigenen Interfaces.

## 31. Technologie-Stack

Python + FastAPI, Celery/Dramatiq, Redis, PostgreSQL, S3/MinIO, YAML, DeepEval,
Promptfoo, FFmpeg/SoX, SIP/WebRTC, OpenTelemetry, Prometheus, Grafana/Next.js,
Docker/Kubernetes.

## 32. MVP

Scenario YAML → User Simulator → Conversation Runner → Bot Adapter → Mock
Backend → Tool/Event Logging → Business Assertions → DeepEval → Test Report.

## 33. Phase 2

Promptfoo Red Teaming, Fault Injection, Regression Management, CI/CD Release
Gates, Production Failure → Regression Case.

## 34. Phase 3

TTS → Audio Chaos → SIP/WebRTC → Phonebot → Voice Metrics.

## 35. Langfristiges Zielbild

Selbstwachsendes Testsystem: Production → Auffälliger Call → Failure Analysis →
Scenario Generation → Regression Case → Test Suite → Bot Änderung → Candidate
Evaluation → PASS ⇒ Deploy / FAIL ⇒ Block.

## 36. Kern-IP der Plattform

Langfristig wertvoll: Scenario Engine, Bot Adapter, Tool Proxy, Backend
Fixtures, Business Assertions, Regression Store. Promptfoo/DeepEval/LLM-Provider
sind ersetzbar; die Szenarien, Business-Regeln, Regression Cases und
historischen Testresultate sind das eigentliche Wissen.

## 37. Wichtigster Designgrundsatz

> LLMs erzeugen Variationen, simulieren Nutzer und bewerten weiche
> Gesprächsqualität. **Deterministische Systeme entscheiden über
> Business-Korrektheit.**

Ein Test prüft nicht nur *"Klang die Antwort richtig?"*, sondern: Wurde das
richtige Tool aufgerufen? Mit den richtigen Parametern? Hat sich der richtige
Backend-Zustand geändert? Wurde die Bestätigung eingeholt? Wurden alle
Sicherheitsregeln eingehalten? Wurde ein technischer Fehler korrekt behandelt?
