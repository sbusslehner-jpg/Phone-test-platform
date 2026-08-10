# CROSS3 mit der Plattform testen

Runbook zum Testen des [`cross3-dms-agent`](https://github.com/sbusslehner-jpg/cross3-dms-agent)
mit dieser Plattform. CROSS3 wird über den `Cross3Adapter` angebunden
(`phonebot_qa/adapters/bot/cross3.py`) — er fährt den Text-Kanal `POST /api/chat`,
der denselben Agent-Kern (Tools, Executor, Prompt) nutzt wie Telefon/Voice.

## Warum der Text-Kanal

Der Chat-Kanal ist deterministisch, braucht kein Audio und testet die *echte*
Geschäftslogik. Er liefert `toolEvents` gleich mit, wodurch die deterministischen
Tool- und Backend-Assertions greifen. Voice (SIP/Realtime) kommt später über den
`SIPTransport` obendrauf.

## Was welche Stufe prüft

| Stufe | Voraussetzung | Assertions |
|---|---|---|
| 1 Gesprächsqualität | nur der Adapter | LLM-Judge, Effizienz, Latenz, Persona |
| 2 Tool-Calls | `toolEvents` (hat CROSS3) | `required_events`, `forbidden_events`, `tool_call_counts` |
| 3 Backend-State | in-memory Mocks + Fault-Hook | `expected.database`, `no_pii_leak`, `false_success_claim` |

## Setup

1. **CROSS3 im Testmodus starten** (im `cross3-dms-agent`-Repo):
   ```bash
   cp .env.example .env
   # .env: DEMO_MODE=true, AZURE_OPENAI_* setzen (Chat ruft echtes Azure OpenAI),
   # AZURE_STORAGE_CONNECTION_STRING leer (in-memory), CROSS_/SBO_API_BASE_URL leer (Mocks)
   npm ci && npm start          # → http://127.0.0.1:8080
   ```
   Prüfen: `curl localhost:8080/healthz` → `{"storage":true,"tenants":>0,"mocks":true,...}`.

2. **Plattform installieren** (dieses Repo):
   ```bash
   pip install -e ".[api,dev]"
   ```

3. **Suite fahren:**
   ```python
   import asyncio
   from phonebot_qa.adapters.bot import Cross3Adapter
   from phonebot_qa.orchestrator import run_suite
   from phonebot_qa.scenario.loader import load_scenarios

   bot = Cross3Adapter("http://127.0.0.1:8080", version="cross3-2026-08")
   scns = load_scenarios("scenarios/cross3")
   summary = asyncio.run(run_suite(scns, bot=bot, seeds=[0, 1, 2]))
   print(summary.pass_rate, [f.critical_failure for f in summary.failures()])
   ```

## Szenarien (Tenant AT997, Autohaus-Senker-Fixtures)

- `cross3_book_pickerl_001` — bekannter Kunde (Max Mustermann, `+436601234567`) bucht Pickerl.
- `cross3_redteam_cross_customer_001` — nicht verifizierter Anrufer will Fremddaten/‑buchung → prüft „Kundendaten nur aus verifiziertem Anruferkontext" + `no_pii_leak`.
- `cross3_fault_book_timeout_001` — das Buchungs-Backend bestätigt NICHT (`bookingConfirmed: false` ⇒ `terminStatus: "angefragt"`) → der Bot darf keine Fix-Zusage machen (§17), **und** der Fault muss konsumiert worden sein (`fault_must_fire`). **Braucht den Fault-Hook** (siehe unten).
- Seit M3 zusätzlich die **19 P1-Szenarien der Fall-Matrix** (`docs/fall-matrix.json`):
  Storno/Verschiebung/Auskunft, Verifikations-Fehlpfade (Verweigerung, Recovery,
  Bruteforce-Sperre), Fremd-Storno/fremde Nummer, Slot-Race (409), Doppelbuchung/
  Idempotenz, Auflegen mitten im Flow, CRM-Lookup fail-closed, Neukunden-Buchung,
  Scan-Onboarding (2×, „braucht M2"-Scan-Driver), Voice-Barge-in gegen die echte
  Bridge, Prompt-Injection-Storno und Storno-Fault. Details und M2-Marker stehen
  als Kommentar in den jeweiligen YAMLs.

Eigene Szenarien: Anruferzeilen in `user.user_visible.redteam_lines` (skriptet),
`tenant_id`/`caller_phone` in `initial_state`, Assertions über CROSS3-Tool-Namen
(`sbo_get_slots`, `sbo_book`, `sbo_cancel`, `sbo_get_my_appointments`, `cross_*`).
Wichtig für den Skript-Zuschnitt: CROSS3 fragt beim Buchen zuerst nach
Zusatzservices und nennt dann den frühesten Slot; Verifikation = PLZ **oder**
letzte 4 FIN-Zeichen; Sperre nach 5 Fehlversuchen.

## M3-Härtung: Degradation, Fault-Konsum, State-Reset, Wall-Latenz

Der QA-Lauf gegen CROSS3 (Fall-Matrix) hat drei Blindstellen aufgedeckt; die
Plattform prüft sie jetzt selbst:

**P1 — Degradations-Erkennung.** Bricht der Azure-Call ab, antwortet CROSS3 mit
„Entschuldigung, ich habe gerade ein technisches Problem …". Der Adapter (bzw.
bot-agnostisch `RunnerConfig.fallback_patterns`, regex-fähig; Default-Muster in
`phonebot_qa/degradation.py`) markiert solche Antworten als `Turn.degraded`,
der Trace erhält `bot_degraded`. Die kritische Assertion `technical:not_degraded`
FAILt, sobald mehr degradierte Turns auftreten als `expected.max_degraded_turns`
erlaubt (Default 0) — ein totes Gespräch kann ein nur-forbidden-Szenario nicht
mehr vakuum-trivial bestehen. `report.json` weist `degraded_turns` je Case aus.

**P2 — Fault-Konsum.** `expected.fault_must_fire: true` erzwingt per Assertion
`fault:consumed`, dass der scharfgeschaltete Fault wirklich gezündet hat
(je nach Ziel: SBO-Call auf dem Fault-Pfad mit *Backend*-Fehler, `sbo_book`
mit `terminStatus: "angefragt"` beim Ziel `service-booking`, oder ein
fail-closed abgebrochenes Tool beim Ziel `customer` ⇒ Event `fault_fired`; die
Entwaffnen-Antwort wird zusätzlich auf einen „war noch scharf"-Vorzustand
geprüft ⇒ Event `fault_not_consumed`). `stop_session` entwaffnet einen noch
scharfen Fault IMMER (leerer Body an den Hook), auch nach einem Crash — der
Runner ruft `stop_session` seit M3 im `finally`.

**P3 — State-Reset pro Case.** `initial_state.cross3_reset: true` (oder
`Cross3Adapter(reset_state=True)` als Default) setzt den Tenant vor
`start_session` neu auf: Konfiguration lesen (`GET /api/admin/tenants`),
`POST /api/admin/tenants/:id/wipe` (löscht die Daten-Partition **und** den
Tenant), Tenant identisch neu anlegen. Buchungen kontaminieren keine
Folge-Cases mehr; die Code-Fixtures des DMS-Mocks bleiben unberührt. Achtung:
Szenarien, die einen VORBESTEHENDEN Termin brauchen (Storno/Verschiebung),
dürfen `cross3_reset` nur zusammen mit `cross3_seed` setzen (siehe unten).

**P4 — Seed pro Case.** `initial_state.cross3_seed` (ein Objekt oder eine Liste)
stellt den Ausgangszustand über `POST /api/admin/test-seed` her — NACH dem
Reset. `{art: "termin", telefon, kunde, fahrzeug: {kennzeichen}, services}`
legt einen vorbestehenden Termin an, `{art: "keine_slots", tage}` bucht den
Terminraster leer. **Das Kennzeichen ist Pflicht und muss das Fahrzeug des
erkannten Anrufers sein**: `sbo_get_my_appointments` filtert die Termine über
die VIN des Anrufer-Fahrzeugs — ein Termin ohne bekanntes Fahrzeug ist für den
Bot unsichtbar. Ein fehlgeschlagener Seed macht den Case zum ERROR statt still
mit falscher Voraussetzung zu laufen.

**P5 — Testisolation der Verifikationssperre.** CROSS3 zählt Verifikations-
*Fehlversuche* prozessweit pro Betrieb+Anrufer (`server/platform/http-guard.mjs`,
5 Versuche / 15 min). Weder `cross3_reset` noch die Case-Grenze leeren diesen
Zähler. Szenarien, die absichtlich Budget verbrennen, tragen deshalb den Tag
`verify_lockout` **und eine im Bestand einzigartige `caller_phone`**
(`tests/test_fall_matrix_scenarios.py` erzwingt das).

**Wall-Latenz.** Jeder Turn trägt zusätzlich echte Wanduhr-Millisekunden
(`Turn.wall_latency_ms`, `LatencyMetrics.wall_avg/wall_p95_latency_ms`,
`wall_p95_latency_ms` im Report) — die deterministische logische Clock
(`bot_think_ms=400`) bleibt unverändert.

## Fault Injection (CROSS3-seitiger Hook)

Für Stufe-3-Fault-Tests nutzt der Adapter den admin-geschützten Hook
`POST /api/admin/test-fault` (Bearer-Token aus `CROSS3_ADMIN_TOKEN`; nur im
Mock-Modus aktiv). Der Hook kennt drei Ziele (`server/routes/admin.mjs`):

| `api` | Body | trifft |
|---|---|---|
| `sbo` (Default) | `{path?, mode: timeout\|500\|401\|409, once?}` | SBO v1 — u. a. `/appointment/cancel`, `/appointment/detail` |
| `service-booking` | `{bookingConfirmed: false, sendOrderConfirmation?, once?}` | Premium Service Booking V1 — der **Buchungs-Schreibpfad** |
| `customer` | `{mode: timeout\|500\|401, once?}` | Customer-V2-Lookup (`/customers/pageable-search`) — Anrufererkennung |

Seit dem API-Umbau schreibt `sbo_book` über `createServiceBooking`; ein
SBO-Fault auf `/appointment/book` trifft die Buchung **nicht** mehr. Ein harter
Ausfall des Schreibpfads (timeout/500/409) ist derzeit gar nicht injizierbar —
der Service-Booking-Mock kennt nur den Ergebnis-Override. Leerer Body
entwaffnet alle Stellen. Der Adapter schaltet
automatisch scharf, wenn ein Szenario `initial_state.cross3_fault` deklariert,
validiert die Hook-Antwort und entwaffnet in `stop_session`. Ohne den Hook ist
ein Fault-Szenario nicht aussagekräftig — mit `fault_must_fire` schlägt es dann
hart fehl statt still durchzulaufen.

## Voice-Pfad (Telefon)

CROSS3s Telefon-Kanal ist eine **Voll-Duplex-Media-Bridge**:

```
PSTN → Peoplefone → Asterisk + Relay ──WSS /ws/phone-media──▶ App ──▶ Azure Realtime
```

Das Asterisk-Relay ist bewusst „dumm": es authentifiziert per `x-relay-token`,
schickt einen `{"type":"start", did, callerId, callUuid, format:"slin"}`-Frame
und streamt dann **binäres SLIN-Audio** (8 kHz, 16-bit LE mono) in beide
Richtungen. Steuer-Frames App→Relay: `{"type":"clear"}` bei Barge-in,
`{"type":"hangup", reason}` am Ende; Relay→App `{"type":"stop"}`.

Die Plattform testet Voice, indem sie **selbst das Relay spielt** —
`Cross3VoicePhoneClient` (`adapters/transport/cross3_phone.py`) spricht genau
dieses Protokoll. Damit läuft CROSS3s echte Voice-Bridge (Media, Azure Realtime
STT/TTS, semantic-VAD Turn-Detection, Barge-in) ohne PSTN/Asterisk. Der Client
ist gegen einen Fake-`/ws/phone-media`-Server verifiziert
(`tests/test_cross3_phone_client.py`) — ohne CROSS3 und ohne Azure.

```python
import asyncio
from phonebot_qa.adapters.transport import Cross3VoicePhoneClient
from phonebot_qa.audio import DeterministicTTS, apply_chaos, get_profile

async def call():
    tts = DeterministicTTS()   # synthesize(..., sample_rate=8000) → SLIN-Rate
    client = Cross3VoicePhoneClient("ws://127.0.0.1:8080", relay_token="<RELAY_TOKEN>")
    await client.connect(did="<AT997-DID>", caller_id="+436601234567")
    greeting = await client.next_bot_turn()          # Agent grüßt zuerst
    caller = apply_chaos(tts.synthesize("Ich möchte einen Pickerl-Termin.", sample_rate=8000),
                         get_profile("bad_connection"), seed=1)
    await client.send_audio(caller)
    reply = await client.next_bot_turn()             # Bot-Audio (SLIN 8k)
    # reply.to_buffer() → an einen echten STTEngine geben, um zu scoren
    await client.close()
asyncio.run(call())
```

Voraussetzungen: `pip install 'phonebot-qa[voice]'`, `RELAY_TOKEN` in CROSS3
gesetzt, ein Azure-**Realtime**-Deployment, und die Tenant-`did` (DID→Tenant via
`resolveTenantByDid`).

### Was Voice testet — und was nicht

| | Chat-Kanal | Voice-Kanal |
|---|---|---|
| Tool-/Backend-Assertions | ✅ (toolEvents) | ⚠️ nicht über den Relay — Buchung nachträglich im SBO-Mock lesen |
| STT/TTS/Turn-Taking | – | ✅ |
| Barge-in + SLA (<300 ms) | – | ✅ (`clear`-Frame) |
| Hangup/Weiterleitung | – | ✅ (`hangup`/`transfer`) |

Voice **ergänzt** Chat, ersetzt es nicht: die deterministischen
Business/Tool-Assertions laufen weiter über den Chat-Kanal (der Voll-Duplex-Relay
leitet keine Tool-Calls durch — die laufen intern in der Bridge). Ob eine per
Voice ausgelöste Buchung wirklich gelandet ist, prüfst du durch einen Lese-Aufruf
auf den SBO-Mock nach dem Anruf.

### Voll gescorter Lauf: `Cross3VoiceRunner`

Der turn-basierte Runner (`phonebot_qa/runner/cross3_voice.py`) verdrahtet den
Client in die bestehende Evaluation: skripteter Caller → TTS (8 kHz) → Audio-Chaos
→ CROSS3 → Bot-Audio zurück, und erzeugt dieselben `RunArtifacts` wie der
Text-Pfad. Damit greifen Antwortlatenz (§18), Barge-in gegen die 300-ms-SLA (§14),
Hangup und — mit echtem STT — Gesprächsqualität. Backend-Wahrheit kommt aus einem
injizierten `state_reader` (liest den SBO-Mock nach dem Anruf).

```python
import asyncio
from phonebot_qa.adapters.transport import Cross3VoicePhoneClient
from phonebot_qa.runner import run_cross3_voice_suite
from phonebot_qa.scenario.loader import load_scenarios

def factory():
    return Cross3VoicePhoneClient("ws://127.0.0.1:8080", relay_token="<RELAY_TOKEN>")

async def read_sbo(client):        # optional: Backend-State nach dem Anruf lesen
    ...                            # GET /mock/sbo/.../appointment/detail → {"appointments": [...]}
    return {}

scns = load_scenarios("scenarios/cross3")   # Szenarien mit audio.did/barge_in
summary = asyncio.run(run_cross3_voice_suite(
    scns, factory, stt=None, state_reader=read_sbo, seeds=[0, 1, 2]))
print(summary.pass_rate)
```

Voice-Szenarien deklarieren die Anbindung im `initial_state` (`did`, `caller_phone`)
und Barge-in im `audio`-Block (`barge_in: {interrupt_after_ms}`, `barge_in_sla_ms`).

Zwei ehrliche Grenzen des Voice-Pfads bleiben:
- **STT zum Scoren:** was CROSS3 *gesagt* hat, kennt die Plattform nur über einen
  **echten** `STTEngine` auf dem Bot-Audio. Ohne STT bleibt das Bot-Transkript
  leer — Barge-in/Latenz/Hangup scoren trotzdem, Gesprächsqualität nicht.
- **Turn-Modell:** der Relay-Stream ist voll-duplex; der Runner schneidet Turns
  über eine Stille-Lücke (VAD-Näherung). Für sehr überlappende Dialoge ist das
  eine Vereinfachung gegenüber dem echten kontinuierlichen Stream.

## Zwei ehrliche Einschränkungen (Chat)

- **Azure OpenAI:** Der Chat-Kanal ruft echtes Azure OpenAI, das LLM variiert.
  Die **deterministischen** Backend/Tool-Assertions bleiben belastbar; die
  weichen LLM-Judge-Scores und exakten Wortlaute schwanken. Für stabile CI:
  mehrere Seeds + Schwellwerte statt Einzel-Lauf.
- **Skriptete Anrufer:** Die `redteam_lines` sind ein realistischer Start; gegen
  den laufenden Agenten ggf. an dessen konkrete Rückfragen anpassen. Für freiere
  Gespräche einen `LLMSimulator` mit eigenem Provider einsetzen
  (`phonebot_qa/simulator/llm.py`).
