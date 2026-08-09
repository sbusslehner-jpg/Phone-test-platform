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
- `cross3_fault_book_timeout_001` — SBO-Buchung faultet → der Bot darf keinen Erfolg melden (§17). **Braucht den Fault-Hook** (siehe unten).

Eigene Szenarien: Anruferzeilen in `user.user_visible.redteam_lines` (skriptet),
`tenant_id`/`caller_phone` in `initial_state`, Assertions über CROSS3-Tool-Namen
(`sbo_get_slots`, `sbo_book`, `sbo_cancel`, `sbo_get_my_appointments`, `cross_*`).

## Fault Injection (CROSS3-seitiger Hook)

Für Stufe-3-Fault-Tests braucht CROSS3 einen test-only Endpunkt, der einen
einmaligen Backend-Fehler scharfschaltet — der zugehörige PR fügt
`POST /mock/sbo/_test/fault` hinzu (nur aktiv wenn Mocks + DEMO_MODE an). Der
Adapter schaltet ihn automatisch scharf, wenn ein Szenario `initial_state.cross3_fault`
deklariert. Ohne den Hook ist das Fault-Szenario nicht aussagekräftig.

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
