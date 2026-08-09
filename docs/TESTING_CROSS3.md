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

## Zwei ehrliche Einschränkungen

- **Azure OpenAI:** Der Chat-Kanal ruft echtes Azure OpenAI, das LLM variiert.
  Die **deterministischen** Backend/Tool-Assertions bleiben belastbar; die
  weichen LLM-Judge-Scores und exakten Wortlaute schwanken. Für stabile CI:
  mehrere Seeds + Schwellwerte statt Einzel-Lauf.
- **Skriptete Anrufer:** Die `redteam_lines` sind ein realistischer Start; gegen
  den laufenden Agenten ggf. an dessen konkrete Rückfragen anpassen. Für freiere
  Gespräche einen `LLMSimulator` mit eigenem Provider einsetzen
  (`phonebot_qa/simulator/llm.py`).
