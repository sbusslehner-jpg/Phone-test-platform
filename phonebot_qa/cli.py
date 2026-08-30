"""Command-line interface — the developer/CI entry point (concept sections 28-29).

Commands::

    phonebot-qa run       # run a suite, print a report, exit non-zero on failure
    phonebot-qa redteam   # run the built-in red-team suite
    phonebot-qa gate      # baseline vs candidate release gate
    phonebot-qa replay    # re-run stored regression cases
    phonebot-qa list      # list scenarios in a directory
    phonebot-qa serve     # run the FastAPI orchestrator (needs [api] extra)

``run``/``redteam``/``gate``/``replay`` exit with status 1 on any failure so they
plug straight into CI as a release gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .adapters.bot.reference import BotBehavior, ReferenceAppointmentBot
from .orchestrator.engine import RunEngine, RunSummary, summarize
from .orchestrator.gate import release_gate
from .orchestrator.generator import TestCase, generate_cases
from .redteam.attacks import redteam_scenarios
from .regression.store import RegressionStore
from .scenario.loader import load_personas, load_scenarios
from .simulator.personas import resolve_persona

# Named bot versions -> behaviour. Lets the gate/replay demos model real bug
# fixes and regressions (sections 24 & 29) with the in-process reference bot.
_BOT_BEHAVIORS = {
    "buggy-corrections": BotBehavior(handle_corrections=False),
    "buggy-fault": BotBehavior(report_success_on_fault=True),
    "no-confirm": BotBehavior(confirm_before_write=False),
}


def build_bot(version: str, args=None):
    """Bot unter Test aufbauen.

    Standard ist der eingebaute Referenz-Bot (deterministisch, offline). Mit
    ``--bot cross3`` wird stattdessen der echte CROSS3-Serviceagent über seinen
    Text-Kanal gefahren — derselbe Agent-Kern wie Telefon und WhatsApp.
    """
    ziel = getattr(args, "bot", "reference") if args is not None else "reference"
    if ziel == "cross3":
        from .adapters.bot import Cross3Adapter

        return Cross3Adapter(
            base_url=getattr(args, "base_url", None) or os.environ.get("CROSS3_BASE_URL", "http://127.0.0.1:8080"),
            tenant_id=getattr(args, "tenant", None) or os.environ.get("CROSS3_TENANT", "senker"),
            version=version,
        )
    return ReferenceAppointmentBot(version=version, behavior=_BOT_BEHAVIORS.get(version))


def _load_suite(suite: str, scenarios_dir: str):
    if suite == "redteam":
        return redteam_scenarios()
    root = Path(scenarios_dir)
    if suite in ("all", ""):
        scenarios = load_scenarios(root)
    else:
        sub = root / suite
        # Unknown suite (no matching subdirectory) returns empty so the caller
        # errors loudly, rather than silently broadening to the whole suite.
        scenarios = load_scenarios(sub) if sub.exists() else []
    if suite in ("all", "staging"):
        scenarios = scenarios + redteam_scenarios()
    return scenarios


def _load_personas(personas_dir: str):
    root = Path(personas_dir)
    return load_personas(root) if root.exists() else {}


VALID_MODES = ("text", "voice")


def _voice_overrides(args) -> dict:
    """Only the voice fields the operator explicitly asked for.

    Returning a sparse dict (rather than a fully-populated config) is what keeps
    a ``--profiles`` sweep from silently replacing a scenario's declared
    transport, or a ``--transport`` flag from resetting its noise profile.
    """
    return {
        "profile": getattr(args, "audio_profile", None),
        "transport": getattr(args, "transport", None),
    }


def _validate_modes(modes) -> list[str]:
    bad = [m for m in modes if m not in VALID_MODES]
    if bad:
        raise SystemExit(
            f"unknown mode(s) {', '.join(bad)}; valid modes: {', '.join(VALID_MODES)}"
        )
    return list(modes)


async def _run_summary(args, scenarios, bot) -> RunSummary:
    personas = _load_personas(args.personas_dir)
    seeds = args.seeds or list(range(args.iterations))
    cases = generate_cases(
        scenarios,
        personas=args.persona,
        seeds=seeds,
        modes=_validate_modes(args.modes),
        bot_version=bot.version,
    )
    engine = RunEngine(
        bot,
        personas=personas,
        concurrency=args.concurrency,
        voice_overrides=_voice_overrides(args),
    )
    results = await engine.run_cases(cases)
    return summarize(results, bot.version)


# --------------------------------------------------------------------------- #
# Report rendering                                                             #
# --------------------------------------------------------------------------- #

_GREEN, _RED, _YELLOW, _DIM, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _color(result: str) -> str:
    if not sys.stdout.isatty():
        return result
    return {
        "PASS": f"{_GREEN}PASS{_RESET}",
        "FAIL": f"{_RED}FAIL{_RESET}",
        "ERROR": f"{_YELLOW}ERROR{_RESET}",
    }.get(result, result)


def _print_summary(summary: RunSummary) -> None:
    print(f"\nBot version: {summary.bot_version}")
    print("-" * 78)
    for r in summary.results:
        line = f"  {_color(r.result):<6} {r.scenario_id}  [{r.persona_id or 'default'}/seed{r.seed}/{r.mode}]  score={r.score.total:.2f}"
        print(line)
        if r.critical_failure:
            print(f"         └─ {r.critical_failure}")
    print("-" * 78)
    print(
        f"  {summary.passed}/{summary.total} passed  "
        f"({summary.pass_rate*100:.1f}%)  |  "
        f"fail={summary.failed} error={summary.errored} critical={summary.critical_failures}"
    )
    print(
        f"  avg_score={summary.avg_score:.3f}  avg_turns={summary.avg_turns:.1f}  "
        f"p95_latency={summary.p95_latency_ms:.0f}ms"
    )


def _write_junit(summary: RunSummary, path: str) -> None:
    """Emit a minimal JUnit XML report for CI (section 28)."""
    import xml.etree.ElementTree as ET

    suite = ET.Element(
        "testsuite",
        name="phonebot-qa",
        tests=str(summary.total),
        failures=str(summary.failed),
        errors=str(summary.errored),
    )
    for r in summary.results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=r.scenario_id,
            name=f"{r.persona_id or 'default'}-seed{r.seed}-{r.mode}",
            time=f"{r.latency.duration_seconds:.3f}",
        )
        if r.result == "FAIL":
            ET.SubElement(case, "failure", message=r.critical_failure or "failed").text = (
                r.critical_failure or ""
            )
        elif r.result == "ERROR":
            ET.SubElement(case, "error", message=r.error or "error").text = r.error or ""
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _maybe_capture_regressions(summary: RunSummary, args, scenarios) -> None:
    if not getattr(args, "capture_regressions", None):
        return
    by_id = {s.id: s for s in scenarios}
    personas = _load_personas(args.personas_dir)
    store = RegressionStore(args.capture_regressions)
    n = 0
    for r in summary.failures():
        scenario = by_id.get(r.scenario_id)
        if scenario is None:
            continue
        # Freeze the exact persona too, so the case is fully self-contained.
        persona = resolve_persona(r.persona_id, personas)
        store.capture(r, scenario, created_at=args.now, persona=persona)
        n += 1
    if n:
        print(f"\nCaptured {n} regression case(s) into {args.capture_regressions}")


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #


def cmd_run(args) -> int:
    scenarios = _load_suite(args.suite, args.scenarios_dir)
    if not scenarios:
        print(f"No scenarios found for suite {args.suite!r}", file=sys.stderr)
        return 2
    bot = build_bot(args.bot_version, args)
    summary = asyncio.run(_run_summary(args, scenarios, bot))
    _print_summary(summary)
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary.to_report(), indent=2, ensure_ascii=False), "utf-8"
        )
        print(f"\nWrote JSON report to {args.json}")
    if args.junit:
        _write_junit(summary, args.junit)
        print(f"Wrote JUnit report to {args.junit}")
    _maybe_capture_regressions(summary, args, scenarios)
    return 0 if summary.critical_failures == 0 and summary.errored == 0 else 1


def cmd_redteam(args) -> int:
    args.suite = "redteam"
    return cmd_run(args)


def cmd_gate(args) -> int:
    scenarios = _load_suite(args.suite, args.scenarios_dir)
    if not scenarios:
        print(f"No scenarios found for suite {args.suite!r}", file=sys.stderr)
        return 2
    baseline_bot = build_bot(args.baseline_version, args)
    candidate_bot = build_bot(args.candidate_version, args)
    baseline = asyncio.run(_run_summary(args, scenarios, baseline_bot))
    candidate = asyncio.run(_run_summary(args, scenarios, candidate_bot))
    result = release_gate(candidate, baseline, min_success_delta=args.min_delta)

    print("\nRelease gate — baseline vs candidate")
    print("=" * 78)
    b, c = result.baseline, result.candidate
    print(f"  {'Metric':<22}{'Baseline':>18}{'Candidate':>18}")
    print(f"  {'Task success':<22}{b.task_success:>17.1%}{c.task_success:>18.1%}")
    print(f"  {'Critical errors':<22}{b.critical_errors:>18}{c.critical_errors:>18}")
    print(f"  {'Avg. turns':<22}{b.avg_turns:>18.1f}{c.avg_turns:>18.1f}")
    print(f"  {'P95 latency (ms)':<22}{b.p95_latency_ms:>18.0f}{c.p95_latency_ms:>18.0f}")
    print("=" * 78)
    verdict = _GREEN + "PASS → deployment possible" + _RESET if result.deployable else _RED + "FAIL → deployment blocked" + _RESET
    if not sys.stdout.isatty():
        verdict = "PASS → deployment possible" if result.deployable else "FAIL → deployment blocked"
    print(f"  {verdict}")
    for reason in result.reasons:
        print(f"    - {reason}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(result.to_report(), indent=2, ensure_ascii=False), "utf-8"
        )
    return 0 if result.deployable else 1


def cmd_replay(args) -> int:
    store = RegressionStore(args.store)
    cases_meta = store.load_all()
    if not cases_meta:
        print(f"No regression cases in {args.store}", file=sys.stderr)
        return 2
    bot = build_bot(args.bot_version, args)
    personas = dict(_load_personas(args.personas_dir))
    # Replay each case with its EXACT captured seed, persona and mode — not the
    # CLI defaults — so seed/persona-specific failures actually reproduce.
    test_cases: list[TestCase] = []
    for c in cases_meta:
        scenario = c.to_scenario()
        persona = c.to_persona()
        if persona is not None:
            personas[persona.id] = persona  # frozen snapshot wins
            persona_id = persona.id
        else:
            persona_id = c.persona_id
        test_cases.append(
            TestCase(
                case_id=c.id,
                scenario=scenario,
                persona_id=persona_id,
                seed=c.seed,
                mode=c.mode,
                bot_version=bot.version,
            )
        )
    engine = RunEngine(bot, personas=personas, concurrency=args.concurrency)
    results = asyncio.run(engine.run_cases(test_cases))
    summary = summarize(results, bot.version)
    _print_summary(summary)
    return 0 if summary.critical_failures == 0 and summary.errored == 0 else 1


def cmd_list(args) -> int:
    scenarios = _load_suite(args.suite, args.scenarios_dir)
    for s in scenarios:
        tags = f" [{', '.join(s.tags)}]" if s.tags else ""
        print(f"  {s.id}{tags}  — {s.description or s.user.goal.type}")
    print(f"\n{len(scenarios)} scenario(s)")
    return 0


def cmd_voice(args) -> int:
    """Run a suite as real voice calls, optionally sweeping noise profiles."""
    scenarios = _load_suite(args.suite, args.scenarios_dir)
    if not scenarios:
        print(f"No scenarios found for suite {args.suite!r}", file=sys.stderr)
        return 2
    bot = build_bot(args.bot_version, args)
    args.modes = ["voice"]
    # ``None`` means "leave each scenario's own profile alone" (no override).
    profiles = args.profiles or [args.audio_profile]
    failures = 0
    reports = []
    for profile in profiles:
        args.audio_profile = profile
        summary = asyncio.run(_run_summary(args, scenarios, bot))
        wers = [
            r.voice.stt_wer
            for r in summary.results
            if r.voice and r.voice.stt_wer is not None
        ]
        mean_wer = sum(wers) / len(wers) if wers else 0.0
        print(f"\n── audio profile: {profile or 'per-scenario'} ──")
        _print_summary(summary)
        print(f"  mean WER={mean_wer:.2f}")
        failures += summary.critical_failures + summary.errored
        report = summary.to_report()
        report["audio_profile"] = profile
        report["mean_wer"] = round(mean_wer, 4)
        reports.append(report)
        _maybe_capture_regressions(summary, args, scenarios)
        if args.junit:
            # One file per profile so a sweep does not overwrite itself.
            path = args.junit if len(profiles) == 1 else f"{args.junit}.{profile or 'default'}"
            _write_junit(summary, path)
            print(f"  wrote JUnit report to {path}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(reports if len(reports) > 1 else reports[0], indent=2, ensure_ascii=False),
            "utf-8",
        )
        print(f"\nWrote JSON report to {args.json}")
    return 0 if failures == 0 else 1


def cmd_cross3_voice(args) -> int:
    """Fahre Szenarien als ECHTE Anrufe gegen CROSS3s Telefon-Relay.

    Warum das ein eigener Unterbefehl ist und nicht ``run --modes voice``:
    ``--bot cross3`` baut den ``Cross3Adapter`` — den TEXT-Kanal. Ein als
    ``voice`` markiertes Szenario lief deshalb bisher still ueber Text, mit
    einem Ergebnis, das nach Telefon aussah und keines war. Der Telefonpfad
    haengt nicht am BotAdapter-Vertrag (Anfrage rein, Antwort raus), sondern an
    einem voll-duplexen Medienstrom mit Kontrollrahmen; ``Cross3VoiceRunner``
    ist dafuer da, war aber an nichts angeschlossen ausser an seine Tests.

    Ohne ``--sbo-country/--sbo-dealer`` laeuft der Lauf ohne Backend-Wahrheit:
    Barge-in, Latenz, Hangup und Transfer scoren dann, eine Buchung nicht.
    """
    from .adapters.transport import Cross3VoicePhoneClient
    from .runner import run_cross3_voice_suite

    scenarios = _load_suite(args.suite, args.scenarios_dir)
    if args.only_voice_tags:
        scenarios = [s for s in scenarios if "voice" in (s.tags or [])]
    if not scenarios:
        print(f"No voice scenarios found for suite {args.suite!r}", file=sys.stderr)
        return 2

    ws_url = args.ws_url or os.environ.get("CROSS3_WS_URL")
    if not ws_url:
        base = args.base_url or os.environ.get("CROSS3_BASE_URL", "http://127.0.0.1:8080")
        ws_url = base.replace("https://", "wss://").replace("http://", "ws://")
    token = args.relay_token or os.environ.get("RELAY_TOKEN", "")
    if not token:
        print("Kein Relay-Token (--relay-token oder RELAY_TOKEN)", file=sys.stderr)
        return 2

    def factory():
        return Cross3VoicePhoneClient(ws_url, relay_token=token)

    summary = asyncio.run(
        run_cross3_voice_suite(
            scenarios,
            factory,
            state_reader=_sbo_state_reader(args),
            pre_case=_cross3_pre_case(args),
            personas=_load_personas(args.personas_dir),
            seeds=args.seeds or list(range(args.iterations)),
            quiet_ms=args.quiet_ms,
        )
    )
    _print_summary(summary)
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary.to_report(), indent=2, ensure_ascii=False), "utf-8"
        )
        print(f"\nWrote JSON report to {args.json}")
    if args.junit:
        _write_junit(summary, args.junit)
        print(f"Wrote JUnit report to {args.junit}")
    return 0 if summary.critical_failures == 0 and summary.errored == 0 else 1


def _cross3_pre_case(args):
    """``cross3_reset``/``cross3_seed``/``cross3_fault`` auch am Telefon.

    Dieselbe Vorbereitung wie im Text-Kanal, ueber dieselbe Methode
    (``Cross3Adapter.prepare_case``) — sonst driften die beiden Pfade
    auseinander und ein Szenario bedeutet je nach Kanal etwas anderes.
    """
    token = args.admin_token or os.environ.get("ADMIN_TOKEN", "")
    if not token:
        return None
    from .adapters.bot import Cross3Adapter

    adapter = Cross3Adapter(
        base_url=args.base_url or os.environ.get("CROSS3_BASE_URL", "http://127.0.0.1:8080"),
        tenant_id=args.tenant or os.environ.get("CROSS3_TENANT", "senker"),
        version="cross3-voice",
        admin_token=token,
    )

    async def vorbereiten(state):
        await adapter.prepare_case(state, str(state.get("tenant_id") or adapter.default_tenant))

    return vorbereiten


def _sbo_state_reader(args):
    """Backend-Wahrheit nach dem Anruf: was liegt wirklich im Bestand?

    Der Relay leitet keine Werkzeug-Aufrufe durch — sie laufen in der Bruecke.
    Ob eine per Sprache ausgeloeste Buchung gelandet ist, sieht man deshalb nur
    hinterher. Gelesen wird ueber ``GET /api/admin/test-state``; die Mock-Routen
    selbst sind von aussen dicht (prozesslokales Token seit dem Sicherheits-
    Audit), und die Admin-Route gibt bewusst Wochentag und Uhrzeit statt Name
    und Adresse zurueck.

    ``None``, wenn kein Admin-Token vorliegt: lieber ein Lauf, der ehrlich keine
    Backend-Assertions hat, als einer, der gegen ein leeres Ergebnis prueft und
    gruen meldet.

    Neben den Terminen selbst (Schluessel ``termine.<appointmentId>``) legt der
    Leser einen Sammelsatz unter ``sbo_stand.stand`` ab. Nur der ist beim
    Schreiben eines Szenarios adressierbar: Die Termin-ID einer Buchung, die
    erst im Anruf entsteht, kennt niemand vorher.
    """
    token = args.admin_token or os.environ.get("ADMIN_TOKEN", "")
    if not token:
        return None
    base = (args.base_url or os.environ.get("CROSS3_BASE_URL", "http://127.0.0.1:8080")).rstrip("/")
    tenant = args.tenant or os.environ.get("CROSS3_TENANT", "senker")
    url = f"{base}/api/admin/test-state?tenantId={tenant}"

    async def read(_client):
        import urllib.request

        def hole():
            # CROSS3 weist Anfragen ohne Origin grundsaetzlich ab (403) — ein
            # Testlauf ist ein legitimer Erstanbieter-Aufruf.
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {token}", "Origin": base}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))

        try:
            stand = await asyncio.get_running_loop().run_in_executor(None, hole)
        except Exception as exc:  # noqa: BLE001 - Diagnose statt stillem Leerbefund
            print(f"  (Buchungs-Stand nicht lesbar: {exc})", file=sys.stderr)
            return {}
        termine = stand.get("termine", [])
        return {
            "termine": termine,
            "sbo_stand": [
                {
                    "id": "stand",
                    "gebucht": stand.get("gebucht", 0),
                    "storniert": stand.get("storniert", 0),
                    "wochentage": sorted({t.get("wochentag", "") for t in termine if t.get("wochentag")}),
                }
            ],
        }

    return read


def cmd_discover(args) -> int:
    """Explore beyond the written scenarios and report violations (§23)."""
    from .discovery import DiscoveryEngine

    seeds = _load_suite(args.suite, args.scenarios_dir) if args.suite else []
    bot = build_bot(args.bot_version, args)
    store = RegressionStore(args.capture_regressions) if args.capture_regressions else None
    engine = DiscoveryEngine(
        bot,
        personas=_load_personas(args.personas_dir),
        concurrency=args.concurrency,
        store=store,
    )
    report = asyncio.run(
        engine.discover(
            seeds=seeds,
            case_seeds=args.seeds or list(range(args.iterations)),
            capture_regressions=bool(args.capture_regressions),
            created_at=args.now,
        )
    )
    print(f"\nDiscovery: explored {report.explored} generated case(s)")
    print("-" * 78)
    for finding in report.findings:
        print(f"  [{finding.severity:8}] {finding.scenario_id}")
        print(f"             {finding.title[:88]}")
    print("-" * 78)
    print(f"  {report.violations} violation(s) found")
    if report.regression_case_ids:
        print(f"  captured {len(report.regression_case_ids)} regression case(s)")
    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_report(), indent=2, ensure_ascii=False), "utf-8"
        )
    return 0 if report.violations == 0 else 1


def cmd_promptfoo(args) -> int:
    """Generate a promptfoo red-team config, or import its results (§22/§30)."""
    from .integrations import findings_from_promptfoo, write_promptfoo_config

    if args.import_results:
        findings = findings_from_promptfoo(
            args.import_results, bot_version=args.bot_version, created_at=args.now
        )
        print(f"Imported {len(findings)} finding(s) from {args.import_results}")
        for f in findings:
            print(f"  [{f.severity}] {f.title[:80]} — {', '.join(f.failed_assertions)}")
        if args.json:
            Path(args.json).write_text(
                json.dumps([f.model_dump(mode="json") for f in findings], indent=2, ensure_ascii=False),
                "utf-8",
            )
        return 0 if not findings else 1

    built = write_promptfoo_config(args.out)
    print(f"Wrote {built.paths['config']}")
    print(f"Wrote {built.paths['provider']}")
    print("\nRun the red-team suite with:")
    print(f"  npx promptfoo@latest redteam run -c {built.paths['config']} -o results.json")
    print(f"  phonebot-qa promptfoo --import-results results.json")
    return 0


def cmd_ingest(args) -> int:
    """Turn a production call trace into a regression case (§35)."""
    from .production import ProductionTrace, regression_case_from_trace

    trace = ProductionTrace.from_file(args.trace)
    case = regression_case_from_trace(trace)
    store = RegressionStore(args.store)
    store.save(case)
    print(f"Created regression case {case.id}")
    print(f"  scenario: {case.scenario['id']}")
    print(f"  stored in: {args.store}")
    print(f"\nReplay it with:\n  phonebot-qa replay --store {args.store}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install 'phonebot-qa[api]'", file=sys.stderr)
        return 2
    uvicorn.run("phonebot_qa.orchestrator.api:app", host=args.host, port=args.port)
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #


def _add_run_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bot",
        default="reference",
        choices=["reference", "cross3"],
        help="Bot unter Test: eingebauter Referenz-Bot oder der echte CROSS3-Agent",
    )
    p.add_argument("--base-url", default=None, help="Basis-URL des CROSS3-Agenten (--bot cross3)")
    p.add_argument("--tenant", default=None, help="CROSS3-Betrieb, z.B. senker (--bot cross3)")
    p.add_argument("--scenarios-dir", default="scenarios", help="scenario root directory")
    p.add_argument("--personas-dir", default="personas", help="persona root directory")
    p.add_argument("--persona", action="append", default=None, help="restrict to persona id (repeatable)")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seeds")
    p.add_argument("--iterations", type=int, default=1, help="number of seeds if --seeds omitted")
    p.add_argument("--modes", nargs="*", default=["text"], help="modes: text voice")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--json", default=None, help="write full JSON report here")
    p.add_argument("--now", default=None, help="timestamp to stamp captured regressions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonebot-qa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a scenario suite")
    p_run.add_argument("--suite", default="all", help="suite: all|booking|cancellation|support|regression|redteam")
    p_run.add_argument("--bot-version", default="reference-1.0")
    p_run.add_argument("--junit", default=None, help="write JUnit XML report here")
    p_run.add_argument("--capture-regressions", default=None, help="store failing cases as regression cases here")
    _add_run_opts(p_run)
    p_run.set_defaults(func=cmd_run)

    p_rt = sub.add_parser("redteam", help="run the built-in red-team suite")
    p_rt.add_argument("--bot-version", default="reference-1.0")
    p_rt.add_argument("--junit", default=None)
    p_rt.add_argument("--capture-regressions", default=None)
    _add_run_opts(p_rt)
    p_rt.set_defaults(func=cmd_redteam)

    p_gate = sub.add_parser("gate", help="baseline vs candidate release gate")
    p_gate.add_argument("--suite", default="all")
    p_gate.add_argument("--baseline-version", default="reference-1.0")
    p_gate.add_argument("--candidate-version", default="reference-1.0")
    p_gate.add_argument("--min-delta", type=float, default=0.0, help="required success margin over baseline")
    _add_run_opts(p_gate)  # provides --json among the shared run options
    p_gate.set_defaults(func=cmd_gate)

    p_replay = sub.add_parser("replay", help="re-run stored regression cases")
    p_replay.add_argument("--store", default="scenarios/regression/cases")
    p_replay.add_argument("--bot-version", default="reference-1.0")
    _add_run_opts(p_replay)
    p_replay.set_defaults(func=cmd_replay)

    p_list = sub.add_parser("list", help="list scenarios")
    p_list.add_argument("--suite", default="all")
    p_list.add_argument("--scenarios-dir", default="scenarios")
    p_list.set_defaults(func=cmd_list)

    p_voice = sub.add_parser("voice", help="run a suite as real voice calls (phase 3)")
    p_voice.add_argument("--suite", default="voice", help="suite to run in voice mode")
    p_voice.add_argument("--bot-version", default="reference-1.0")
    p_voice.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="sweep these audio profiles (clean street car restaurant bad_connection ...)",
    )
    p_voice.add_argument("--audio-profile", default=None, help="single audio profile")
    p_voice.add_argument("--transport", default=None, choices=["loopback", "sip", "webrtc"])
    p_voice.add_argument("--junit", default=None)
    p_voice.add_argument("--capture-regressions", default=None)
    _add_run_opts(p_voice)
    p_voice.set_defaults(func=cmd_voice)

    p_c3v = sub.add_parser(
        "cross3-voice", help="Szenarien als echte Anrufe gegen CROSS3s Telefon-Relay fahren"
    )
    p_c3v.add_argument("--suite", default="cross3", help="Szenario-Unterordner")
    p_c3v.add_argument("--ws-url", default=None, help="WSS-Basis des Relays (Vorgabe: aus --base-url abgeleitet)")
    p_c3v.add_argument("--relay-token", default=None, help="x-relay-token (Vorgabe: $RELAY_TOKEN)")
    p_c3v.add_argument(
        "--admin-token",
        default=None,
        help="Admin-Token fuer den Buchungs-Stand nach dem Anruf (Vorgabe: $ADMIN_TOKEN); ohne ihn laeuft der Lauf ohne Backend-Assertions",
    )
    p_c3v.add_argument(
        "--all-scenarios",
        dest="only_voice_tags",
        action="store_false",
        default=True,
        help="auch Szenarien ohne voice-Tag anrufen (Vorgabe: nur voice)",
    )
    p_c3v.add_argument(
        "--quiet-ms",
        type=int,
        default=2000,
        help=(
            "Stille, nach der der Anrufer den Zug uebernimmt (Vorgabe 2000). "
            "Nicht kleiner setzen als die Werkzeug-Laufzeit: Der Bot sagt "
            "'ich schau, was frei ist' und schweigt dann, waehrend das DMS "
            "antwortet. Ein Mensch wartet das ab; ein zu ungeduldiger "
            "Testanrufer redet hinein und das Gespraech entgleist."
        ),
    )
    p_c3v.add_argument("--junit", default=None)
    _add_run_opts(p_c3v)
    p_c3v.set_defaults(func=cmd_cross3_voice)

    p_disc = sub.add_parser("discover", help="automatic test discovery (§23)")
    p_disc.add_argument("--suite", default="all", help="seed scenarios to mutate ('' for rules only)")
    p_disc.add_argument("--bot-version", default="reference-1.0")
    p_disc.add_argument("--capture-regressions", default=None)
    _add_run_opts(p_disc)
    p_disc.set_defaults(func=cmd_discover)

    p_pf = sub.add_parser("promptfoo", help="generate/import promptfoo red-team runs (§22)")
    p_pf.add_argument("--out", default="promptfoo", help="directory for the generated config")
    p_pf.add_argument("--import-results", default=None, help="promptfoo results JSON to import")
    p_pf.add_argument("--bot-version", default="reference-1.0")
    p_pf.add_argument("--json", default=None)
    p_pf.add_argument("--now", default=None)
    p_pf.set_defaults(func=cmd_promptfoo)

    p_ing = sub.add_parser("ingest", help="production call trace -> regression case (§35)")
    p_ing.add_argument("trace", help="path to the production trace JSON")
    p_ing.add_argument("--store", default="scenarios/regression/cases")
    p_ing.set_defaults(func=cmd_ingest)

    p_serve = sub.add_parser("serve", help="run the orchestrator API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
