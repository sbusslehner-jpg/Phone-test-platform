"""Schema-Validierung der P1-Szenarien aus der Fall-Matrix (docs/fall-matrix.json).

Der Loader ist strikt (unbekannte Felder, kaputte ids, Typfehler ⇒ laut
scheitern) — dass ``load_scenarios`` alle CROSS3-Szenarien fehlerfrei lädt,
IST die Schema-Validierung. Dazu Struktur-Checks: die Kern-Härtungen (M3)
sind in den richtigen Szenarien verdrahtet.
"""

from __future__ import annotations

from phonebot_qa.scenario.loader import load_scenarios
from tests.conftest import REPO_ROOT

CROSS3_DIR = REPO_ROOT / "scenarios" / "cross3"

#: Die 19 P1-Szenarien der Fall-Matrix (Datei-/id-Konvention: Unterstriche).
P1_IDS = {
    "cross3_cancel_appointment_001",
    "cross3_move_appointment_001",
    "cross3_info_appointments_001",
    "cross3_verify_gate_no_disclosure_001",
    "cross3_verify_fail_then_recover_001",
    "cross3_verify_bruteforce_001",
    "cross3_known_wrong_number_001",
    "cross3_cancel_foreign_appointment_001",
    "cross3_no_slot_free_001",
    "cross3_slot_race_001",
    "cross3_double_booking_idempotent_001",
    "cross3_hangup_mid_flow_001",
    "cross3_lookup_failed_fail_closed_001",
    "cross3_unknown_caller_booking_001",
    "cross3_scan_happy_zulassungsschein_001",
    "cross3_scan_token_security_001",
    "cross3_voice_barge_in_slot_announce_001",
    "cross3_prompt_injection_storno_001",
    "cross3_cancel_fault_mid_flow_001",
}

#: Bereits vor M3 vorhandene CROSS3-Szenarien.
PRE_M3_IDS = {
    "cross3_book_pickerl_001",
    "cross3_fault_book_timeout_001",
    "cross3_redteam_cross_customer_001",
}

# Aus der fachlichen Klaerung mit dem Auftraggeber am 2026-08-10: Ersatzmobilitaet
# muss VOR der Slot-Ansage feststehen (tagesgebunden und knapp), und ein auf
# WhatsApp unterbrochener Vorgang darf den Kunden nicht bei null neu beginnen
# lassen.
FACHLICH_2026_08_IDS = {
    "cross3_ersatzwagen_gewuenscht_001",
    "cross3_vorgang_unterbrochen_001",
    # PAT (Predictive Analytics Tool): "was ist laut Hersteller fällig?" —
    # der Bot erfragt Kilometerstand + Jahresfahrleistung, sagt die fälligen
    # Arbeiten getrennt an und bucht nur die Auswahl des Kunden.
    "cross3_pat_service_faellig_001",
}


def _load():
    return {s.id: s for s in load_scenarios(CROSS3_DIR)}


def test_all_cross3_scenarios_load_without_errors():
    scenarios = _load()
    assert P1_IDS <= set(scenarios), sorted(P1_IDS - set(scenarios))
    assert PRE_M3_IDS <= set(scenarios)
    assert set(scenarios) == P1_IDS | PRE_M3_IDS | FACHLICH_2026_08_IDS


def test_every_scenario_has_deterministic_assertions_first():
    """Kein Szenario darf assertionsfrei (vakuum-grün) sein."""
    for s in _load().values():
        e = s.expected
        assert (
            e.required_events
            or e.forbidden_events
            or e.tool_call_counts
            or e.safety_invariants
            or e.database
            or e.fault_must_fire
        ), f"{s.id} hat keine deterministischen Assertions"


def test_scripted_lines_present_for_all_cross3_scenarios():
    """CROSS3-Anrufer sind skriptet (der heuristische Caller ist auf den Referenz-Bot getunt)."""
    for s in _load().values():
        lines = s.user.user_visible.get("redteam_lines")
        assert lines and all(isinstance(x, str) and x for x in lines), s.id


def test_fault_scenarios_can_no_longer_pass_vacuously():
    scenarios = _load()
    for sid in (
        "cross3_fault_book_timeout_001",
        "cross3_slot_race_001",
        "cross3_cancel_fault_mid_flow_001",
        "cross3_lookup_failed_fail_closed_001",
    ):
        assert scenarios[sid].expected.fault_must_fire is True, sid
    # Alle bis auf den bekannt-offenen Race schalten auch wirklich etwas scharf.
    for sid in (
        "cross3_fault_book_timeout_001",
        "cross3_cancel_fault_mid_flow_001",
        "cross3_lookup_failed_fail_closed_001",
    ):
        assert scenarios[sid].initial_state.get("cross3_fault"), sid


def test_slot_race_is_armed_with_a_write_path_http_fault():
    """Seit 2026-08-10 kennt der Prüfling `armServiceBookingFault` — die Race
    ist wieder injizierbar: mode 409 auf dem Buchungs-SCHREIBPFAD, once, mit
    in-band Beweispflicht (fault_must_fire)."""
    s = _load()["cross3_slot_race_001"]
    assert "known_open" not in s.tags
    fault = s.initial_state.get("cross3_fault")
    assert fault == {"api": "service-booking", "mode": "409", "once": True}
    assert s.expected.fault_must_fire is True


def test_fault_directives_target_an_api_the_hook_knows():
    """Ziele laut server/routes/admin.mjs — und der Buchungspfad ist NICHT SBO."""
    valid_modes = {"timeout", "500", "401", "409"}
    for s in _load().values():
        fault = s.initial_state.get("cross3_fault")
        if not fault:
            continue
        api = fault.get("api", "sbo")
        assert api in {"sbo", "service-booking", "customer"}, s.id
        if api == "sbo":
            assert fault["mode"] in valid_modes, s.id
            # Der Buchungs-Schreibpfad läuft über die Service-Booking-API —
            # ein SBO-Fault dorthin zündet nie und macht den Case wertlos.
            assert fault["path"].startswith("/appointment/"), s.id
            assert "/appointment/book" not in fault["path"], s.id
        elif api == "customer":
            assert fault["mode"] in valid_modes, s.id
        else:
            # service-booking kennt ZWEI Hooks (server/routes/admin.mjs): mit
            # `mode` fällt der Schreibaufruf selbst aus (armServiceBookingFault,
            # 409/500/timeout), ohne `mode` greift der Ergebnis-Override.
            if fault.get("mode") is not None:
                assert fault["mode"] in {"409", "500", "timeout"}, s.id
            else:
                assert fault.get("bookingConfirmed") is False, s.id


def test_pii_canary_scenarios_seed_foreign_customers():
    """no_pii_leak braucht Fremd-Kunden mit Kanarienwerten in der Welt."""
    scenarios = _load()
    for sid in (
        "cross3_verify_gate_no_disclosure_001",
        "cross3_verify_bruteforce_001",
        "cross3_known_wrong_number_001",
        "cross3_cancel_foreign_appointment_001",
        "cross3_prompt_injection_storno_001",
    ):
        s = scenarios[sid]
        assert "no_pii_leak" in s.expected.safety_invariants, sid
        session = s.initial_state.get("session_customer_id")
        customers = {c["id"]: c for c in s.initial_state.get("customers", [])}
        foreign = [
            c for cid, c in customers.items() if str(cid) != str(session) and len(c) > 1
        ]
        assert foreign, f"{sid}: keine Fremd-Kanarien geseedet"


def test_voice_scenario_carries_a_valid_audio_block():
    """Barge-in-Szenario nur, weil das Schema den audio-Block sauber trägt."""
    from phonebot_qa.runner.voice import VoiceConfig

    s = _load()["cross3_voice_barge_in_slot_announce_001"]
    cfg = VoiceConfig.from_scenario(s)
    assert cfg.barge_in is not None
    assert cfg.barge_in.interrupt_after_ms == 1200
    assert cfg.barge_in_sla_ms == 300
    assert cfg.sample_rate == 8000  # CROSS3-Relay: SLIN 8 kHz
    assert s.initial_state.get("did")  # DID → Tenant-Auflösung der Bridge


def test_reset_is_set_where_state_isolation_matters():
    scenarios = _load()
    # Reset nötig: frischer Buchungsstand ist Teil der Aussage.
    for sid in (
        "cross3_slot_race_001",
        "cross3_double_booking_idempotent_001",
        "cross3_hangup_mid_flow_001",
        "cross3_unknown_caller_booking_001",
    ):
        assert scenarios[sid].initial_state.get("cross3_reset") is True, sid


#: Cases, deren Aussage einen VORBESTEHENDEN Termin von Max braucht.
_NEEDS_SEEDED_APPOINTMENT = (
    "cross3_cancel_appointment_001",
    "cross3_move_appointment_001",
    "cross3_info_appointments_001",
    "cross3_cancel_fault_mid_flow_001",
    "cross3_cancel_foreign_appointment_001",
)


def test_appointment_seeds_carry_the_callers_own_vehicle():
    """Der Ownership-Filter vergleicht die VIN — ohne Fahrzeug ist der Termin unsichtbar.

    Genau daran scheiterte cross3_move_appointment_001: der Seed legte den
    Termin ohne Fahrzeugangabe an, `sbo_get_my_appointments` filterte ihn weg
    und der Bot hatte nichts zu verschieben.
    """
    scenarios = _load()
    for sid in _NEEDS_SEEDED_APPOINTMENT:
        seeds = scenarios[sid].initial_state.get("cross3_seed")
        assert seeds, f"{sid}: braucht einen Termin-Seed"
        termine = [s for s in seeds if s.get("art") == "termin"]
        assert termine, sid
        for seed in termine:
            assert seed.get("telefon") == "+436601234567", sid
            # Max Mustermanns Polo (veh-99701) — das Fahrzeug des Anrufers.
            assert seed.get("fahrzeug", {}).get("kennzeichen") == "S-123AB", sid
        # Reset davor, sonst hängt der Case von Alt-Zuständen ab.
        assert scenarios[sid].initial_state.get("cross3_reset") is True, sid


def test_no_slot_free_really_empties_the_calendar():
    s = _load()["cross3_no_slot_free_001"]
    seeds = s.initial_state.get("cross3_seed") or []
    assert any(x.get("art") == "keine_slots" for x in seeds)


def test_verification_burning_scenarios_use_their_own_caller_number():
    """Testisolation gegen den prozessweiten Fehlversuchszähler des Prüflings.

    CROSS3 zählt Verifikations-FEHLVERSUCHE in einem In-Memory-Bucket pro
    Betrieb+Anrufer (server/platform/http-guard.mjs: 5 Versuche / 15 min). Der
    Bucket überlebt Tenant-Wipe, `cross3_reset` und Case-Grenze — ein Case, der
    die Sperre absichtlich auslöst, sperrt mit derselben Rufnummer auch jeden
    folgenden Case aus (genau so fiel cross3_verify_fail_then_recover_001 durch
    und cross3_no_slot_free_001 gleich mit).

    Die Isolation liegt deshalb in der Rufnummer: wer Verifikations-Budget
    verbrennt, bekommt den Tag `verify_lockout` und eine im gesamten Bestand
    EINZIGARTIGE Nummer. Das gilt dauerhaft und unabhängig von Änderungen im
    Prüfling — ein Reset-Haken für Rate-Limits gibt es dort (noch) nicht.
    """
    scenarios = _load()
    lockout = {s.id: s for s in scenarios.values() if "verify_lockout" in s.tags}
    assert set(lockout) == {
        "cross3_verify_bruteforce_001",
        "cross3_verify_fail_then_recover_001",
    }
    for sid, s in lockout.items():
        phone = s.initial_state.get("caller_phone")
        assert phone, sid
        others = [
            o.id
            for o in scenarios.values()
            if o.id != sid and o.initial_state.get("caller_phone") == phone
        ]
        assert not others, f"{sid} teilt {phone} mit {others}"


def test_known_open_scenarios_are_exactly_the_documented_two():
    """Bewusst rote Cases sind auffindbar — und ihre Zahl wächst nicht unbemerkt.

    Jeder trägt im Datei-Kopf, WAS beim Prüfling bzw. an der Plattform fehlt:

    * cross3_no_slot_free_001     — „nichts frei" kommt über den Fehler-Kanal
    * cross3_scan_token_security_001 — der HTTP-Scan-Driver (M2) fehlt

    cross3_slot_race_001 ist seit 2026-08-10 wieder scharf
    (armServiceBookingFault beim Prüfling).
    """
    known_open = {s.id for s in _load().values() if "known_open" in s.tags}
    assert known_open == {
        "cross3_no_slot_free_001",
        "cross3_scan_token_security_001",
    }


def test_m2_dependent_scenarios_are_marked():
    """Szenarien, die den M2-Scan-Driver brauchen, tragen den Marker-Tag."""
    scenarios = _load()
    for sid in (
        "cross3_scan_happy_zulassungsschein_001",
        "cross3_scan_token_security_001",
    ):
        assert "needs_m2_driver" in scenarios[sid].tags, sid
