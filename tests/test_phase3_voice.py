"""Phase 3: audio core, transports, voice runner, barge-in and voice metrics."""

from __future__ import annotations

import copy

import pytest

from phonebot_qa.adapters.bot.reference import ReferenceAppointmentBot
from phonebot_qa.adapters.transport import LoopbackTransport, SIPTransport, WebRTCTransport
from phonebot_qa.audio import (
    BARGE_IN_SLA_MS,
    AudioBuffer,
    BargeInConfig,
    BargeInController,
    ChaosConfig,
    DeterministicSTT,
    DeterministicTTS,
    PROFILES,
    apply_chaos,
    get_profile,
    word_error_rate,
)
from phonebot_qa.evaluation.voice import latency_breakdown
from phonebot_qa.observability import EventLog
from phonebot_qa.orchestrator import run_suite
from phonebot_qa.scenario.loader import load_scenario, load_scenarios
from tests.conftest import SCENARIOS_DIR

VOICE_DIR = SCENARIOS_DIR / "voice"
UTTERANCE = "Ich möchte meinen Termin verschieben, und zwar auf 2026-08-14T10:00:00+02:00."


# --- Audio buffer ----------------------------------------------------------- #


def test_buffer_duration_and_wav():
    buf = AudioBuffer.tone(500, sample_rate=16000)
    assert 495 <= buf.duration_ms <= 505
    assert buf.rms > 0
    wav = buf.to_wav_bytes()
    assert wav[:4] == b"RIFF" and b"WAVE" in wav[:16]


def test_buffer_bytes_round_trip():
    buf = AudioBuffer.tone(100)
    again = AudioBuffer.from_bytes(buf.to_bytes(), sample_rate=buf.sample_rate)
    assert list(again.samples) == list(buf.samples)


# --- TTS -------------------------------------------------------------------- #


def test_tts_is_deterministic_and_scales_with_text():
    tts = DeterministicTTS()
    a, b = tts.synthesize(UTTERANCE), tts.synthesize(UTTERANCE)
    assert list(a.samples) == list(b.samples)
    assert a.metadata["transcript"] == UTTERANCE
    short = tts.synthesize("Ja.")
    assert short.duration_ms < a.duration_ms


def test_tts_speed_shortens_audio():
    tts = DeterministicTTS()
    assert tts.synthesize(UTTERANCE, speed=1.5).duration_ms < tts.synthesize(UTTERANCE).duration_ms


# --- Chaos ------------------------------------------------------------------ #


def test_all_named_profiles_apply():
    tts = DeterministicTTS()
    clean = tts.synthesize(UTTERANCE)
    for name in PROFILES:
        out = apply_chaos(clean, get_profile(name), seed=1)
        assert out.metadata["degradation"]["name"] == name
        assert len(out) > 0


def test_chaos_is_deterministic_per_seed():
    clean = DeterministicTTS().synthesize(UTTERANCE)
    a = apply_chaos(clean, get_profile("street"), seed=7)
    b = apply_chaos(clean, get_profile("street"), seed=7)
    c = apply_chaos(clean, get_profile("street"), seed=8)
    assert list(a.samples) == list(b.samples)
    assert list(a.samples) != list(c.samples)


def test_noise_lowers_snr_and_packet_loss_zeroes_frames():
    clean = DeterministicTTS().synthesize(UTTERANCE)
    noisy = apply_chaos(clean, ChaosConfig(name="n", snr_db=5.0, noise="white"), seed=1)
    assert noisy.metadata["degradation"]["applied_snr_db"] == 5.0
    lossy = apply_chaos(clean, ChaosConfig(name="l", packet_loss=0.3), seed=1)
    assert lossy.metadata["degradation"]["lost_ms"] > 0


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        get_profile("does-not-exist")


# --- STT -------------------------------------------------------------------- #


def test_wer_metric():
    assert word_error_rate("a b c", "a b c") == 0.0
    assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)
    assert word_error_rate("a b c", "") == 1.0


def test_stt_near_perfect_on_clean_audio_and_degrades_with_noise():
    tts, stt = DeterministicTTS(), DeterministicSTT(seed=3)
    clean = apply_chaos(tts.synthesize(UTTERANCE), get_profile("clean"), seed=3)
    # Real recognizers keep a small error floor even on pristine audio, so this
    # asserts "at most one word wrong", not "identical".
    one_word = 1.0 / len(UTTERANCE.split())
    assert stt.transcribe(clean).wer <= one_word

    prev = -1.0
    for profile in ("office", "street", "restaurant"):
        deg = apply_chaos(tts.synthesize(UTTERANCE), get_profile(profile), seed=3)
        p = stt.error_probability(deg)
        assert p > prev, f"{profile} should be harder than the previous profile"
        prev = p


def test_stt_is_deterministic():
    tts = DeterministicTTS()
    deg = apply_chaos(tts.synthesize(UTTERANCE), get_profile("street"), seed=2)
    a = DeterministicSTT(seed=2).transcribe(deg)
    b = DeterministicSTT(seed=2).transcribe(deg)
    assert a.text == b.text and a.wer == b.wer


# --- Transports ------------------------------------------------------------- #


async def test_transports_carry_audio_and_report_stats():
    audio = DeterministicTTS().synthesize(UTTERANCE)
    for transport in (
        LoopbackTransport(latency_ms=20, seed=1),
        SIPTransport(seed=1),
        WebRTCTransport(seed=1),
    ):
        await transport.connect(session_id="s")
        out = await transport.send(audio)
        stats = transport.stats()
        assert len(out) > 0
        assert stats.one_way_latency_ms >= 0
        await transport.disconnect()


async def test_sip_and_webrtc_emit_signalling_events():
    log = EventLog()
    sip = SIPTransport(seed=1, events=log)
    await sip.connect(session_id="s")
    await sip.disconnect()
    assert log.has("sip_invite") and log.has("sip_answered") and log.has("sip_bye")

    log2 = EventLog()
    rtc = WebRTCTransport(seed=1, events=log2)
    await rtc.connect(session_id="s")
    await rtc.disconnect()
    assert log2.has("webrtc_offer") and log2.has("webrtc_closed")


# --- Barge-in (§14) --------------------------------------------------------- #


def test_barge_in_detected_within_sla():
    log = EventLog()
    result = BargeInController(detection_ms=120, stop_ms=60).interrupt(
        bot_audio_ms=3000, config=BargeInConfig(interrupt_after_ms=850), events=log
    )
    assert result.detected and result.within_sla
    assert result.stop_latency_ms == 180
    assert log.has("barge_in_detected") and log.has("bot_audio_stop")


def test_barge_in_sla_breach_flagged():
    result = BargeInController(detection_ms=BARGE_IN_SLA_MS + 100).interrupt(
        bot_audio_ms=3000, config=BargeInConfig(interrupt_after_ms=500)
    )
    assert result.detected and result.within_sla is False


def test_bot_without_barge_in_loses_caller_audio():
    log = EventLog()
    result = BargeInController(supports_barge_in=False).interrupt(
        bot_audio_ms=3000, config=BargeInConfig(interrupt_after_ms=850), events=log
    )
    assert result.detected is False
    assert result.user_audio_lost_ms == 2150
    assert log.has("barge_in_missed")


def test_no_interrupt_when_bot_finishes_first():
    result = BargeInController().interrupt(
        bot_audio_ms=400, config=BargeInConfig(interrupt_after_ms=850)
    )
    assert result.attempted is False


# --- Voice end-to-end ------------------------------------------------------- #


async def test_voice_scenarios_pass_on_healthy_bot():
    scenarios = load_scenarios(VOICE_DIR)
    summary = await run_suite(scenarios, modes=["voice"], seeds=[0])
    assert summary.passed == summary.total, [f.critical_failure for f in summary.failures()]


async def test_same_scenario_runs_in_both_modes():
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_appointment_001.yaml")
    summary = await run_suite([scenario], modes=["text", "voice"], seeds=[1])
    by_mode = {r.mode: r for r in summary.results}
    assert by_mode["text"].result == "PASS"
    assert by_mode["voice"].result == "PASS"
    # Voice carries metrics text does not.
    assert by_mode["text"].voice is None
    assert by_mode["voice"].voice is not None
    # A real call takes far longer than a text run (speech is real time).
    assert by_mode["voice"].latency.duration_seconds > by_mode["text"].latency.duration_seconds


async def test_voice_emits_full_lifecycle_and_latency_breakdown():
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_appointment_001.yaml")
    result = (await run_suite([scenario], modes=["voice"], seeds=[1])).results[0]
    types = {e.type for e in result.events}
    for expected in (
        "user_audio_started",
        "user_audio_finished",
        "stt_started",
        "stt_finished",
        "tts_started",
        "tts_finished",
        "bot_audio_started",
        "bot_audio_finished",
    ):
        assert expected in types, f"missing voice event {expected}"

    breakdown = latency_breakdown(result.events)
    assert breakdown["stt"] > 0 and breakdown["tts"] > 0 and breakdown["tool"] > 0


async def test_voice_run_is_deterministic():
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_appointment_001.yaml")
    a = (await run_suite([scenario], modes=["voice"], seeds=[5])).results[0]
    b = (await run_suite([scenario], modes=["voice"], seeds=[5])).results[0]
    assert [t.bot for t in a.conversation.turns] == [t.bot for t in b.conversation.turns]
    assert a.voice.stt_wer == b.voice.stt_wer


async def test_heavy_noise_degrades_success_rate():
    """Clean audio must be reliable; heavy noise must actually break calls."""
    base = load_scenario(SCENARIOS_DIR / "booking" / "move_appointment_001.yaml")
    seeds = list(range(12))

    clean = copy.deepcopy(base)
    clean.audio = {"profile": "clean"}
    clean_summary = await run_suite([clean], modes=["voice"], seeds=seeds)
    assert clean_summary.passed == clean_summary.total

    noisy = copy.deepcopy(base)
    noisy.audio = {"profile": "restaurant"}
    noisy_summary = await run_suite([noisy], modes=["voice"], seeds=seeds)
    assert noisy_summary.passed < clean_summary.passed, "noise must have an effect"

    clean_wer = [r.voice.stt_wer for r in clean_summary.results]
    noisy_wer = [r.voice.stt_wer for r in noisy_summary.results]
    mean_clean = sum(clean_wer) / len(clean_wer)
    mean_noisy = sum(noisy_wer) / len(noisy_wer)
    # Clean audio keeps only the recognizer's small error floor; heavy noise is
    # an order of magnitude worse.
    assert mean_clean < 0.05
    assert mean_noisy > 0.2
    assert mean_noisy > mean_clean * 4


async def test_barge_in_failure_is_critical():
    scenario = load_scenario(VOICE_DIR / "voice_barge_in_001.yaml")
    broken = copy.deepcopy(scenario)
    broken.audio = {**scenario.audio, "supports_barge_in": False}
    result = (await run_suite([broken], modes=["voice"], seeds=[0])).results[0]
    assert result.result == "FAIL"
    assert "barge_in" in (result.critical_failure or "")


async def test_wer_budget_assertion():
    scenario = load_scenario(SCENARIOS_DIR / "booking" / "move_appointment_001.yaml")
    strict = copy.deepcopy(scenario)
    strict.audio = {"profile": "restaurant", "max_wer": 0.01, "wer_critical": True}
    result = (await run_suite([strict], modes=["voice"], seeds=[0])).results[0]
    assert result.result == "FAIL"
    assert "wer_budget" in (result.critical_failure or "")
