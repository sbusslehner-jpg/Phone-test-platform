"""CLI exit codes — the CI release-gate contract."""

from __future__ import annotations

import json

from phonebot_qa.cli import main
from tests.conftest import SCENARIOS_DIR

_SD = str(SCENARIOS_DIR)


def test_run_good_bot_exits_zero():
    assert main(["run", "--suite", "booking", "--scenarios-dir", _SD]) == 0


def test_run_buggy_bot_exits_nonzero():
    code = main(
        ["run", "--suite", "booking", "--scenarios-dir", _SD, "--bot-version", "buggy-corrections"]
    )
    assert code == 1


def test_redteam_passes():
    assert main(["redteam", "--scenarios-dir", _SD]) == 0


def test_gate_blocks_buggy_candidate():
    code = main(
        [
            "gate",
            "--suite",
            "booking",
            "--scenarios-dir",
            _SD,
            "--baseline-version",
            "reference-1.0",
            "--candidate-version",
            "buggy-corrections",
        ]
    )
    assert code == 1


def test_run_writes_json_report(tmp_path):
    out = tmp_path / "report.json"
    code = main(["run", "--suite", "booking", "--scenarios-dir", _SD, "--json", str(out)])
    assert code == 0
    report = json.loads(out.read_text("utf-8"))
    assert report["total"] >= 1
    assert "cases" in report


def test_list_command():
    assert main(["list", "--suite", "all", "--scenarios-dir", _SD]) == 0


def test_skip_tag_laesst_voice_szenarien_weg():
    """``run`` faehrt Text — ein Telefon-Szenario kann dort nicht bestehen."""
    from argparse import Namespace

    from phonebot_qa.cli import _skip_tags

    class S:
        def __init__(self, sid, tags):
            self.id, self.tags = sid, tags

    alle = [S("a", ["cross3", "voice"]), S("b", ["cross3"]), S("c", None)]
    behalten = _skip_tags(alle, Namespace(skip_tag=["voice"]))
    assert [s.id for s in behalten] == ["b", "c"]
    # Ohne Angabe bleibt alles, wie es war.
    assert _skip_tags(alle, Namespace(skip_tag=None)) is alle


def test_run_writes_transcripts_of_failing_cases_only(tmp_path):
    # Nachtlauf 2026-10-07: cross3_book_pickerl_001 rot, lokal nicht
    # nachstellbar, das Gespräch nirgends gespeichert. --transcripts legt es ab.
    ziel = tmp_path / "transkripte"
    code = main(["run", "--suite", "booking", "--scenarios-dir", _SD, "--bot-version", "buggy-corrections", "--transcripts", str(ziel)])
    assert code != 0
    dateien = sorted(ziel.glob("*.json"))
    assert dateien, "kein Transkript für einen roten Fall"
    daten = json.loads(dateien[0].read_text("utf-8"))
    assert daten["result"] in ("FAIL", "ERROR")
    assert daten["turns"] and {"user", "bot"} <= set(daten["turns"][0])
    assert isinstance(daten["tool_calls"], list) and isinstance(daten["events"], list)


def test_run_without_failures_writes_no_transcripts(tmp_path):
    ziel = tmp_path / "transkripte"
    assert main(["run", "--suite", "booking", "--scenarios-dir", _SD, "--transcripts", str(ziel)]) == 0
    assert not list(ziel.glob("*.json"))
