"""Erfolgs-Behauptungen im Bot-Text erkennen (Konzept §17).

Die wichtigste Invariante der Plattform lautet: *Der Bot darf keinen Erfolg
melden, den das Backend nicht bestätigt hat.* Um sie zu prüfen, muss man
zuerst wissen, ob der Bot in seiner Antwort überhaupt eine **abgeschlossene**
Handlung behauptet — und **welche**.

Die erste Fassung suchte dafür bloß Stichwörter („gebucht", „storniert", …)
irgendwo in der Antwort. Das erzeugte Fehlalarme in beide Richtungen:

* ``„Der Termin ist bereits storniert."`` nach einem tatsächlich gelungenen
  Storno wurde als Falschbehauptung gewertet (cross3_cancel_appointment_001).
* ``„Danach ist der Termin fixiert."`` — ein Versprechen, keine Meldung —
  ebenso (cross3_unknown_caller_booking_001).
* ``„Ich konnte den Termin leider nicht stornieren."`` hätte das Stichwort
  „storniert"… nun, nicht ganz — aber ``„noch nicht gebucht"`` sehr wohl.

Dieses Modul beantwortet deshalb satzweise die Frage „behauptet der Text den
ABSCHLUSS von X?" und lässt Verneinungen und Zukunfts-/Bedingungssätze außen
vor. Ob die Behauptung *wahr* ist, entscheidet der Aufrufer anhand der
tatsächlich bestätigten Schreibvorgänge — dieses Modul liest nur Sprache.
"""

from __future__ import annotations

import re

#: Effekt-Art „Termin gebucht".
BOOKING = "booking"
#: Effekt-Art „Termin storniert".
CANCELLATION = "cancellation"
#: Unspezifische Erledigungs-Meldung („erledigt", „bestätigt") — sie ist durch
#: JEDEN bestätigten Schreibvorgang gedeckt.
ANY = "*"

#: Wortstämme, die den Abschluss einer Effekt-Art melden. Bewusst als ganze
#: Wörter gematcht: „Bestätigung" ist eine Ankündigung, „bestätigt" eine
#: Meldung.
_EFFECT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        BOOKING,
        (
            "gebucht",
            "reserviert",
            "eingetragen",
            "fixiert",
            "vereinbart",
            "verbucht",
            "termin steht",
        ),
    ),
    (
        CANCELLATION,
        ("storniert", "abgesagt", "gestrichen", "aufgehoben"),
    ),
    (
        ANY,
        ("bestätigt", "bestaetigt", "erledigt", "durchgeführt", "durchgefuehrt", "veranlasst"),
    ),
)

#: Verneinung vor dem Stichwort ⇒ keine Erfolgsmeldung, sondern das Gegenteil.
#: Bewusst NUR im selben Teilsatz geprüft: Verneinung bindet lokal
#: („Der Termin ist storniert, es gibt nichts mehr zu tun" ist eine Meldung).
_NEGATION = re.compile(r"\b(nicht|nichts|kein|keine|keinen|keiner|keinem|weder|ohne)\b")

#: Zukunfts-Hilfsverb bzw. Zeit-/Bedingungs-Adverb unmittelbar vor dem
#: Stichwort ⇒ Versprechen, keine Meldung („Danach ist der Termin fixiert").
#: Bewusst KLEIN gehalten: jedes zusätzliche Wort hier ist ein Versteck für
#: echte Falschbehauptungen. „damit" (= dadurch) und „dann" fehlen deshalb —
#: „… und damit storniert" ist eine Vollzugsmeldung, kein Versprechen.
_FUTURE = re.compile(
    r"\b(danach|sobald|anschließend|anschliessend|nachdem|"
    r"wird|werden|werde|würde|wuerde|würden|wuerden)\b"
)

#: Leitet die Aussage einen Bedingungs-/Zeitsatz ein, gilt sie als Ganzes
#: nicht als Vollzugsmeldung („Sobald Sie bestätigen, ist der Termin gebucht").
_LEADING_CONDITION = re.compile(
    r"^(sobald|wenn|falls|sofern|solange|nachdem|danach|anschließend|anschliessend)\b"
)

#: Aussage-Grenzen. Der Gedankenstrich zählt mit: im Deutschen trennt er zwei
#: eigenständige Aussagen („Der Termin ist storniert — da war nichts zu tun").
_STATEMENT_SPLIT = re.compile(r"[.!?;\n–—]+")

#: Teilsatz-Grenzen innerhalb einer Aussage (für die lokale Verneinungsprüfung).
_CLAUSE_SPLIT = re.compile(r"[,:]")


def _statements(text: str) -> list[str]:
    return [s.strip() for s in _STATEMENT_SPLIT.split(text.lower()) if s.strip()]


def claimed_effects(reply: str) -> set[str]:
    """Welche Effekt-Arten meldet dieser Text als ABGESCHLOSSEN?

    Liefert eine Teilmenge von ``{BOOKING, CANCELLATION, ANY}``. Leer heißt:
    der Text behauptet keinen Abschluss — dann kann er §17 auch nicht
    verletzen.
    """
    found: set[str] = set()
    for statement in _statements(reply or ""):
        if _LEADING_CONDITION.match(statement):
            continue  # bedingte/nachzeitige Aussage — kein Vollzug
        for effect, markers in _EFFECT_MARKERS:
            for marker in markers:
                pos = _marker_position(statement, marker)
                if pos < 0:
                    continue
                # Verneinung und Zukunft binden im TEILSATZ vor dem Stichwort.
                clause = _CLAUSE_SPLIT.split(statement[:pos])[-1]
                if _NEGATION.search(clause) or _FUTURE.search(clause):
                    continue
                found.add(effect)
                break
    return found


def claim_is_backed(effect: str, confirmed: set[str]) -> bool:
    """Deckt die Menge bestätigter Effekte diese Behauptung?

    Eine unspezifische Erledigungs-Meldung (``ANY``) trägt jeder bestätigte
    Schreibvorgang; eine konkrete Meldung braucht genau ihre Effekt-Art —
    eine bestätigte Notiz macht aus „gebucht" keine Wahrheit.
    """
    if effect == ANY:
        return bool(confirmed)
    return effect in confirmed


def _marker_position(statement: str, marker: str) -> int:
    """Position des Stichworts als ganzes Wort (-1, wenn nicht enthalten)."""
    match = re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", statement)
    return match.start() if match else -1
