"""Degradations-Erkennung (M3 / P1 der Fall-Matrix, docs/fall-matrix.json).

Ein Bot, dessen LLM-Backend wegbricht (Azure-429, Abbruch, Timeout), antwortet
oft mit einer generischen Fallback-Floskel statt mit Geschäftslogik. Solche
Turns sind DEGRADIERT: kein Tool-Call, keine Buchung — und ein Szenario, das
nur forbidden-/Safety-Assertions trägt, würde trivial „bestehen", obwohl der
Bot faktisch tot war. Genau diese Blindstelle hat der QA-Lauf gegen CROSS3
aufgedeckt (P1).

Mechanik:

* Adapter (z. B. :class:`~phonebot_qa.adapters.bot.cross3.Cross3Adapter`)
  markieren erkannte Fallback-Antworten mit ``BotResponse.metadata["degraded"]``.
* Der :class:`~phonebot_qa.runner.conversation.ConversationRunner` übernimmt die
  Markierung auf den :class:`~phonebot_qa.models.Turn` (``Turn.degraded``),
  wendet zusätzlich eigene ``RunnerConfig.fallback_patterns`` bot-agnostisch an
  und emittiert das Event ``bot_degraded``.
* Die Evaluation prüft ``technical:not_degraded`` (kritisch): mehr degradierte
  Turns als ``expected.max_degraded_turns`` (Default 0) ⇒ FAIL.

Dieses Modul ist bewusst abhängigkeitsfrei (nur ``re``), damit Adapter, Runner
und Evaluation es ohne Importzyklen teilen können.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

#: CROSS3s Fallback-Text, wenn der Azure-OpenAI-Call abbricht (429 u. Ä.):
#: „Entschuldigung, ich habe gerade ein technisches Problem. Bitte versuchen
#: Sie es in ein paar Minuten noch einmal — oder rufen Sie uns direkt an."
#: Der erste Satz ist das stabile Erkennungsmerkmal; als Regex formuliert,
#: damit Whitespace-/Interpunktions-Varianten ebenfalls matchen.
CROSS3_FALLBACK_PATTERN = (
    r"entschuldigung\W+ich\s+habe\s+gerade\s+ein\s+technisches\s+problem"
)

#: Default-Muster (regex-fähig, case-insensitiv). Adapter und Runner können
#: eigene Muster ergänzen oder komplett ersetzen.
DEFAULT_FALLBACK_PATTERNS: tuple[str, ...] = (CROSS3_FALLBACK_PATTERN,)


def compile_patterns(
    patterns: Iterable[str] | None,
) -> tuple[re.Pattern[str], ...]:
    """Regex-Muster einmalig kompilieren (case-insensitive Suche)."""
    return tuple(re.compile(p, re.IGNORECASE) for p in (patterns or ()))


def is_degraded(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    """True, wenn der Bot-Text wie eine Fallback-/Degradationsantwort aussieht."""
    if not text:
        return False
    return any(p.search(text) for p in patterns)
