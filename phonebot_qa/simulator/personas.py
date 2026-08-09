"""Built-in personas + persona resolution (concept section 9).

Personas are orthogonal to scenarios: any persona combines with any scenario.
A handful of built-ins are provided so the platform works out of the box; more
can be authored as YAML under ``personas/`` and loaded via
:func:`phonebot_qa.scenario.load_personas`.
"""

from __future__ import annotations

from ..models import Persona

# The default caller when a scenario names no persona.
DEFAULT_PERSONA_ID = "normal"

BUILTIN_PERSONAS: dict[str, Persona] = {
    p.id: p
    for p in [
        Persona(
            id="normal",
            description="Kooperativer, klarer Anrufer.",
            patience="medium",
            verbosity="medium",
        ),
        Persona(
            id="impatient",
            description="Ungeduldiger Anrufer, will schnell ans Ziel.",
            patience="low",
            verbosity="short",
            interruption_probability=0.4,
            correction_probability=0.2,
        ),
        Persona(
            id="confused",
            description="Verwirrter Anrufer, unsichere Angaben.",
            patience="medium",
            verbosity="long",
            correction_probability=0.4,
            off_topic_probability=0.2,
        ),
        Persona(
            id="talkative",
            description="Redseliger Anrufer, schweift ab.",
            patience="high",
            verbosity="long",
            off_topic_probability=0.4,
        ),
        Persona(
            id="terse",
            description="Sehr kurze, knappe Antworten.",
            patience="low",
            verbosity="short",
        ),
        Persona(
            id="elderly",
            description="Älterer, nicht technikaffiner Anrufer.",
            patience="high",
            verbosity="long",
            tech_savvy="low",
            correction_probability=0.3,
        ),
        Persona(
            id="non_native",
            description="Anrufer mit schlechteren Deutschkenntnissen.",
            patience="medium",
            verbosity="short",
            language_proficiency="low",
        ),
    ]
}


def resolve_persona(
    persona_id: str | None, extra: dict[str, Persona] | None = None
) -> Persona:
    """Resolve a persona id to a :class:`Persona`, preferring ``extra``.

    Falls back to the default persona when ``persona_id`` is ``None``. Raises
    ``KeyError`` for an unknown non-null id so typos fail loudly.
    """
    extra = extra or {}
    if persona_id is None:
        return extra.get(DEFAULT_PERSONA_ID) or BUILTIN_PERSONAS[DEFAULT_PERSONA_ID]
    if persona_id in extra:
        return extra[persona_id]
    if persona_id in BUILTIN_PERSONAS:
        return BUILTIN_PERSONAS[persona_id]
    raise KeyError(f"unknown persona id {persona_id!r}")
