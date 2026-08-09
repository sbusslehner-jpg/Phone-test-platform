"""Load & validate scenario and persona YAML files (concept sections 5 & 9).

The YAML file is the *source of truth* (section 5). Loading is strict: unknown
fields, malformed ids and type errors fail loudly so a broken scenario can never
silently pass a test run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ..models import Persona, Scenario


class ScenarioValidationError(ValueError):
    """Raised when a scenario/persona file fails schema validation."""


def _read_yaml(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ScenarioValidationError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        raise ScenarioValidationError(f"{path}: file is empty")
    if not isinstance(raw, dict):
        raise ScenarioValidationError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return raw


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a single scenario YAML file."""
    path = Path(path)
    data = _read_yaml(path)
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioValidationError(f"{path}: {exc}") from exc


def load_scenarios(root: str | Path) -> list[Scenario]:
    """Recursively load every ``*.yaml``/``*.yml`` scenario under ``root``.

    Returns scenarios sorted by id for deterministic ordering. Duplicate ids
    are an error — ids must be globally unique (they key the regression store
    and the report).
    """
    root = Path(root)
    if root.is_file():
        return [load_scenario(root)]
    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}
    for file in sorted(root.rglob("*.y*ml")):
        # Persona files live alongside scenarios in some layouts; skip anything
        # that doesn't look like a scenario (no ``user`` section).
        data = _read_yaml(file)
        if "user" not in data:
            continue
        scenario = load_scenario(file)
        if scenario.id in seen:
            raise ScenarioValidationError(
                f"duplicate scenario id {scenario.id!r} in {file} "
                f"(already defined in {seen[scenario.id]})"
            )
        seen[scenario.id] = file
        scenarios.append(scenario)
    scenarios.sort(key=lambda s: s.id)
    return scenarios


def load_persona(path: str | Path) -> Persona:
    """Load and validate a single persona YAML file."""
    path = Path(path)
    data = _read_yaml(path)
    try:
        return Persona.model_validate(data)
    except ValidationError as exc:
        raise ScenarioValidationError(f"{path}: {exc}") from exc


def load_personas(root: str | Path) -> dict[str, Persona]:
    """Load every persona under ``root`` into an ``id -> Persona`` map."""
    root = Path(root)
    if root.is_file():
        p = load_persona(root)
        return {p.id: p}
    personas: dict[str, Persona] = {}
    for file in sorted(root.rglob("*.y*ml")):
        data = _read_yaml(file)
        # Persona files have persona-ish keys but no ``user`` section.
        if "user" in data or "goal" in data:
            continue
        persona = load_persona(file)
        if persona.id in personas:
            raise ScenarioValidationError(
                f"duplicate persona id {persona.id!r} in {file}"
            )
        personas[persona.id] = persona
    return personas
