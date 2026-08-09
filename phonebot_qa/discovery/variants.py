"""Scenario variant generation (concept section 23).

Discovery explores the space *around* a hand-written scenario by mutating it
along axes that are known to break real phonebots. Every mutation rewrites the
scenario's expectations so the variant stays a *correct* test — a generated case
must never fail for a reason the bot is not responsible for.

Three sound mutation families are implemented:

``persona``
    Swap the caller. Persona and scenario are orthogonal by design (section 9),
    so expectations are unchanged — this hunts for bots that only work for the
    "nice" caller.

``latency``
    Inject backend slowness. The task must still complete, so expectations are
    unchanged — this hunts for premature timeouts and impatient behaviour.

``write_fault``
    Fault the scenario's *write* tool. Expectations are rewritten to the
    unchanged-state contract: the record must keep its original value, the write
    event becomes forbidden, and the bot must not claim success (section 17).
    This automatically gives every happy-path scenario a matching
    backend-failure test.
"""

from __future__ import annotations

import copy
from typing import Any

from ..models import Scenario

#: write event -> tool that produces it
WRITE_EVENT_TOOLS = {
    "appointment_updated": "appointment.update",
    "appointment_created": "appointment.create",
    "appointment_cancelled": "appointment.cancel",
}

DEFAULT_VARIANT_PERSONAS = ("impatient", "confused", "elderly", "terse", "non_native")


def _clone(scenario: Scenario) -> dict[str, Any]:
    return copy.deepcopy(scenario.model_dump(mode="json"))


def _initial_record(scenario: Scenario, dotted_key: str) -> dict[str, Any] | None:
    """Look up ``collection.record_id`` in the scenario's initial_state."""
    collection, _, record_id = dotted_key.partition(".")
    records = scenario.initial_state.get(collection)
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and str(record.get("id")) == record_id:
                return record
    elif isinstance(records, dict) and str(records.get("id")) == record_id:
        return records
    return None


def persona_variants(
    scenario: Scenario, personas: tuple[str, ...] = DEFAULT_VARIANT_PERSONAS
) -> list[Scenario]:
    """One variant per persona (expectations unchanged)."""
    out = []
    for persona in personas:
        if persona == scenario.user.persona:
            continue
        data = _clone(scenario)
        data["id"] = f"{scenario.id}__persona_{persona}"
        data["tags"] = [*scenario.tags, "discovery", "variant:persona"]
        data["user"]["persona"] = persona
        out.append(Scenario.model_validate(data))
    return out


def latency_variants(scenario: Scenario, latency_ms: int = 2500) -> list[Scenario]:
    """A variant where every backend call is slow (expectations unchanged)."""
    data = _clone(scenario)
    data["id"] = f"{scenario.id}__slow_backend"
    data["tags"] = [*scenario.tags, "discovery", "variant:latency"]
    faults = dict(data.get("faults") or {})
    for service in ("appointment", "availability", "customer"):
        cfg = dict(faults.get(service) or {})
        cfg["latency_ms"] = latency_ms
        faults[service] = cfg
    data["faults"] = faults
    # Slow backend must not blow the duration budget for the *test*; give it room.
    limits = dict(data.get("limits") or {})
    limits["max_duration_seconds"] = max(
        int(limits.get("max_duration_seconds", 180)), 600
    )
    data["limits"] = limits
    return [Scenario.model_validate(data)]


def write_fault_variants(scenario: Scenario) -> list[Scenario]:
    """Fault the scenario's write tool and rewrite it to the no-change contract.

    Produces nothing for scenarios that perform no write (there is nothing to
    fault) — e.g. red-team probes.
    """
    write_events = [
        e for e in scenario.expected.required_events if e in WRITE_EVENT_TOOLS
    ]
    if not write_events:
        return []
    write_event = write_events[0]
    write_tool = WRITE_EVENT_TOOLS[write_event]

    data = _clone(scenario)
    data["id"] = f"{scenario.id}__write_fault"
    data["tags"] = [*scenario.tags, "discovery", "variant:write_fault"]
    faults = dict(data.get("faults") or {})
    faults[write_tool] = {"fail": True, "kind": "timeout", "latency_ms": 1500}
    data["faults"] = faults

    expected = dict(data.get("expected") or {})
    # The record must keep its ORIGINAL value: nothing was confirmed.
    original_db: dict[str, dict[str, Any]] = {}
    for key, fields in (scenario.expected.database or {}).items():
        record = _initial_record(scenario, key)
        if record is None:
            # A record that did not exist before (a create) must not appear.
            continue
        original_db[key] = {
            field: record.get(field) for field in fields if field in record
        }
    expected["database"] = original_db
    # Keep the pre-write steps, drop the write itself, require the fault.
    expected["required_events"] = [
        e for e in scenario.expected.required_events if e not in WRITE_EVENT_TOOLS
    ] + ["tool_fault_injected"]
    expected["forbidden_events"] = sorted(
        {*scenario.expected.forbidden_events, write_event}
    )
    # No successful write may be recorded.
    counts = dict(scenario.expected.tool_call_counts or {})
    counts[write_tool] = 0
    expected["tool_call_counts"] = counts
    data["expected"] = expected

    limits = dict(data.get("limits") or {})
    limits["max_duration_seconds"] = max(
        int(limits.get("max_duration_seconds", 180)), 600
    )
    data["limits"] = limits
    return [Scenario.model_validate(data)]


def generate_variants(
    scenario: Scenario,
    *,
    personas: tuple[str, ...] = DEFAULT_VARIANT_PERSONAS,
    include: tuple[str, ...] = ("persona", "latency", "write_fault"),
) -> list[Scenario]:
    """All requested variants of one scenario."""
    variants: list[Scenario] = []
    if "persona" in include:
        variants += persona_variants(scenario, personas)
    if "latency" in include:
        variants += latency_variants(scenario)
    if "write_fault" in include:
        variants += write_fault_variants(scenario)
    return variants
