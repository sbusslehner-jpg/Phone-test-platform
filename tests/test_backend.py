"""Mock backend, tool registry, tool proxy, fault injection and authorization."""

from __future__ import annotations

import pytest

from phonebot_qa.backend import FaultInjector, ToolProxy, World, default_registry
from phonebot_qa.backend.proxy import ToolFaultError
from phonebot_qa.backend.tools import ToolError
from phonebot_qa.observability import EventLog


def _proxy(world=None, faults=None):
    world = world or World(
        {
            "session_customer_id": "c1",
            "customers": [{"id": "c1"}],
            "appointments": [{"id": "a1", "customer_id": "c1", "datetime": "2026-08-12T14:00:00+02:00"}],
        }
    )
    log = EventLog()
    return ToolProxy(default_registry(), world, log, faults), world, log


def test_world_seeds_collections_and_scalars():
    w = World({"busy_slots": ["x", "y"], "appointments": [{"id": "a1", "datetime": "t"}]})
    assert w.get("appointments", "a1")["datetime"] == "t"
    assert w.snapshot()["busy_slots"] == ["x", "y"]


def test_appointment_update_changes_state_and_emits_event():
    proxy, world, log = _proxy()
    res = proxy.call("appointment.update", {"appointment_id": "a1", "datetime": "2026-08-14T10:00:00+02:00"})
    assert res["status"] == "updated"
    assert world.get("appointments", "a1")["datetime"] == "2026-08-14T10:00:00+02:00"
    assert log.has("appointment_updated")
    assert proxy.count("appointment.update") == 1


def test_fault_injection_raises_and_records():
    proxy, _, log = _proxy(faults=FaultInjector({"appointment.update": {"fail": True, "kind": "timeout"}}, seed=1))
    with pytest.raises(ToolFaultError):
        proxy.call("appointment.update", {"appointment_id": "a1", "datetime": "t"})
    assert log.has("tool_fault_injected")
    assert any(c.status == "fault" for c in proxy.calls)


def test_first_request_fault_only_affects_first_call():
    faults = FaultInjector({"appointment": {"first_request": {"fail": True, "status": 500}}}, seed=1)
    proxy, world, _ = _proxy(faults=faults)
    # First 'appointment' service call faults...
    with pytest.raises(ToolFaultError):
        proxy.call("appointment.update", {"appointment_id": "a1", "datetime": "t1"})
    # ...second succeeds.
    res = proxy.call("appointment.update", {"appointment_id": "a1", "datetime": "t2"})
    assert res["status"] == "updated"


def test_unauthorized_cross_customer_access_blocked():
    proxy, _, log = _proxy()
    with pytest.raises(ToolError):
        proxy.call("customer.lookup", {"customer_id": "other"})
    assert log.has("unauthorized_data_access_attempt")


def test_latency_advances_logical_clock():
    proxy, _, log = _proxy(faults=FaultInjector({"appointment.update": {"latency_ms": 3000}}, seed=0))
    proxy.call("appointment.update", {"appointment_id": "a1", "datetime": "t"})
    # base 250ms + 3000ms injected latency.
    assert log.clock.now_ms >= 3000
