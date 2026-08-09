"""Mock backend, tool registry, tool proxy and fault injection.

Concept sections 15-17: phonebot tests never run against production. A defined
world state is created before each test, all tool calls flow through a proxy
that logs them and can inject faults, and the final state is asserted on.
"""

from __future__ import annotations

from .faults import FaultInjector, FaultResult
from .proxy import ToolProxy
from .tools import Tool, ToolRegistry, default_registry
from .world import World

__all__ = [
    "FaultInjector",
    "FaultResult",
    "Tool",
    "ToolProxy",
    "ToolRegistry",
    "World",
    "default_registry",
]
