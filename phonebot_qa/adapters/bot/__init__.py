"""Bot adapters — keep the platform independent of any specific phonebot."""

from __future__ import annotations

from .base import BotAdapter, BotResponse, BotSession
from .reference import ReferenceAppointmentBot
from .rest import RESTBotAdapter

__all__ = [
    "BotAdapter",
    "BotResponse",
    "BotSession",
    "ReferenceAppointmentBot",
    "RESTBotAdapter",
]
