"""Adapters that connect the platform to a phonebot under test (section 11)."""

from __future__ import annotations

from .bot.base import BotAdapter, BotResponse, BotSession

__all__ = ["BotAdapter", "BotResponse", "BotSession"]
