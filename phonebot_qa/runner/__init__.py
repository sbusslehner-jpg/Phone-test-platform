"""Conversation execution (concept sections 7 & 12).

Two runners share one contract (:class:`RunArtifacts`), so the evaluation
pipeline, regression store and release gate are identical for both test levels:

``ConversationRunner``
    Text mode (section 12.1) — fast, cheap, massively parallel. The bulk of the
    suite.

``VoiceConversationRunner``
    Voice end-to-end (section 12.2) — the same scenarios executed as real calls
    through TTS, the audio chaos layer, SIP/WebRTC and the bot's STT.
"""

from __future__ import annotations

from .conversation import ConversationRunner, RunArtifacts, RunnerConfig
from .voice import VoiceConfig, VoiceConversationRunner, build_transport

__all__ = [
    "ConversationRunner",
    "RunArtifacts",
    "RunnerConfig",
    "VoiceConfig",
    "VoiceConversationRunner",
    "build_transport",
]
