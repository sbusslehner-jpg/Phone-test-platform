"""Voice transports (concept sections 11, 12.2 & 34).

The transport carries audio between the simulated caller and the phonebot. It is
an interface so the same voice scenarios can run over an in-process loopback (CI),
a SIP trunk, or a WebRTC peer connection — exactly the adapter list in section 11.

The bundled implementations model the *network behaviour* that matters to a voice
agent (one-way latency, jitter, packet loss, codec band-limiting) deterministically,
so voice tests are reproducible without a PBX. Real SIP/WebRTC stacks
(``pjsua2``/``aiortc``) plug in behind the same interface.
"""

from __future__ import annotations

from .base import TransportStats, VoiceTransport
from .loopback import LoopbackTransport
from .sip import SIPTransport
from .webrtc import WebRTCTransport

__all__ = [
    "LoopbackTransport",
    "SIPTransport",
    "TransportStats",
    "VoiceTransport",
    "WebRTCTransport",
]
