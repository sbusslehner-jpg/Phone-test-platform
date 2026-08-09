"""Named audio test profiles (concept section 13).

The concept lists the profiles a phonebot must survive::

    clean · street · office · car · restaurant · station · wind
    bad_connection · low_volume · fast_speaker · slow_speaker

Each is a :class:`~phonebot_qa.audio.chaos.ChaosConfig`. Scenarios reference them
by name (``audio: {profile: street}``), so a noise sweep is just a list of
profile names in a suite definition.
"""

from __future__ import annotations

from .chaos import ChaosConfig

#: Noise colours used by the profiles.
NOISE_PROFILES = ("white", "pink", "brown", "babble", "hum")

PROFILES: dict[str, ChaosConfig] = {
    "clean": ChaosConfig(name="clean"),
    "street": ChaosConfig(name="street", snr_db=10.0, noise="pink", volume=0.95),
    "office": ChaosConfig(name="office", snr_db=20.0, noise="babble"),
    "car": ChaosConfig(name="car", snr_db=12.0, noise="brown", volume=0.9),
    "restaurant": ChaosConfig(name="restaurant", snr_db=8.0, noise="babble"),
    "station": ChaosConfig(
        name="station", snr_db=6.0, noise="pink", packet_loss=0.01, volume=0.9
    ),
    "wind": ChaosConfig(name="wind", snr_db=7.0, noise="brown", volume=1.1),
    "bad_connection": ChaosConfig(
        name="bad_connection",
        snr_db=14.0,
        noise="white",
        packet_loss=0.05,
        latency_ms=180,
        jitter_ms=60,
    ),
    "low_volume": ChaosConfig(name="low_volume", volume=0.35, snr_db=18.0, noise="hum"),
    "fast_speaker": ChaosConfig(name="fast_speaker", speed=1.25),
    "slow_speaker": ChaosConfig(name="slow_speaker", speed=0.8),
}

#: A reasonable nightly sweep (concept §28: nightly = full regression + noise).
NIGHTLY_SWEEP = ("clean", "street", "car", "restaurant", "bad_connection", "fast_speaker")


def get_profile(name: str) -> ChaosConfig:
    """Look up a named profile, raising a helpful error for typos."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown audio profile {name!r}; available: {', '.join(sorted(PROFILES))}"
        ) from None
