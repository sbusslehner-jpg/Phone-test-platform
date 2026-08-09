"""Voice-mode evaluation (concept sections 13-14 & 18) — phase 3.

In text mode this evaluator returns ``None`` (no voice metrics). In voice mode
it derives barge-in latency, audio loss and an overall voice score from the
voice lifecycle events on the trace. The MVP ships the structure and the SLA
check (barge-in detection < 300 ms, section 14) so voice tests slot in without
reshaping the result model.
"""

from __future__ import annotations

from ..models import Event, VoiceMetrics

#: Barge-in detection SLA in milliseconds (section 14).
BARGE_IN_SLA_MS = 300


class VoiceEvaluator:
    """Compute :class:`VoiceMetrics` from voice lifecycle events."""

    def evaluate(self, events: list[Event], *, mode: str = "text") -> VoiceMetrics | None:
        if mode != "voice":
            return None

        barge_events = [e for e in events if e.type == "barge_in_detected"]
        metrics = VoiceMetrics()
        if barge_events:
            metrics.barge_in_detected = True
            # ``stop_latency_ms`` / ``user_audio_lost_ms`` are carried on the event
            # payload by the (future) audio layer; default to None if absent.
            first = barge_events[0]
            metrics.stop_latency_ms = first.payload.get("stop_latency_ms")
            metrics.user_audio_lost_ms = first.payload.get("user_audio_lost_ms")

        metrics.score = self._score(metrics)
        return metrics

    @staticmethod
    def _score(metrics: VoiceMetrics) -> float:
        score = 1.0
        if metrics.barge_in_detected is False:
            score -= 0.5
        if metrics.stop_latency_ms is not None and metrics.stop_latency_ms > BARGE_IN_SLA_MS:
            score -= 0.3
        if metrics.stt_wer is not None:
            score -= min(0.5, metrics.stt_wer)
        return max(0.0, round(score, 3))
