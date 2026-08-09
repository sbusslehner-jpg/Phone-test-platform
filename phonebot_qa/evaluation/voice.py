"""Voice evaluation (concept sections 12.2, 14 & 18).

In text mode this returns ``None``. In voice mode it derives the concept's voice
metrics from the trace and the runner's voice artifacts:

* **barge-in**: detected at all, stop latency, caller audio lost, SLA compliance
  (detection < 300 ms, section 14);
* **recognition**: mean word-error rate across the call;
* **latency decomposition**: STT / think+tool / TTS / bot-audio budgets from the
  voice lifecycle events (section 18);
* an overall ``voice`` score feeding the weighted total (section 27).
"""

from __future__ import annotations

from ..models import Event, VoiceMetrics

#: Concept §14 SLA: barge-in must be detected within this budget.
BARGE_IN_SLA_MS = 300
#: WER at or above which recognition is considered to have failed the call.
WER_FAIL_THRESHOLD = 0.5


def latency_breakdown(events: list[Event]) -> dict[str, int]:
    """Decompose the voice latency budget from lifecycle events (§18).

    Returns cumulative milliseconds spent in each stage across the whole call::

        {"stt": 420, "llm": 1850, "tool": 870, "tts": 650, "bot_audio": 3100}
    """
    totals = {"stt": 0, "llm": 0, "tool": 0, "tts": 0, "bot_audio": 0}
    open_at: dict[str, int] = {}
    pairs = {
        "stt_started": ("stt_finished", "stt"),
        "tts_started": ("tts_finished", "tts"),
        "tool_called": ("tool_result", "tool"),
        "bot_processing_started": ("bot_message", "llm"),
        "bot_audio_started": ("bot_audio_finished", "bot_audio"),
    }
    ends = {end: (bucket, start) for start, (end, bucket) in pairs.items()}

    for event in events:
        if event.type in pairs:
            open_at[event.type] = event.t_ms
        elif event.type in ends:
            bucket, start_type = ends[event.type]
            start = open_at.pop(start_type, None)
            if start is not None:
                totals[bucket] += max(0, event.t_ms - start)
    # The LLM bucket includes tool time (tools run inside bot processing);
    # subtract it so the stages sum to the total rather than double-counting.
    totals["llm"] = max(0, totals["llm"] - totals["tool"])
    return totals


class VoiceEvaluator:
    """Compute :class:`VoiceMetrics` for a voice-mode run."""

    def evaluate(
        self,
        events: list[Event],
        *,
        mode: str = "text",
        artifacts_metadata: dict | None = None,
    ) -> VoiceMetrics | None:
        if mode != "voice":
            return None

        meta = artifacts_metadata or {}
        metrics = VoiceMetrics()

        # -- barge-in (section 14) ----------------------------------------- #
        barge_records = meta.get("barge_in") or []
        barge_events = [e for e in events if e.type == "barge_in_detected"]
        attempted = bool(barge_records) or bool(barge_events) or any(
            e.type in ("interrupt_start", "barge_in_missed") for e in events
        )
        if attempted:
            if barge_records:
                first = barge_records[0]
                metrics.barge_in_detected = bool(first.get("barge_in_detected"))
                metrics.stop_latency_ms = first.get("stop_latency_ms")
                metrics.user_audio_lost_ms = first.get("user_audio_lost_ms")
            elif barge_events:
                payload = barge_events[0].payload
                metrics.barge_in_detected = True
                metrics.stop_latency_ms = payload.get("stop_latency_ms")
                metrics.user_audio_lost_ms = payload.get("user_audio_lost_ms")
            else:
                metrics.barge_in_detected = False

        # -- recognition ---------------------------------------------------- #
        wer = meta.get("stt_wer")
        if wer is None:
            wers = [
                e.payload.get("wer")
                for e in events
                if e.type == "stt_finished" and e.payload.get("wer") is not None
            ]
            wer = sum(wers) / len(wers) if wers else None
        metrics.stt_wer = round(float(wer), 4) if wer is not None else None

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
            # A perfect transcript costs nothing; 50%+ WER costs the full 0.5.
            score -= min(0.5, metrics.stt_wer)
        return max(0.0, round(score, 3))
