"""Core domain models — the shared contract for the whole platform.

These Pydantic models are the *stable interface* between every component:
scenario loading, user simulation, conversation execution, tool proxying,
event logging, deterministic assertions, LLM/voice evaluation and scoring all
speak these types. The concept document (section 25) lists the conceptual data
model; this module is its runtime realisation.

Design rules that show up here:
* The **scenario is the source of truth** (section 5).
* **Knowledge isolation** (section 10): the user simulator only ever sees
  ``Scenario.user_visible`` — never ``evaluator_only`` — so evaluation
  expectations can never leak into the simulated caller.
* **Deterministic truth first** (section 3 / 37): backend state, tool calls and
  events are structured, comparable data; LLM judgements are separate and never
  override a critical deterministic failure on their own.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Personas (section 9)                                                         #
# --------------------------------------------------------------------------- #


class Persona(BaseModel):
    """A reusable caller personality, independent of the business scenario.

    Persona and scenario are orthogonal: any persona may be combined with any
    scenario (section 9). Probabilities drive the (seeded, reproducible)
    behaviour of the user simulator.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    # Qualitative traits fed to the LLM simulator's system prompt.
    patience: Literal["low", "medium", "high"] = "medium"
    verbosity: Literal["short", "medium", "long"] = "medium"
    tech_savvy: Literal["low", "medium", "high"] = "medium"
    language_proficiency: Literal["low", "medium", "high"] = "high"
    # Quantitative behaviour knobs (0..1), consumed by simulators deterministically.
    interruption_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    correction_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    off_topic_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Scenario (sections 5 & 10)                                                   #
# --------------------------------------------------------------------------- #


class UserGoal(BaseModel):
    """What the simulated caller is trying to achieve."""

    model_config = ConfigDict(extra="allow")

    type: str
    # Free-form, goal-specific parameters (target_datetime, appointment_id, ...).
    # Declared ``extra=allow`` so scenario authors are not boxed in.


class ScenarioLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=15, ge=1)
    max_duration_seconds: int = Field(default=180, ge=1)


class ExpectedOutcome(BaseModel):
    """The evaluator-only contract: what a correct run must (not) do.

    This is deliberately *not* visible to the user simulator (section 10).
    """

    model_config = ConfigDict(extra="forbid")

    # Expected backend state after the conversation, keyed by
    # "<collection>.<id>" -> {field: value}. Compared field-by-field.
    database: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Events that MUST appear at least once.
    required_events: list[str] = Field(default_factory=list)
    # Events that must NEVER appear.
    forbidden_events: list[str] = Field(default_factory=list)
    # Exact tool-call counts, keyed by tool name. Absent tool => not constrained.
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    # Safety invariants that must hold (evaluated by named checks, section 27).
    safety_invariants: list[str] = Field(default_factory=list)


class UserSpec(BaseModel):
    """The user side of a scenario: goal, persona (by id) and visible knowledge."""

    model_config = ConfigDict(extra="forbid")

    goal: UserGoal
    persona: str | None = None
    # Everything a real caller would plausibly know. ONLY this reaches the simulator.
    user_visible: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    """The central abstraction of the platform (section 5).

    A scenario fully specifies a test: starting world state, the caller's goal
    and knowledge, the expected outcome (evaluator-only), and technical limits.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    # Initial backend/world state (customers, appointments, ...), section 15.
    initial_state: dict[str, Any] = Field(default_factory=dict)
    user: UserSpec
    expected: ExpectedOutcome = Field(default_factory=ExpectedOutcome)
    limits: ScenarioLimits = Field(default_factory=ScenarioLimits)
    # Optional fault-injection config for the tool proxy (section 17).
    faults: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_id(self) -> "Scenario":
        if not re.fullmatch(r"[a-zA-Z0-9_.\-]+", self.id):
            raise ValueError(
                f"scenario id {self.id!r} must match [a-zA-Z0-9_.-]+"
            )
        return self


# --------------------------------------------------------------------------- #
# Events & tool calls (sections 16 & 18)                                       #
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    """Canonical structured event names (section 18).

    Kept as an enum for the well-known lifecycle events, but the event log also
    accepts arbitrary string event names so business/bot code can emit its own
    domain events (e.g. ``availability_checked``) that scenarios assert on.
    """

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    USER_MESSAGE = "user_message"
    BOT_PROCESSING_STARTED = "bot_processing_started"
    BOT_MESSAGE = "bot_message"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    TOOL_FAULT_INJECTED = "tool_fault_injected"
    TURN_COMPLETED = "turn_completed"
    LIMIT_REACHED = "limit_reached"
    # Voice lifecycle (phase 3, section 18) — recorded when in voice mode.
    STT_STARTED = "stt_started"
    STT_FINISHED = "stt_finished"
    TTS_STARTED = "tts_started"
    TTS_FINISHED = "tts_finished"
    BARGE_IN_DETECTED = "barge_in_detected"


class Event(BaseModel):
    """A single structured, timestamped event on the trace (section 16/18)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    # Monotonic logical time in milliseconds from session start (deterministic,
    # not wall-clock, so runs are reproducible and latencies are exact).
    t_ms: int = 0
    turn: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A recorded tool invocation flowing through the tool proxy (section 16)."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    status: Literal["success", "error", "fault"] = "success"
    duration_ms: int = 0
    turn: int | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Conversation (sections 7 & 25)                                              #
# --------------------------------------------------------------------------- #


class Turn(BaseModel):
    """One user/bot exchange."""

    model_config = ConfigDict(extra="forbid")

    index: int
    user: str
    bot: str
    latency_ms: int = 0


class Conversation(BaseModel):
    """The full transcript plus its turn count and duration."""

    model_config = ConfigDict(extra="forbid")

    turns: list[Turn] = Field(default_factory=list)
    duration_ms: int = 0

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def as_history(self) -> list[dict[str, str]]:
        """Flat chat history for simulators / judges (role-tagged messages)."""
        history: list[dict[str, str]] = []
        for turn in self.turns:
            history.append({"role": "user", "content": turn.user})
            history.append({"role": "assistant", "content": turn.bot})
        return history


# --------------------------------------------------------------------------- #
# Evaluation results (sections 19-21, 26-27)                                   #
# --------------------------------------------------------------------------- #


class AssertionResult(BaseModel):
    """Outcome of one deterministic assertion (section 20)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Category drives scoring weight & critical-override (section 27).
    category: Literal["business", "safety", "tool", "technical"]
    passed: bool
    critical: bool = False
    detail: str = ""


class EvalScores(BaseModel):
    """Qualitative LLM-judge scores, 1..5 (section 21)."""

    model_config = ConfigDict(extra="forbid")

    naturalness: float | None = None
    clarity: float | None = None
    efficiency: float | None = None
    correction_handling: float | None = None
    rationale: str = ""

    def normalized(self) -> float | None:
        """Average of present scores, scaled to 0..1 (None if no scores)."""
        vals = [
            v
            for v in (
                self.naturalness,
                self.clarity,
                self.efficiency,
                self.correction_handling,
            )
            if v is not None
        ]
        if not vals:
            return None
        return (sum(vals) / len(vals)) / 5.0


class VoiceMetrics(BaseModel):
    """Voice-mode metrics (phase 3, sections 14/18). Optional in MVP."""

    model_config = ConfigDict(extra="forbid")

    barge_in_detected: bool | None = None
    stop_latency_ms: int | None = None
    user_audio_lost_ms: int | None = None
    stt_wer: float | None = None
    score: float | None = None


class LatencyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turns: int = 0
    duration_seconds: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


class ScoreBreakdown(BaseModel):
    """Weighted score components (section 27)."""

    model_config = ConfigDict(extra="forbid")

    business: float = 0.0
    safety: float = 0.0
    conversation: float = 0.0
    latency: float = 0.0
    voice: float = 0.0
    total: float = 0.0


class CaseResult(BaseModel):
    """The complete result of running one test case (section 26).

    A "case" is one concrete (scenario × persona × seed × mode × bot version)
    combination produced by the orchestrator (section 6).
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    scenario_id: str
    scenario_tags: list[str] = Field(default_factory=list)
    persona_id: str | None = None
    bot_version: str = "unknown"
    mode: Literal["text", "voice"] = "text"
    seed: int = 0

    result: Literal["PASS", "FAIL", "ERROR"] = "FAIL"
    critical_failure: str | None = None

    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    eval_scores: EvalScores | None = None
    voice: VoiceMetrics | None = None

    conversation: Conversation = Field(default_factory=Conversation)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)

    final_state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def assertion_summary(self) -> dict[str, bool]:
        """Compact ``name -> passed`` map for the report (section 26)."""
        return {a.name: a.passed for a in self.assertions}
