"""
Log Entry Data Structures for Vibe Orchestrator.

Defines structured log entries for GLM interactions, Claude executions,
and session lifecycle events.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class GLMLogEntry:
    """Log entry for GLM API interactions."""

    # Identity
    timestamp: str  # ISO 8601
    request_id: str  # UUID for correlating request/response
    session_id: str  # Vibe session ID

    # Method info
    method: str  # "chat", "decompose_task", "review_changes", etc.

    # Request
    system_prompt: str = ""
    user_prompt: str = ""
    temperature: float = 0.0
    max_tokens: int = 0

    # Response
    response_content: str = ""
    model: str = ""
    finish_reason: str = ""

    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

    # Context
    project_name: str = ""

    # Error (if any)
    error: str | None = None
    error_type: str | None = None

    # Cost estimation (calculated)
    cost_usd: float = 0.0

    def estimate_cost(self) -> float:
        """
        Estimate cost based on token counts.
        GLM-4.7 pricing (approximate): $0.001/1K input, $0.002/1K output
        """
        input_cost = (self.prompt_tokens / 1000) * 0.001
        output_cost = (self.completion_tokens / 1000) * 0.002
        self.cost_usd = round(input_cost + output_cost, 6)
        return self.cost_usd

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GLMLogEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClaudeLogEntry:
    """Log entry for Claude CLI executions."""

    # Identity
    timestamp: str  # ISO 8601
    execution_id: str  # UUID
    session_id: str  # Vibe session ID
    task_id: str = ""  # From task decomposition

    # Input
    prompt: str = ""  # Full prompt sent to Claude
    files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    timeout_tier: str = ""
    allowed_tools: list[str] = field(default_factory=list)

    # Output
    result: str | None = None
    success: bool = False
    error: str | None = None

    # Tool usage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)

    # Metrics
    duration_ms: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0

    # Context
    project_path: str = ""
    attempt_number: int = 1
    previous_feedback: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaudeLogEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionLogEntry:
    """Log entry for session lifecycle events."""

    timestamp: str  # ISO 8601
    session_id: str
    event_type: str  # "start", "state_change", "request", "end", "error"

    # Event-specific fields
    from_state: str | None = None
    to_state: str | None = None
    user_request: str = ""
    project_name: str = ""

    # Session end metrics (populated on "end" event)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_glm_tokens: int = 0
    total_duration_seconds: float = 0.0

    # Error info (populated on "error" event)
    error: str | None = None
    error_type: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionLogEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GeminiLogEntry:
    """Log entry for Gemini (brain/orchestrator) API interactions."""

    # Identity
    timestamp: str  # ISO 8601
    request_id: str  # UUID for correlating request/response
    session_id: str  # Vibe session ID

    # Method info
    method: str  # "chat", "decompose_task", "check_clarification", etc.

    # Request
    system_prompt: str = ""
    user_prompt: str = ""
    temperature: float = 0.0
    max_tokens: int = 0

    # Response
    response_content: str = ""
    model: str = ""
    finish_reason: str = ""

    # Parsed output (for decompose_task)
    tasks_generated: int = 0
    task_descriptions: list[str] = field(default_factory=list)

    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

    # Context
    project_name: str = ""
    user_request: str = ""  # Original user request (for debugging prompt quality)

    # Error (if any)
    error: str | None = None
    error_type: str | None = None

    # Cost estimation
    cost_usd: float = 0.0

    def estimate_cost(self) -> float:
        """
        Estimate cost based on token counts.
        Gemini 2.0 Flash pricing (approximate): $0.0001/1K input, $0.0004/1K output
        """
        input_cost = (self.prompt_tokens / 1000) * 0.0001
        output_cost = (self.completion_tokens / 1000) * 0.0004
        self.cost_usd = round(input_cost + output_cost, 6)
        return self.cost_usd

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeminiLogEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def now_iso() -> str:
    """Get current time as ISO 8601 string."""
    return datetime.now().isoformat()
