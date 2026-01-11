"""
Vibe Persistence Models

Dataclasses that map to SQLite tables for the Vibe Orchestrator.
Designed for:
- Type safety with enums and Optional types
- Easy serialization to/from database rows
- JSON field handling for nested data
- Immutable snapshots with frozen option where appropriate
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ============================================================================
# ENUMS - Type-safe status values matching SQL schema
# ============================================================================


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    CRASHED = "crashed"
    COMPLETED = "completed"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Task lifecycle status."""

    PENDING = "pending"
    QUEUED = "queued"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class AttemptResult(str, Enum):
    """Result of a Claude execution attempt."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    PARTIAL = "partial"


class MessageRole(str, Enum):
    """Role of a message in conversation."""

    USER = "user"
    GLM = "glm"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class MessageType(str, Enum):
    """Type of message for categorization."""

    CHAT = "chat"
    CLARIFICATION = "clarification"
    DECOMPOSITION = "decomposition"
    REVIEW = "review"
    ERROR = "error"
    STATUS = "status"


class ChangeType(str, Enum):
    """Type of file change."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class ContextCategory(str, Enum):
    """Category for context items."""

    TASK = "task"
    DECISION = "decision"
    PROGRESS = "progress"
    NOTE = "note"
    ERROR = "error"
    WARNING = "warning"
    CONVENTION = "convention"


class Priority(str, Enum):
    """Priority level for context items."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Get current datetime as ISO string."""
    return datetime.now().isoformat()


def parse_json_or_list(value: str | list | None) -> list:
    """Parse JSON string to list, or return empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def parse_json_or_dict(value: str | dict | None) -> dict:
    """Parse JSON string to dict, or return empty dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def to_json(value: list | dict | None) -> str | None:
    """Convert list or dict to JSON string."""
    if value is None:
        return None
    return json.dumps(value)


def parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse ISO datetime string to datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


# ============================================================================
# CORE ENTITIES
# ============================================================================


@dataclass
class Project:
    """
    A registered project for Vibe to manage.

    Maps to: projects table
    """

    id: str = field(default_factory=generate_id)
    name: str = ""
    path: str = ""
    starmap: str = "STARMAP.md"
    claude_md: str = "CLAUDE.md"
    test_command: str = "pytest -v"
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime | None = None
    is_active: bool = True

    @classmethod
    def from_row(cls, row: tuple) -> Project:
        """Create from database row."""
        return cls(
            id=row[0],
            name=row[1],
            path=row[2],
            starmap=row[3] or "STARMAP.md",
            claude_md=row[4] or "CLAUDE.md",
            test_command=row[5] or "pytest -v",
            description=row[6],
            created_at=parse_datetime(row[7]) or datetime.now(),
            updated_at=parse_datetime(row[8]) or datetime.now(),
            last_accessed_at=parse_datetime(row[9]),
            is_active=bool(row[10]),
        )

    def to_row(self) -> tuple:
        """Convert to database row for INSERT/UPDATE."""
        return (
            self.id,
            self.name,
            self.path,
            self.starmap,
            self.claude_md,
            self.test_command,
            self.description,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            1 if self.is_active else 0,
        )


@dataclass
class Session:
    """
    A Vibe orchestrator session.

    Maps to: sessions table
    Tracks lifecycle for crash recovery and audit.
    """

    id: str = field(default_factory=generate_id)
    project_id: str = ""
    status: SessionStatus = SessionStatus.INITIALIZING
    pid: int | None = None
    hostname: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    summary: str | None = None
    error_message: str | None = None
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_cost_usd: float = 0.0

    @classmethod
    def from_row(cls, row: tuple) -> Session:
        """Create from database row."""
        return cls(
            id=row[0],
            project_id=row[1],
            status=SessionStatus(row[2]) if row[2] else SessionStatus.INITIALIZING,
            pid=row[3],
            hostname=row[4],
            started_at=parse_datetime(row[5]) or datetime.now(),
            ended_at=parse_datetime(row[6]),
            last_heartbeat_at=parse_datetime(row[7]),
            summary=row[8],
            error_message=row[9],
            total_tasks_completed=row[10] or 0,
            total_tasks_failed=row[11] or 0,
            total_cost_usd=row[12] or 0.0,
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.project_id,
            self.status.value,
            self.pid,
            self.hostname,
            self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            self.ended_at.isoformat() if self.ended_at else None,
            self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None,
            self.summary,
            self.error_message,
            self.total_tasks_completed,
            self.total_tasks_failed,
            self.total_cost_usd,
        )

    def update_heartbeat(self) -> None:
        """Update the heartbeat timestamp."""
        self.last_heartbeat_at = datetime.now()

    def mark_active(self) -> None:
        """Mark session as active."""
        self.status = SessionStatus.ACTIVE
        self.update_heartbeat()

    def mark_completed(self, summary: str | None = None) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.ended_at = datetime.now()
        if summary:
            self.summary = summary

    def mark_error(self, error_message: str) -> None:
        """Mark session as ended with error."""
        self.status = SessionStatus.ERROR
        self.ended_at = datetime.now()
        self.error_message = error_message

    def mark_crashed(self) -> None:
        """Mark session as crashed (detected orphan)."""
        self.status = SessionStatus.CRASHED
        self.ended_at = datetime.now()
        self.error_message = "Session detected as orphaned (no heartbeat)"


@dataclass
class Message:
    """
    A message in the conversation history.

    Maps to: messages table
    Supports threading via parent_message_id.
    """

    id: str = field(default_factory=generate_id)
    session_id: str = ""
    role: MessageRole = MessageRole.USER
    content: str = ""
    message_type: MessageType | None = None
    parent_message_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    tokens_used: int | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: tuple) -> Message:
        """Create from database row."""
        return cls(
            id=row[0],
            session_id=row[1],
            role=MessageRole(row[2]) if row[2] else MessageRole.USER,
            content=row[3] or "",
            message_type=MessageType(row[4]) if row[4] else None,
            parent_message_id=row[5],
            created_at=parse_datetime(row[6]) or datetime.now(),
            tokens_used=row[7],
            cost_usd=row[8],
            metadata=parse_json_or_dict(row[9]),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.session_id,
            self.role.value,
            self.content,
            self.message_type.value if self.message_type else None,
            self.parent_message_id,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.tokens_used,
            self.cost_usd,
            to_json(self.metadata) if self.metadata else None,
        )


@dataclass
class Task:
    """
    A task to be executed by Claude.

    Maps to: tasks table
    Supports parent-child relationships for subtasks.
    """

    id: str = field(default_factory=generate_id)
    session_id: str = ""
    parent_task_id: str | None = None
    sequence_num: int = 0
    description: str = ""
    files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_criteria: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_by: str | None = None
    original_request: str | None = None

    @classmethod
    def from_row(cls, row: tuple) -> Task:
        """Create from database row."""
        return cls(
            id=row[0],
            session_id=row[1],
            parent_task_id=row[2],
            sequence_num=row[3] or 0,
            description=row[4] or "",
            files=parse_json_or_list(row[5]),
            constraints=parse_json_or_list(row[6]),
            success_criteria=row[7],
            status=TaskStatus(row[8]) if row[8] else TaskStatus.PENDING,
            priority=row[9] or 0,
            created_at=parse_datetime(row[10]) or datetime.now(),
            started_at=parse_datetime(row[11]),
            completed_at=parse_datetime(row[12]),
            created_by=row[13],
            original_request=row[14],
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.session_id,
            self.parent_task_id,
            self.sequence_num,
            self.description,
            to_json(self.files),
            to_json(self.constraints),
            self.success_criteria,
            self.status.value,
            self.priority,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.started_at.isoformat() if self.started_at else None,
            self.completed_at.isoformat() if self.completed_at else None,
            self.created_by,
            self.original_request,
        )

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.EXECUTING
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "files": self.files,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
        }


@dataclass
class TaskStatusTransition:
    """
    Record of a task status change.

    Maps to: task_status_transitions table
    Provides full audit trail of task lifecycle.
    """

    id: int | None = None  # Auto-increment
    task_id: str = ""
    from_status: TaskStatus | None = None
    to_status: TaskStatus = TaskStatus.PENDING
    reason: str | None = None
    transitioned_at: datetime = field(default_factory=datetime.now)
    triggered_by: str | None = None

    @classmethod
    def from_row(cls, row: tuple) -> TaskStatusTransition:
        """Create from database row."""
        return cls(
            id=row[0],
            task_id=row[1],
            from_status=TaskStatus(row[2]) if row[2] else None,
            to_status=TaskStatus(row[3]) if row[3] else TaskStatus.PENDING,
            reason=row[4],
            transitioned_at=parse_datetime(row[5]) or datetime.now(),
            triggered_by=row[6],
        )

    def to_row(self) -> tuple:
        """Convert to database row (without id for INSERT)."""
        return (
            self.task_id,
            self.from_status.value if self.from_status else None,
            self.to_status.value,
            self.reason,
            self.transitioned_at.isoformat() if isinstance(self.transitioned_at, datetime) else self.transitioned_at,
            self.triggered_by,
        )


@dataclass
class TaskAttempt:
    """
    A single Claude execution attempt for a task.

    Maps to: task_attempts table
    Stores full prompt and response for debugging.
    """

    id: str = field(default_factory=generate_id)
    task_id: str = ""
    attempt_num: int = 1

    # Input
    prompt: str = ""
    timeout_tier: str = "code"
    allowed_tools: list[str] = field(default_factory=list)

    # Output
    result: AttemptResult = AttemptResult.PENDING
    response_text: str | None = None
    error_message: str | None = None
    summary: str | None = None

    # Metrics
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: int | None = None
    cost_usd: float = 0.0
    tokens_used: int | None = None
    num_turns: int = 0
    claude_session_id: str | None = None

    # Tool tracking
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: tuple) -> TaskAttempt:
        """Create from database row."""
        return cls(
            id=row[0],
            task_id=row[1],
            attempt_num=row[2] or 1,
            prompt=row[3] or "",
            timeout_tier=row[4] or "code",
            allowed_tools=parse_json_or_list(row[5]),
            result=AttemptResult(row[6]) if row[6] else AttemptResult.PENDING,
            response_text=row[7],
            error_message=row[8],
            summary=row[9],
            started_at=parse_datetime(row[10]) or datetime.now(),
            completed_at=parse_datetime(row[11]),
            duration_ms=row[12],
            cost_usd=row[13] or 0.0,
            tokens_used=row[14],
            num_turns=row[15] or 0,
            claude_session_id=row[16],
            tool_calls=parse_json_or_list(row[17]),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.task_id,
            self.attempt_num,
            self.prompt,
            self.timeout_tier,
            to_json(self.allowed_tools),
            self.result.value,
            self.response_text,
            self.error_message,
            self.summary,
            self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            self.completed_at.isoformat() if self.completed_at else None,
            self.duration_ms,
            self.cost_usd,
            self.tokens_used,
            self.num_turns,
            self.claude_session_id,
            to_json(self.tool_calls),
        )

    def complete_success(self, response: str, summary: str | None = None, duration_ms: int = 0) -> None:
        """Mark attempt as successful."""
        self.result = AttemptResult.SUCCESS
        self.response_text = response
        self.summary = summary
        self.completed_at = datetime.now()
        self.duration_ms = duration_ms

    def complete_failure(self, error: str, duration_ms: int = 0) -> None:
        """Mark attempt as failed."""
        self.result = AttemptResult.FAILED
        self.error_message = error
        self.completed_at = datetime.now()
        self.duration_ms = duration_ms

    def complete_timeout(self, timeout_seconds: int) -> None:
        """Mark attempt as timed out."""
        self.result = AttemptResult.TIMEOUT
        self.error_message = f"Timed out after {timeout_seconds}s"
        self.completed_at = datetime.now()

    def complete_rejected(self, feedback: str) -> None:
        """Mark attempt as rejected by GLM review."""
        self.result = AttemptResult.REJECTED
        self.error_message = f"Rejected by review: {feedback}"
        self.completed_at = datetime.now()


@dataclass
class FileChange:
    """
    A file change made during a task attempt.

    Maps to: file_changes table
    """

    id: int | None = None  # Auto-increment
    attempt_id: str = ""
    file_path: str = ""
    change_type: ChangeType = ChangeType.MODIFY
    old_path: str | None = None
    diff_content: str | None = None
    lines_added: int | None = None
    lines_removed: int | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_row(cls, row: tuple) -> FileChange:
        """Create from database row."""
        return cls(
            id=row[0],
            attempt_id=row[1],
            file_path=row[2],
            change_type=ChangeType(row[3]) if row[3] else ChangeType.MODIFY,
            old_path=row[4],
            diff_content=row[5],
            lines_added=row[6],
            lines_removed=row[7],
            created_at=parse_datetime(row[8]) or datetime.now(),
        )

    def to_row(self) -> tuple:
        """Convert to database row (without id for INSERT)."""
        return (
            self.attempt_id,
            self.file_path,
            self.change_type.value,
            self.old_path,
            self.diff_content,
            self.lines_added,
            self.lines_removed,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        )


@dataclass
class Review:
    """
    GLM review of a task attempt.

    Maps to: reviews table
    """

    id: str = field(default_factory=generate_id)
    attempt_id: str = ""
    approved: bool = False
    issues: list[str] = field(default_factory=list)
    feedback: str = ""
    suggested_next_steps: list[str] = field(default_factory=list)
    reviewed_at: datetime = field(default_factory=datetime.now)
    review_duration_ms: int | None = None
    tokens_used: int | None = None

    @classmethod
    def from_row(cls, row: tuple) -> Review:
        """Create from database row."""
        return cls(
            id=row[0],
            attempt_id=row[1],
            approved=bool(row[2]),
            issues=parse_json_or_list(row[3]),
            feedback=row[4] or "",
            suggested_next_steps=parse_json_or_list(row[5]),
            reviewed_at=parse_datetime(row[6]) or datetime.now(),
            review_duration_ms=row[7],
            tokens_used=row[8],
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.attempt_id,
            1 if self.approved else 0,
            to_json(self.issues),
            self.feedback,
            to_json(self.suggested_next_steps),
            self.reviewed_at.isoformat() if isinstance(self.reviewed_at, datetime) else self.reviewed_at,
            self.review_duration_ms,
            self.tokens_used,
        )


# ============================================================================
# DEBUG ENTITIES
# ============================================================================


@dataclass
class DebugSession:
    """
    An extended debugging workflow session.

    Maps to: debug_sessions table
    Tracks multi-iteration debugging with hypothesis management.
    """

    id: str = field(default_factory=generate_id)
    session_id: str = ""
    problem: str = ""
    hypothesis: str | None = None
    must_preserve: list[str] = field(default_factory=list)
    is_active: bool = True
    is_solved: bool = False
    initial_git_commit: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None

    @classmethod
    def from_row(cls, row: tuple) -> DebugSession:
        """Create from database row."""
        return cls(
            id=row[0],
            session_id=row[1],
            problem=row[2] or "",
            hypothesis=row[3],
            must_preserve=parse_json_or_list(row[4]),
            is_active=bool(row[5]),
            is_solved=bool(row[6]),
            initial_git_commit=row[7],
            created_at=parse_datetime(row[8]) or datetime.now(),
            updated_at=parse_datetime(row[9]) or datetime.now(),
            resolved_at=parse_datetime(row[10]),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.session_id,
            self.problem,
            self.hypothesis,
            to_json(self.must_preserve),
            1 if self.is_active else 0,
            1 if self.is_solved else 0,
            self.initial_git_commit,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            self.resolved_at.isoformat() if self.resolved_at else None,
        )

    def solve(self) -> None:
        """Mark debug session as solved."""
        self.is_solved = True
        self.is_active = False
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()

    def close(self) -> None:
        """Close debug session without solving."""
        self.is_active = False
        self.updated_at = datetime.now()


@dataclass
class DebugIteration:
    """
    A single iteration within a debug session.

    Maps to: debug_iterations table
    Contains task, output, and review in one record.
    """

    id: str = field(default_factory=generate_id)
    debug_session_id: str = ""
    iteration_num: int = 1

    # Task given to Claude
    task_description: str = ""
    starting_points: list[str] = field(default_factory=list)
    what_to_look_for: str | None = None
    success_criteria: str | None = None

    # Claude's output
    output: str | None = None
    files_examined: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    structured_findings: dict[str, Any] = field(default_factory=dict)

    # GLM review
    review_approved: bool | None = None
    review_is_solved: bool | None = None
    review_feedback: str | None = None
    review_next_task: str | None = None

    # Metrics
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: int | None = None

    # Rollback support
    git_checkpoint: str | None = None

    @classmethod
    def from_row(cls, row: tuple) -> DebugIteration:
        """Create from database row."""
        return cls(
            id=row[0],
            debug_session_id=row[1],
            iteration_num=row[2] or 1,
            task_description=row[3] or "",
            starting_points=parse_json_or_list(row[4]),
            what_to_look_for=row[5],
            success_criteria=row[6],
            output=row[7],
            files_examined=parse_json_or_list(row[8]),
            files_changed=parse_json_or_list(row[9]),
            structured_findings=parse_json_or_dict(row[10]),
            review_approved=bool(row[11]) if row[11] is not None else None,
            review_is_solved=bool(row[12]) if row[12] is not None else None,
            review_feedback=row[13],
            review_next_task=row[14],
            started_at=parse_datetime(row[15]) or datetime.now(),
            completed_at=parse_datetime(row[16]),
            duration_ms=row[17],
            git_checkpoint=row[18],
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.debug_session_id,
            self.iteration_num,
            self.task_description,
            to_json(self.starting_points),
            self.what_to_look_for,
            self.success_criteria,
            self.output,
            to_json(self.files_examined),
            to_json(self.files_changed),
            to_json(self.structured_findings),
            1 if self.review_approved else 0 if self.review_approved is not None else None,
            1 if self.review_is_solved else 0 if self.review_is_solved is not None else None,
            self.review_feedback,
            self.review_next_task,
            self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            self.completed_at.isoformat() if self.completed_at else None,
            self.duration_ms,
            self.git_checkpoint,
        )

    def format_summary(self) -> str:
        """Format iteration for history display."""
        files_str = ", ".join(self.files_changed) if self.files_changed else "none"
        output_preview = self.output[:500] + "..." if self.output and len(self.output) > 500 else (self.output or "")
        return (
            f"### Iteration {self.iteration_num}\n"
            f"**Task:** {self.task_description}\n"
            f"**Files Changed:** {files_str}\n"
            f"**Duration:** {self.duration_ms or 0}ms\n"
            f"**Output:**\n{output_preview}\n"
        )


# ============================================================================
# SUPPORT ENTITIES
# ============================================================================


@dataclass
class ContextItem:
    """
    Key-value storage for project context.

    Maps to: context_items table
    """

    id: str = field(default_factory=generate_id)
    project_id: str = ""
    session_id: str | None = None
    key: str = ""
    value: str = ""
    category: ContextCategory = ContextCategory.NOTE
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    @classmethod
    def from_row(cls, row: tuple) -> ContextItem:
        """Create from database row."""
        return cls(
            id=row[0],
            project_id=row[1],
            session_id=row[2],
            key=row[3] or "",
            value=row[4] or "",
            category=ContextCategory(row[5]) if row[5] else ContextCategory.NOTE,
            priority=Priority(row[6]) if row[6] else Priority.NORMAL,
            created_at=parse_datetime(row[7]) or datetime.now(),
            updated_at=parse_datetime(row[8]) or datetime.now(),
            expires_at=parse_datetime(row[9]),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.project_id,
            self.session_id,
            self.key,
            self.value,
            self.category.value,
            self.priority.value,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            self.expires_at.isoformat() if self.expires_at else None,
        )


@dataclass
class Convention:
    """
    Global convention across projects.

    Maps to: conventions table
    """

    id: str = field(default_factory=generate_id)
    key: str = ""
    convention: str = ""
    applies_to: str = "all"
    created_by_project: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    @classmethod
    def from_row(cls, row: tuple) -> Convention:
        """Create from database row."""
        return cls(
            id=row[0],
            key=row[1] or "",
            convention=row[2] or "",
            applies_to=row[3] or "all",
            created_by_project=row[4],
            created_at=parse_datetime(row[5]) or datetime.now(),
            updated_at=parse_datetime(row[6]) or datetime.now(),
            is_active=bool(row[7]),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.key,
            self.convention,
            self.applies_to,
            self.created_by_project,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            1 if self.is_active else 0,
        )


@dataclass
class Checkpoint:
    """
    Recovery checkpoint with git state.

    Maps to: checkpoints table
    """

    id: str = field(default_factory=generate_id)
    session_id: str = ""
    name: str = ""
    description: str | None = None
    git_branch: str | None = None
    git_commit: str | None = None
    git_status: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_row(cls, row: tuple) -> Checkpoint:
        """Create from database row."""
        return cls(
            id=row[0],
            session_id=row[1],
            name=row[2] or "",
            description=row[3],
            git_branch=row[4],
            git_commit=row[5],
            git_status=row[6],
            created_at=parse_datetime(row[7]) or datetime.now(),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.session_id,
            self.name,
            self.description,
            self.git_branch,
            self.git_commit,
            self.git_status,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        )


@dataclass
class ToolUsage:
    """
    Aggregate tool usage statistics per session.

    Maps to: tool_usage table
    """

    id: int | None = None  # Auto-increment
    session_id: str = ""
    tool_name: str = ""
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: int = 0
    last_used_at: datetime | None = None

    @classmethod
    def from_row(cls, row: tuple) -> ToolUsage:
        """Create from database row."""
        return cls(
            id=row[0],
            session_id=row[1],
            tool_name=row[2] or "",
            invocation_count=row[3] or 0,
            success_count=row[4] or 0,
            failure_count=row[5] or 0,
            total_duration_ms=row[6] or 0,
            last_used_at=parse_datetime(row[7]),
        )

    def to_row(self) -> tuple:
        """Convert to database row (without id for INSERT)."""
        return (
            self.session_id,
            self.tool_name,
            self.invocation_count,
            self.success_count,
            self.failure_count,
            self.total_duration_ms,
            self.last_used_at.isoformat() if self.last_used_at else None,
        )

    def record_invocation(self, success: bool, duration_ms: int = 0) -> None:
        """Record a tool invocation."""
        self.invocation_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_duration_ms += duration_ms
        self.last_used_at = datetime.now()


@dataclass
class Request:
    """
    User request record.

    Maps to: requests table
    """

    id: str = field(default_factory=generate_id)
    session_id: str = ""
    request_text: str = ""
    result_summary: str | None = None
    tasks_created: int = 0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @classmethod
    def from_row(cls, row: tuple) -> Request:
        """Create from database row."""
        return cls(
            id=row[0],
            session_id=row[1],
            request_text=row[2] or "",
            result_summary=row[3],
            tasks_created=row[4] or 0,
            status=row[5] or "pending",
            created_at=parse_datetime(row[6]) or datetime.now(),
            completed_at=parse_datetime(row[7]),
        )

    def to_row(self) -> tuple:
        """Convert to database row."""
        return (
            self.id,
            self.session_id,
            self.request_text,
            self.result_summary,
            self.tasks_created,
            self.status,
            self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            self.completed_at.isoformat() if self.completed_at else None,
        )

    def complete(self, summary: str) -> None:
        """Mark request as completed."""
        self.status = "completed"
        self.result_summary = summary
        self.completed_at = datetime.now()

    def fail(self, reason: str) -> None:
        """Mark request as failed."""
        self.status = "failed"
        self.result_summary = reason
        self.completed_at = datetime.now()
