"""
Vibe Persistence Layer

Provides SQLite-based persistence for the Vibe Orchestrator.
Single source of truth - no in-memory caches needed.
"""

from vibe.persistence.models import (
    # Enums
    SessionStatus,
    TaskStatus,
    AttemptResult,
    MessageRole,
    MessageType,
    ChangeType,
    ContextCategory,
    Priority,
    # Core entities
    Project,
    Session,
    Message,
    Task,
    TaskStatusTransition,
    TaskAttempt,
    FileChange,
    Review,
    # Debug entities
    DebugSession,
    DebugIteration,
    # Support entities
    ContextItem,
    Convention,
    Checkpoint,
    ToolUsage,
    Request,
)

from vibe.persistence.repository import VibeRepository

__all__ = [
    # Enums
    "SessionStatus",
    "TaskStatus",
    "AttemptResult",
    "MessageRole",
    "MessageType",
    "ChangeType",
    "ContextCategory",
    "Priority",
    # Core entities
    "Project",
    "Session",
    "Message",
    "Task",
    "TaskStatusTransition",
    "TaskAttempt",
    "FileChange",
    "Review",
    # Debug entities
    "DebugSession",
    "DebugIteration",
    # Support entities
    "ContextItem",
    "Convention",
    "Checkpoint",
    "ToolUsage",
    "Request",
    # Repository
    "VibeRepository",
]
