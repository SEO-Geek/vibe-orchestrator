"""
Vibe Orchestrator Logging System.

Provides structured JSONL logging for:
- GLM API interactions (prompts, responses, tokens, timing)
- Claude CLI executions (prompts, outputs, tool calls, timing)
- Session lifecycle events (state transitions, requests)

Usage:
    from vibe.logging import glm_logger, claude_logger, session_logger
    from vibe.logging import GLMLogEntry, ClaudeLogEntry, SessionLogEntry

    # Log GLM interaction
    entry = GLMLogEntry(
        timestamp=now_iso(),
        request_id=str(uuid.uuid4()),
        session_id="sess-123",
        method="decompose_task",
        ...
    )
    glm_logger.info(entry.to_json())

Logs are written to ~/.vibe/logs/:
    - glm.jsonl: GLM API interactions
    - claude.jsonl: Claude CLI executions
    - session.jsonl: Session lifecycle events
"""

import threading
from typing import Any

from .config import LogConfig, get_config, set_config
from .entries import (
    ClaudeLogEntry,
    GeminiLogEntry,
    GLMLogEntry,
    SessionLogEntry,
    now_iso,
)
from .handlers import create_jsonl_logger

# Thread-local storage for session context
_context = threading.local()


def set_session_id(session_id: str) -> None:
    """Set the current session ID for log correlation."""
    _context.session_id = session_id


def get_session_id() -> str:
    """Get the current session ID, or 'unknown' if not set."""
    return getattr(_context, "session_id", "unknown")


def set_project_name(project_name: str) -> None:
    """Set the current project name for log context."""
    _context.project_name = project_name


def get_project_name() -> str:
    """Get the current project name, or empty string if not set."""
    return getattr(_context, "project_name", "")


# Lazy-initialized loggers to avoid creating files before needed
_glm_logger: Any = None
_gemini_logger: Any = None
_claude_logger: Any = None
_session_logger: Any = None
_init_lock = threading.Lock()


def _ensure_loggers() -> None:
    """Initialize loggers on first use."""
    global _glm_logger, _gemini_logger, _claude_logger, _session_logger

    if _glm_logger is not None:
        return

    with _init_lock:
        # Double-check after acquiring lock
        if _glm_logger is not None:
            return

        config = get_config()

        _gemini_logger = create_jsonl_logger(
            "vibe.gemini",
            config.gemini_log_path,
            level=config.gemini_level,
            max_bytes=config.max_file_size_bytes,
            backup_count=config.backup_count,
        )

        _glm_logger = create_jsonl_logger(
            "vibe.glm",
            config.glm_log_path,
            level=config.glm_level,
            max_bytes=config.max_file_size_bytes,
            backup_count=config.backup_count,
        )

        _claude_logger = create_jsonl_logger(
            "vibe.claude",
            config.claude_log_path,
            level=config.claude_level,
            max_bytes=config.max_file_size_bytes,
            backup_count=config.backup_count,
        )

        _session_logger = create_jsonl_logger(
            "vibe.session",
            config.session_log_path,
            level=config.session_level,
            max_bytes=config.max_file_size_bytes,
            backup_count=config.backup_count,
        )


class _LazyLogger:
    """Lazy wrapper that initializes the actual logger on first use."""

    def __init__(self, name: str):
        self._name = name

    def _get_logger(self) -> Any:
        _ensure_loggers()
        if self._name == "gemini":
            return _gemini_logger
        elif self._name == "glm":
            return _glm_logger
        elif self._name == "claude":
            return _claude_logger
        else:
            return _session_logger

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_logger().debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_logger().info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_logger().warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_logger().error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_logger().exception(msg, *args, **kwargs)


# Public logger instances
gemini_logger = _LazyLogger("gemini")
glm_logger = _LazyLogger("glm")
claude_logger = _LazyLogger("claude")
session_logger = _LazyLogger("session")


__all__ = [
    # Loggers
    "gemini_logger",
    "glm_logger",
    "claude_logger",
    "session_logger",
    # Log entries
    "GeminiLogEntry",
    "GLMLogEntry",
    "ClaudeLogEntry",
    "SessionLogEntry",
    # Utilities
    "now_iso",
    "get_session_id",
    "set_session_id",
    "get_project_name",
    "set_project_name",
    # Config
    "LogConfig",
    "get_config",
    "set_config",
]
