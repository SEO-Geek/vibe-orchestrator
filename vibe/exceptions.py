"""
Vibe Orchestrator - Exception Hierarchy

Modeled after Athena's exception patterns for consistent error handling.
All Vibe-specific exceptions inherit from VibeError.
"""

from typing import Any


class VibeError(Exception):
    """Base exception for all Vibe-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Startup and Configuration Errors
class StartupError(VibeError):
    """Raised when startup validation fails."""

    pass


class ConfigError(VibeError):
    """Raised when configuration is invalid or missing."""

    pass


class ProjectNotFoundError(ConfigError):
    """Raised when a project is not found in the registry."""

    pass


# GLM (OpenRouter) Errors
class GLMError(VibeError):
    """Base exception for GLM/OpenRouter API errors."""

    pass


class GLMConnectionError(GLMError):
    """Raised when connection to OpenRouter fails."""

    pass


class GLMResponseError(GLMError):
    """Raised when GLM returns an unexpected response."""

    pass


class GLMRateLimitError(GLMError):
    """Raised when OpenRouter rate limit is hit."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


# Claude Errors
class ClaudeError(VibeError):
    """Base exception for Claude CLI errors."""

    pass


class ClaudeNotFoundError(ClaudeError):
    """Raised when Claude CLI is not installed."""

    pass


class ClaudeTimeoutError(ClaudeError):
    """Raised when Claude task times out."""

    def __init__(self, message: str, timeout_seconds: int):
        super().__init__(message, {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class ClaudeExecutionError(ClaudeError):
    """Raised when Claude task fails during execution."""

    def __init__(self, message: str, exit_code: int, stderr: str | None = None):
        super().__init__(message, {"exit_code": exit_code, "stderr": stderr})
        self.exit_code = exit_code
        self.stderr = stderr


class ClaudeCircuitOpenError(ClaudeError):
    """Raised when circuit breaker is open due to repeated failures."""

    def __init__(self, message: str, failures: int, reset_time: float):
        super().__init__(message, {"failures": failures, "reset_time": reset_time})
        self.failures = failures
        self.reset_time = reset_time


# Memory Errors (prefixed to avoid shadowing built-in MemoryError)
class VibeMemoryError(VibeError):
    """Base exception for memory-keeper errors."""

    pass


class MemoryConnectionError(VibeMemoryError):
    """Raised when SQLite database connection fails."""

    pass


class MemoryNotFoundError(VibeMemoryError):
    """Raised when requested memory item is not found."""

    pass


# Review Gate Errors
class ReviewError(VibeError):
    """Base exception for review gate errors."""

    pass


class ReviewRejectedError(ReviewError):
    """Raised when GLM rejects Claude's work."""

    def __init__(self, message: str, issues: list[str], feedback: str):
        super().__init__(message, {"issues": issues, "feedback": feedback})
        self.issues = issues
        self.feedback = feedback


# Task Errors
class TaskError(VibeError):
    """Base exception for task-related errors."""

    pass


class TaskParseError(TaskError):
    """Raised when GLM's task decomposition cannot be parsed."""

    pass


class TaskQueueFullError(TaskError):
    """Raised when task queue is at capacity."""

    pass


# Integration Errors
class ResearchError(VibeError):
    """Base exception for Perplexity research errors."""

    pass


class GitHubError(VibeError):
    """Base exception for GitHub CLI errors."""

    pass
