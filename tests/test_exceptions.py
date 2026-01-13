"""Tests for exception hierarchy."""

import pytest

from vibe.exceptions import (
    ClaudeCircuitOpenError,
    ClaudeExecutionError,
    ClaudeTimeoutError,
    ConfigError,
    GLMConnectionError,
    GLMError,
    GLMRateLimitError,
    GLMResponseError,
    MemoryConnectionError,
    MemoryNotFoundError,
    ProjectNotFoundError,
    ReviewRejectedError,
    ReviewTimeoutError,
    StartupError,
    StateTransitionError,
    TaskParseError,
    TaskQueueFullError,
    VibeError,
    VibeMemoryError,
)


class TestVibeError:
    """Tests for base VibeError."""

    def test_basic_error(self):
        """Test basic error creation."""
        err = VibeError("Something went wrong")
        assert err.message == "Something went wrong"
        assert err.details == {}
        assert str(err) == "Something went wrong"

    def test_error_with_details(self):
        """Test error with details dict."""
        err = VibeError("Error occurred", {"code": 500, "reason": "internal"})
        assert err.details == {"code": 500, "reason": "internal"}
        assert "code" in str(err)
        assert "500" in str(err)


class TestStartupAndConfigErrors:
    """Tests for startup and config errors."""

    def test_startup_error(self):
        """Test StartupError inherits from VibeError."""
        err = StartupError("Startup failed")
        assert isinstance(err, VibeError)

    def test_config_error(self):
        """Test ConfigError inherits from VibeError."""
        err = ConfigError("Bad config")
        assert isinstance(err, VibeError)

    def test_project_not_found_error(self):
        """Test ProjectNotFoundError inherits from ConfigError."""
        err = ProjectNotFoundError("Project not found")
        assert isinstance(err, ConfigError)
        assert isinstance(err, VibeError)


class TestGLMErrors:
    """Tests for GLM/OpenRouter errors."""

    def test_glm_error(self):
        """Test base GLMError."""
        err = GLMError("GLM failed")
        assert isinstance(err, VibeError)

    def test_glm_connection_error(self):
        """Test GLMConnectionError."""
        err = GLMConnectionError("Connection refused")
        assert isinstance(err, GLMError)

    def test_glm_response_error(self):
        """Test GLMResponseError."""
        err = GLMResponseError("Unexpected response")
        assert isinstance(err, GLMError)

    def test_glm_rate_limit_error(self):
        """Test GLMRateLimitError with retry_after."""
        err = GLMRateLimitError("Rate limited", retry_after=30)
        assert isinstance(err, GLMError)
        assert err.retry_after == 30
        assert err.details["retry_after"] == 30


class TestClaudeErrors:
    """Tests for Claude CLI errors."""

    def test_claude_timeout_error(self):
        """Test ClaudeTimeoutError with checkpoint info."""
        err = ClaudeTimeoutError(
            "Task timed out",
            timeout_seconds=300,
            checkpoint_summary="Completed 3/5 steps",
            files_modified=["file1.py", "file2.py"],
            tool_calls_count=15,
        )
        assert err.timeout_seconds == 300
        assert err.checkpoint_summary == "Completed 3/5 steps"
        assert err.files_modified == ["file1.py", "file2.py"]
        assert err.tool_calls_count == 15
        assert err.details["checkpoint_summary"] == "Completed 3/5 steps"

    def test_claude_timeout_error_defaults(self):
        """Test ClaudeTimeoutError default values."""
        err = ClaudeTimeoutError("Timeout", timeout_seconds=60)
        assert err.checkpoint_summary is None
        assert err.files_modified == []
        assert err.tool_calls_count == 0

    def test_claude_execution_error(self):
        """Test ClaudeExecutionError."""
        err = ClaudeExecutionError(
            "Claude failed",
            exit_code=1,
            stderr="Error: something bad",
        )
        assert err.exit_code == 1
        assert err.stderr == "Error: something bad"
        assert err.details["exit_code"] == 1

    def test_claude_circuit_open_error(self):
        """Test ClaudeCircuitOpenError."""
        err = ClaudeCircuitOpenError(
            "Circuit open",
            failures=3,
            reset_time=60.0,
        )
        assert err.failures == 3
        assert err.reset_time == 60.0


class TestMemoryErrors:
    """Tests for memory errors."""

    def test_vibe_memory_error(self):
        """Test base VibeMemoryError."""
        err = VibeMemoryError("Memory error")
        assert isinstance(err, VibeError)

    def test_memory_connection_error(self):
        """Test MemoryConnectionError."""
        err = MemoryConnectionError("SQLite connection failed")
        assert isinstance(err, VibeMemoryError)

    def test_memory_not_found_error(self):
        """Test MemoryNotFoundError."""
        err = MemoryNotFoundError("Item not found")
        assert isinstance(err, VibeMemoryError)


class TestReviewErrors:
    """Tests for review gate errors."""

    def test_review_rejected_error(self):
        """Test ReviewRejectedError with issues and feedback."""
        err = ReviewRejectedError(
            "Review rejected",
            issues=["Missing tests", "No documentation"],
            feedback="Please add unit tests",
        )
        assert err.issues == ["Missing tests", "No documentation"]
        assert err.feedback == "Please add unit tests"
        assert err.details["issues"] == ["Missing tests", "No documentation"]

    def test_review_timeout_error(self):
        """Test ReviewTimeoutError."""
        err = ReviewTimeoutError("Review timed out", timeout_seconds=30.0)
        assert err.timeout_seconds == 30.0


class TestTaskErrors:
    """Tests for task errors."""

    def test_task_parse_error(self):
        """Test TaskParseError."""
        err = TaskParseError("Could not parse task")
        assert isinstance(err, VibeError)

    def test_task_queue_full_error(self):
        """Test TaskQueueFullError."""
        err = TaskQueueFullError("Queue is full")
        assert isinstance(err, VibeError)


class TestStateTransitionError:
    """Tests for StateTransitionError."""

    def test_state_transition_error(self):
        """Test StateTransitionError stores state info."""
        err = StateTransitionError(
            "Invalid transition",
            from_state="IDLE",
            to_state="REVIEWING",
        )
        assert err.from_state == "IDLE"
        assert err.to_state == "REVIEWING"
        assert err.details["from_state"] == "IDLE"
        assert err.details["to_state"] == "REVIEWING"


class TestExceptionHierarchy:
    """Tests for exception hierarchy relationships."""

    def test_all_inherit_from_vibe_error(self):
        """All custom exceptions should inherit from VibeError."""
        exceptions = [
            StartupError("test"),
            ConfigError("test"),
            ProjectNotFoundError("test"),
            GLMError("test"),
            GLMConnectionError("test"),
            GLMResponseError("test"),
            GLMRateLimitError("test"),
            VibeMemoryError("test"),
            MemoryConnectionError("test"),
            MemoryNotFoundError("test"),
            TaskParseError("test"),
            TaskQueueFullError("test"),
            StateTransitionError("test", "A", "B"),
        ]

        for exc in exceptions:
            assert isinstance(exc, VibeError)

    def test_exceptions_are_catchable_by_parent(self):
        """Child exceptions should be catchable by parent type."""
        # GLM hierarchy
        with pytest.raises(GLMError):
            raise GLMConnectionError("test")

        # Memory hierarchy
        with pytest.raises(VibeMemoryError):
            raise MemoryNotFoundError("test")

        # Config hierarchy
        with pytest.raises(ConfigError):
            raise ProjectNotFoundError("test")
