"""
End-to-end integration tests for the Supervisor workflow.

Tests the full flow: user request → Gemini decomposition → Claude execution → GLM review.

Architecture:
  User → Gemini (brain/orchestrator) → Claude (worker)
                     ↓                      ↓
                  GLM (code review/verification)

Uses mocked Gemini, GLM, and Claude clients to avoid real API calls.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe.claude.executor import TaskResult
from vibe.config import Project
from vibe.exceptions import GeminiError
from vibe.orchestrator.supervisor import Supervisor, SupervisorCallbacks
from vibe.state import SessionState


@dataclass
class MockExecutorContext:
    """Mock context manager for ClaudeExecutor."""

    result: TaskResult

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, **kwargs):
        return self.result

    def load_checkpoint_from_disk(self, task_id: str):
        """Mock checkpoint loading - no checkpoints in tests."""
        return None

    def clear_checkpoint_from_disk(self, task_id: str):
        """Mock checkpoint clearing."""
        pass


def prepare_supervisor_for_request(supervisor: Supervisor) -> None:
    """Prepare supervisor state for processing a request."""
    # Transition from INITIALIZING -> IDLE -> AWAITING_INPUT
    supervisor.context.transition_to(SessionState.IDLE)
    supervisor.context.transition_to(SessionState.AWAITING_INPUT)


class TestSupervisorE2E:
    """End-to-end tests for Supervisor workflow."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a temporary project directory."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        # Create minimal project structure
        (project_dir / "README.md").write_text("# Test Project")
        return Project(name="test_project", path=str(project_dir))

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mocked Gemini client (brain/orchestrator)."""
        client = MagicMock()

        # Mock check_clarification - no clarification needed
        async def mock_check_clarification(*args, **kwargs):
            return {"needs_clarification": False}

        # Mock decompose_task - return a simple task
        async def mock_decompose(*args, **kwargs):
            return [
                {
                    "id": "task-1",
                    "description": "Create a hello.py file",
                    "files": ["hello.py"],
                    "constraints": ["Must print 'Hello, World!'"],
                }
            ]

        client.check_clarification = AsyncMock(side_effect=mock_check_clarification)
        client.decompose_task = AsyncMock(side_effect=mock_decompose)
        client.get_usage_stats = MagicMock(
            return_value={
                "model": "mock-gemini",
                "request_count": 2,
                "total_tokens": 100,
            }
        )

        return client

    @pytest.fixture
    def mock_glm_client(self):
        """Create a mocked GLM client (code reviewer only)."""
        client = MagicMock()

        # Mock review_changes - approve the changes
        async def mock_review(*args, **kwargs):
            return {
                "approved": True,
                "issues": [],
                "feedback": "Changes look good.",
            }

        client.review_changes = AsyncMock(side_effect=mock_review)
        client.get_usage_stats = MagicMock(
            return_value={
                "model": "mock-glm",
                "request_count": 1,
                "total_tokens": 50,
            }
        )

        return client

    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks to track events."""
        # Use actual callback fields from SupervisorCallbacks
        return SupervisorCallbacks(
            on_status=MagicMock(),
            on_progress=MagicMock(),
            on_task_start=MagicMock(),
            on_task_complete=MagicMock(),
            on_review_result=MagicMock(),
            on_error=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_simple_task_workflow(self, project, mock_gemini_client, mock_glm_client, mock_callbacks):
        """Test a simple task workflow from request to completion."""
        # Create supervisor with mocked clients
        supervisor = Supervisor(
            gemini_client=mock_gemini_client,  # Brain/orchestrator
            glm_client=mock_glm_client,  # Code reviewer only
            project=project,
            callbacks=mock_callbacks,
            use_circuit_breaker=False,  # Disable for testing
        )
        prepare_supervisor_for_request(supervisor)

        # Mock the ClaudeExecutor to return a successful result
        mock_result = TaskResult(
            success=True,
            result="Created hello.py with print statement",
            file_changes=["hello.py"],
            duration_ms=1000,
            tool_calls=[],
            error=None,
        )

        mock_executor = MockExecutorContext(result=mock_result)

        with patch("vibe.claude.executor.ClaudeExecutor", return_value=mock_executor):
            # Run the workflow
            result = await supervisor.process_user_request("Create a hello world script")

        # Verify the workflow completed successfully
        assert result.success is True
        assert result.tasks_completed == 1
        assert result.tasks_failed == 0

        # Verify Gemini (brain) was called for decomposition
        mock_gemini_client.decompose_task.assert_called_once()
        # Verify GLM (reviewer) was called for code review
        mock_glm_client.review_changes.assert_called_once()

        # Verify callbacks were triggered
        mock_callbacks.on_status.assert_called()
        mock_callbacks.on_task_start.assert_called()
        mock_callbacks.on_task_complete.assert_called()

    @pytest.mark.asyncio
    async def test_investigation_task_skips_clarification(
        self, project, mock_gemini_client, mock_glm_client, mock_callbacks
    ):
        """Test that investigation tasks skip the clarification step."""
        supervisor = Supervisor(
            gemini_client=mock_gemini_client,
            glm_client=mock_glm_client,
            project=project,
            callbacks=mock_callbacks,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        # Mock decompose to return an investigation task
        async def mock_decompose(*args, **kwargs):
            return [
                {
                    "id": "task-1",
                    "description": "Search for all TODO comments",
                    "files": [],
                    "constraints": [],
                }
            ]

        mock_gemini_client.decompose_task = AsyncMock(side_effect=mock_decompose)

        # Mock executor
        mock_result = TaskResult(
            success=True,
            result="Found 5 TODO comments",
            file_changes=[],  # No changes - investigation only
            duration_ms=500,
            tool_calls=[],
            error=None,
        )
        mock_executor = MockExecutorContext(result=mock_result)

        with patch("vibe.claude.executor.ClaudeExecutor", return_value=mock_executor):
            # Run investigation query (starts with "What")
            result = await supervisor.process_user_request("What TODO comments exist in the codebase?")

        # Verify investigation detection worked
        assert result.success is True
        # Clarification should not have been called (investigation task)
        mock_gemini_client.check_clarification.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_change_task_skips_review(self, project, mock_gemini_client, mock_glm_client, mock_callbacks):
        """Test that tasks with no file changes skip the review step."""
        supervisor = Supervisor(
            gemini_client=mock_gemini_client,
            glm_client=mock_glm_client,
            project=project,
            callbacks=mock_callbacks,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        # Mock executor to return no file changes
        mock_result = TaskResult(
            success=True,
            result="Analyzed the codebase structure",
            file_changes=[],  # No file changes
            duration_ms=300,
            tool_calls=[],
            error=None,
        )
        mock_executor = MockExecutorContext(result=mock_result)

        with patch("vibe.claude.executor.ClaudeExecutor", return_value=mock_executor):
            result = await supervisor.process_user_request("Analyze the project structure")

        # Verify task completed
        assert result.success is True
        # Review should not have been called for no-change tasks
        mock_glm_client.review_changes.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_failure_handling(self, project, mock_gemini_client, mock_glm_client, mock_callbacks):
        """Test that task failures are handled gracefully."""
        supervisor = Supervisor(
            gemini_client=mock_gemini_client,
            glm_client=mock_glm_client,
            project=project,
            callbacks=mock_callbacks,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        # Mock executor to fail
        mock_result = TaskResult(
            success=False,
            result=None,
            file_changes=[],
            duration_ms=100,
            tool_calls=[],
            error="Execution timed out",
        )
        mock_executor = MockExecutorContext(result=mock_result)

        with patch("vibe.claude.executor.ClaudeExecutor", return_value=mock_executor):
            result = await supervisor.process_user_request("Do something that fails")

        # Verify failure was captured
        assert result.success is False
        assert result.tasks_failed >= 1
        mock_callbacks.on_error.assert_called()

    @pytest.mark.asyncio
    async def test_glm_rejection_triggers_retry(self, project, mock_gemini_client, mock_glm_client, mock_callbacks):
        """Test that GLM rejections trigger retries with feedback."""
        supervisor = Supervisor(
            gemini_client=mock_gemini_client,
            glm_client=mock_glm_client,
            project=project,
            callbacks=mock_callbacks,
            max_retries=2,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        # Track review call count
        review_call_count = 0

        # Mock review to reject first, then approve
        async def mock_review(*args, **kwargs):
            nonlocal review_call_count
            review_call_count += 1
            if review_call_count == 1:
                return {
                    "approved": False,
                    "issues": ["Missing error handling"],
                    "feedback": "Add try/except block",
                }
            return {
                "approved": True,
                "issues": [],
                "feedback": "Looks good now",
            }

        mock_glm_client.review_changes = AsyncMock(side_effect=mock_review)

        # Mock executor
        mock_result = TaskResult(
            success=True,
            result="Created the file",
            file_changes=["test.py"],
            duration_ms=500,
            tool_calls=[],
            error=None,
        )
        mock_executor = MockExecutorContext(result=mock_result)

        with patch("vibe.claude.executor.ClaudeExecutor", return_value=mock_executor):
            result = await supervisor.process_user_request("Create a test file")

        # Verify retry happened
        assert review_call_count == 2
        # Final result should be success after retry
        assert result.success is True

    @pytest.mark.asyncio
    async def test_multi_task_workflow(self, project, mock_gemini_client, mock_glm_client, mock_callbacks):
        """Test workflow with multiple tasks from Gemini decomposition."""
        supervisor = Supervisor(
            gemini_client=mock_gemini_client,
            glm_client=mock_glm_client,
            project=project,
            callbacks=mock_callbacks,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        # Mock decompose to return multiple tasks (using Gemini)
        async def mock_decompose(*args, **kwargs):
            return [
                {
                    "id": "task-1",
                    "description": "Create config module",
                    "files": ["config.py"],
                    "constraints": [],
                },
                {
                    "id": "task-2",
                    "description": "Create utils module",
                    "files": ["utils.py"],
                    "constraints": [],
                },
                {
                    "id": "task-3",
                    "description": "Create main module",
                    "files": ["main.py"],
                    "constraints": [],
                },
            ]

        mock_gemini_client.decompose_task = AsyncMock(side_effect=mock_decompose)

        # Track execution count
        call_count = 0

        # Create mock executor that tracks calls
        class CountingMockExecutor:
            def __init__(self, *args, **kwargs):
                pass  # Accept any arguments

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def execute(self, **kwargs):
                nonlocal call_count
                call_count += 1
                return TaskResult(
                    success=True,
                    result=f"Created module {call_count}",
                    file_changes=[f"file{call_count}.py"],
                    duration_ms=200,
                    tool_calls=[],
                    error=None,
                )

            def load_checkpoint_from_disk(self, task_id: str):
                return None

            def clear_checkpoint_from_disk(self, task_id: str):
                pass

        with patch("vibe.claude.executor.ClaudeExecutor", CountingMockExecutor):
            result = await supervisor.process_user_request("Set up a basic Python project structure")

        # Verify all tasks completed
        assert result.success is True
        assert result.tasks_completed == 3
        assert result.tasks_failed == 0

        # Verify callbacks were called for each task
        assert mock_callbacks.on_task_start.call_count == 3
        assert mock_callbacks.on_task_complete.call_count == 3


class TestSupervisorIntegration:
    """Integration tests that verify component interactions."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a temporary project."""
        project_dir = tmp_path / "integration_test"
        project_dir.mkdir()
        return Project(name="integration_test", path=str(project_dir))

    @pytest.mark.asyncio
    async def test_supervisor_with_memory(self, project):
        """Test Supervisor integration with memory system."""
        # Mock Gemini (brain/orchestrator)
        mock_gemini = MagicMock()
        mock_gemini.check_clarification = AsyncMock(return_value={"needs_clarification": False})
        mock_gemini.decompose_task = AsyncMock(
            return_value=[{"id": "t1", "description": "Test task", "files": [], "constraints": []}]
        )
        mock_gemini.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        # Mock GLM (code reviewer only)
        mock_glm = MagicMock()
        mock_glm.review_changes = AsyncMock(return_value={"approved": True, "issues": []})
        mock_glm.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        mock_memory = MagicMock()
        mock_memory.save = MagicMock()
        mock_memory.load_conventions = MagicMock(return_value=[])
        mock_memory.create_checkpoint_with_git = MagicMock()

        supervisor = Supervisor(
            gemini_client=mock_gemini,
            glm_client=mock_glm,
            project=project,
            memory=mock_memory,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        # Create mock executor
        mock_result = TaskResult(
            success=True,
            result="Done",
            file_changes=[],
            duration_ms=100,
            tool_calls=[],
            error=None,
        )
        mock_executor = MockExecutorContext(result=mock_result)

        with patch("vibe.claude.executor.ClaudeExecutor", return_value=mock_executor):
            result = await supervisor.process_user_request("Test task")

        # Memory should be used for conventions
        assert result.success is True
        mock_memory.load_conventions.assert_called()


class TestSupervisorEdgeCases:
    """Edge case tests for Supervisor robustness."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a temporary project."""
        project_dir = tmp_path / "edge_test"
        project_dir.mkdir()
        return Project(name="edge_test", path=str(project_dir))

    @pytest.mark.asyncio
    async def test_empty_task_list_from_gemini(self, project):
        """Test handling when Gemini returns no tasks."""
        mock_gemini = MagicMock()
        mock_gemini.check_clarification = AsyncMock(return_value={"needs_clarification": False})
        mock_gemini.decompose_task = AsyncMock(return_value=[])  # Empty list
        mock_gemini.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        mock_glm = MagicMock()
        mock_glm.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        supervisor = Supervisor(
            gemini_client=mock_gemini,
            glm_client=mock_glm,
            project=project,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        result = await supervisor.process_user_request("Do something")

        # Should fail gracefully with error
        assert result.success is False
        assert result.error is not None
        assert "no tasks" in result.error.lower()

    @pytest.mark.asyncio
    async def test_gemini_decomposition_error(self, project):
        """Test handling when Gemini decomposition fails."""
        mock_gemini = MagicMock()
        mock_gemini.check_clarification = AsyncMock(return_value={"needs_clarification": False})
        mock_gemini.decompose_task = AsyncMock(side_effect=GeminiError("API error"))
        mock_gemini.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        mock_glm = MagicMock()
        mock_glm.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        supervisor = Supervisor(
            gemini_client=mock_gemini,
            glm_client=mock_glm,
            project=project,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        result = await supervisor.process_user_request("Do something")

        # Should fail with error message
        assert result.success is False
        assert result.error is not None
        assert "decomposition failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_clarification_request(self, project):
        """Test handling when Gemini requests clarification."""
        mock_gemini = MagicMock()
        mock_gemini.check_clarification = AsyncMock(
            return_value={"needs_clarification": True, "question": "Do you want to use Python or JavaScript?"}
        )
        mock_gemini.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        mock_glm = MagicMock()
        mock_glm.get_usage_stats = MagicMock(return_value={"total_tokens": 0})

        supervisor = Supervisor(
            gemini_client=mock_gemini,
            glm_client=mock_glm,
            project=project,
            use_circuit_breaker=False,
        )
        prepare_supervisor_for_request(supervisor)

        result = await supervisor.process_user_request("Create a web app")

        # Should return success with clarification
        assert result.success is True
        assert result.clarification_asked is not None
        assert "Python or JavaScript" in result.clarification_asked
        # No tasks should have been executed
        assert result.tasks_completed == 0
