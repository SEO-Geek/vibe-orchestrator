"""Tests for reviewer module - GLM code review gate."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe.orchestrator.reviewer import (
    MAX_REVIEW_ATTEMPTS,
    MAX_TRACKED_TASKS,
    ReviewResult,
    Reviewer,
)
from vibe.state import Task


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_review_result_defaults(self):
        """Test ReviewResult default values."""
        result = ReviewResult(approved=True)
        assert result.approved is True
        assert result.issues == []
        assert result.feedback == ""
        assert result.task_id == ""
        assert result.attempt == 1
        assert isinstance(result.reviewed_at, datetime)

    def test_review_result_full(self):
        """Test ReviewResult with all fields."""
        result = ReviewResult(
            approved=False,
            issues=["Missing tests", "No docs"],
            feedback="Please fix",
            task_id="task-1",
            attempt=2,
        )
        assert not result.approved
        assert len(result.issues) == 2
        assert result.attempt == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ReviewResult(
            approved=True,
            issues=["minor issue"],
            feedback="LGTM",
            task_id="task-1",
            attempt=1,
        )
        data = result.to_dict()

        assert data["approved"] is True
        assert data["issues"] == ["minor issue"]
        assert data["feedback"] == "LGTM"
        assert data["task_id"] == "task-1"
        assert "reviewed_at" in data


class TestReviewer:
    """Tests for Reviewer class."""

    @pytest.fixture
    def mock_glm_client(self):
        """Create a mock GLM client."""
        client = MagicMock()
        client.review_changes = AsyncMock(return_value={
            "approved": True,
            "issues": [],
            "feedback": "Looks good!",
        })
        return client

    @pytest.fixture
    def reviewer(self, mock_glm_client):
        """Create a Reviewer instance with mock GLM client."""
        return Reviewer(
            glm_client=mock_glm_client,
            project_path="/test/project",
        )

    @pytest.fixture
    def task(self):
        """Create a test task."""
        return Task(id="task-1", description="Fix the bug")

    @pytest.fixture
    def mock_task_result(self):
        """Create a mock TaskResult."""
        result = MagicMock()
        result.result = "Fixed the issue"
        result.file_changes = ["file1.py"]
        return result

    def test_initial_state(self, reviewer):
        """Reviewer starts with empty tracking dicts."""
        assert reviewer._attempt_counts == {}
        assert reviewer._last_reviews == {}

    def test_get_attempt_count_empty(self, reviewer):
        """Get attempt count returns 0 for unknown task."""
        assert reviewer.get_attempt_count("unknown-task") == 0

    def test_should_retry_under_limit(self, reviewer):
        """should_retry returns True when under max attempts."""
        reviewer._attempt_counts["task-1"] = 1
        assert reviewer.should_retry("task-1")

    def test_should_retry_at_limit(self, reviewer):
        """should_retry returns False when at max attempts."""
        reviewer._attempt_counts["task-1"] = MAX_REVIEW_ATTEMPTS
        assert not reviewer.should_retry("task-1")

    def test_reset_attempts(self, reviewer):
        """reset_attempts clears task tracking."""
        reviewer._attempt_counts["task-1"] = 2
        reviewer._last_reviews["task-1"] = ReviewResult(approved=False)

        reviewer.reset_attempts("task-1")

        assert "task-1" not in reviewer._attempt_counts
        assert "task-1" not in reviewer._last_reviews

    def test_reset_all(self, reviewer):
        """reset_all clears all tracking."""
        reviewer._attempt_counts = {"t1": 1, "t2": 2}
        reviewer._last_reviews = {"t1": MagicMock(), "t2": MagicMock()}

        reviewer.reset_all()

        assert reviewer._attempt_counts == {}
        assert reviewer._last_reviews == {}

    def test_build_retry_context_empty(self, reviewer):
        """build_retry_context returns empty for unknown task."""
        assert reviewer.build_retry_context("unknown") == ""

    def test_build_retry_context_with_review(self, reviewer):
        """build_retry_context formats previous review feedback."""
        reviewer._last_reviews["task-1"] = ReviewResult(
            approved=False,
            issues=["Missing tests", "No documentation"],
            feedback="Please add unit tests for the new function.",
            task_id="task-1",
            attempt=1,
        )

        context = reviewer.build_retry_context("task-1")

        assert "PREVIOUS REVIEW FEEDBACK" in context
        assert "attempt (#1)" in context
        assert "Missing tests" in context
        assert "No documentation" in context
        assert "Please add unit tests" in context

    def test_get_last_review(self, reviewer):
        """get_last_review returns stored review."""
        result = ReviewResult(approved=True, task_id="task-1")
        reviewer._last_reviews["task-1"] = result

        assert reviewer.get_last_review("task-1") == result
        assert reviewer.get_last_review("unknown") is None

    def test_cleanup_completed_task(self, reviewer):
        """cleanup_completed_task removes task tracking."""
        reviewer._attempt_counts["task-1"] = 2
        reviewer._last_reviews["task-1"] = ReviewResult(approved=True)

        reviewer.cleanup_completed_task("task-1")

        assert "task-1" not in reviewer._attempt_counts
        assert "task-1" not in reviewer._last_reviews

    def test_get_stats(self, reviewer):
        """get_stats returns correct statistics."""
        reviewer._attempt_counts = {"t1": 2, "t2": 1}
        reviewer._last_reviews = {
            "t1": ReviewResult(approved=True),
            "t2": ReviewResult(approved=False),
        }

        stats = reviewer.get_stats()

        assert stats["active_tasks"] == 2
        assert stats["total_reviews"] == 3  # 2 + 1
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["max_attempts"] == MAX_REVIEW_ATTEMPTS


class TestReviewerMemoryBounds:
    """Tests for reviewer memory bounds enforcement."""

    @pytest.fixture
    def reviewer(self):
        """Create a reviewer with mock GLM client."""
        client = MagicMock()
        return Reviewer(
            glm_client=client,
            project_path="/test",
        )

    def test_enforce_max_size_under_limit(self, reviewer):
        """No eviction when under limit."""
        for i in range(50):
            reviewer._last_reviews[f"task-{i}"] = ReviewResult(approved=True)
            reviewer._attempt_counts[f"task-{i}"] = 1

        reviewer._enforce_max_size()

        assert len(reviewer._last_reviews) == 50

    def test_enforce_max_size_at_limit(self, reviewer):
        """No eviction when exactly at limit."""
        for i in range(MAX_TRACKED_TASKS):
            reviewer._last_reviews[f"task-{i}"] = ReviewResult(approved=True)
            reviewer._attempt_counts[f"task-{i}"] = 1

        reviewer._enforce_max_size()

        assert len(reviewer._last_reviews) == MAX_TRACKED_TASKS

    def test_enforce_max_size_over_limit(self, reviewer):
        """Evicts oldest when over limit."""
        # Add tasks with staggered timestamps
        base_time = datetime.now()
        for i in range(MAX_TRACKED_TASKS + 10):
            result = ReviewResult(
                approved=True,
                reviewed_at=base_time + timedelta(seconds=i),
            )
            reviewer._last_reviews[f"task-{i}"] = result
            reviewer._attempt_counts[f"task-{i}"] = 1

        reviewer._enforce_max_size()

        # Should have evicted oldest 10
        assert len(reviewer._last_reviews) == MAX_TRACKED_TASKS

        # Oldest tasks should be evicted
        for i in range(10):
            assert f"task-{i}" not in reviewer._last_reviews

        # Newest tasks should remain
        for i in range(MAX_TRACKED_TASKS + 10 - MAX_TRACKED_TASKS, MAX_TRACKED_TASKS + 10):
            assert f"task-{i}" in reviewer._last_reviews

    def test_cleanup_stale_tasks(self, reviewer):
        """cleanup_stale_tasks removes old tasks."""
        now = datetime.now()

        # Fresh task
        reviewer._last_reviews["fresh"] = ReviewResult(
            approved=True,
            reviewed_at=now,
        )
        reviewer._attempt_counts["fresh"] = 1

        # Stale task (over 1 hour old)
        reviewer._last_reviews["stale"] = ReviewResult(
            approved=True,
            reviewed_at=now - timedelta(hours=2),
        )
        reviewer._attempt_counts["stale"] = 1

        cleaned = reviewer.cleanup_stale_tasks(max_age_seconds=3600)

        assert cleaned == 1
        assert "fresh" in reviewer._last_reviews
        assert "stale" not in reviewer._last_reviews


class TestReviewerReview:
    """Tests for Reviewer.review() method."""

    @pytest.fixture
    def mock_glm_client(self):
        """Create a mock GLM client."""
        client = MagicMock()
        client.review_changes = AsyncMock(return_value={
            "approved": True,
            "issues": [],
            "feedback": "LGTM",
        })
        return client

    @pytest.fixture
    def reviewer(self, mock_glm_client):
        """Create a Reviewer with mock client."""
        return Reviewer(
            glm_client=mock_glm_client,
            project_path="/test",
        )

    @pytest.fixture
    def task(self):
        """Create a test task."""
        return Task(id="task-1", description="Fix bug")

    @pytest.fixture
    def mock_result(self):
        """Create a mock TaskResult."""
        result = MagicMock()
        result.result = "Done"
        result.file_changes = []
        return result

    @pytest.mark.asyncio
    async def test_review_approved(self, reviewer, task, mock_result):
        """Test successful approval."""
        with patch("vibe.claude.executor.get_git_diff", return_value=("diff output", False)):
            result = await reviewer.review(task, mock_result)

        assert result.approved
        assert result.task_id == "task-1"
        assert result.attempt == 1

    @pytest.mark.asyncio
    async def test_review_rejected(self, reviewer, task, mock_result, mock_glm_client):
        """Test rejection with issues."""
        mock_glm_client.review_changes.return_value = {
            "approved": False,
            "issues": ["Missing tests"],
            "feedback": "Please add tests",
        }

        with patch("vibe.claude.executor.get_git_diff", return_value=("diff", False)):
            result = await reviewer.review(task, mock_result)

        assert not result.approved
        assert "Missing tests" in result.issues
        assert result.feedback == "Please add tests"

    @pytest.mark.asyncio
    async def test_review_increments_attempt(self, reviewer, task, mock_result):
        """Review increments attempt count."""
        with patch("vibe.claude.executor.get_git_diff", return_value=("diff", False)):
            await reviewer.review(task, mock_result)
            result = await reviewer.review(task, mock_result)

        assert result.attempt == 2
        assert reviewer.get_attempt_count("task-1") == 2

    @pytest.mark.asyncio
    async def test_review_stores_result(self, reviewer, task, mock_result):
        """Review stores result for retry context."""
        with patch("vibe.claude.executor.get_git_diff", return_value=("diff", False)):
            await reviewer.review(task, mock_result)

        stored = reviewer.get_last_review("task-1")
        assert stored is not None
        assert stored.approved

    @pytest.mark.asyncio
    async def test_review_with_truncated_diff(self, reviewer, task, mock_result, mock_glm_client):
        """Test that truncated diff warning is included."""
        with patch("vibe.claude.executor.get_git_diff", return_value=("truncated diff", True)):
            await reviewer.review(task, mock_result)

        # Check that GLM was called with truncation warning
        call_args = mock_glm_client.review_changes.call_args
        diff_arg = call_args.kwargs.get("changes_diff", call_args.args[1] if len(call_args.args) > 1 else "")
        assert "TRUNCATED" in diff_arg or "⚠️" in diff_arg

    @pytest.mark.asyncio
    async def test_review_with_provided_diff(self, reviewer, task, mock_result):
        """Test review with pre-provided diff."""
        with patch("vibe.claude.executor.get_git_diff") as mock_diff:
            await reviewer.review(task, mock_result, diff="custom diff content")

            # get_git_diff should not be called when diff is provided
            mock_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_review_glm_error(self, reviewer, task, mock_result, mock_glm_client):
        """Test handling of GLM errors."""
        mock_glm_client.review_changes.side_effect = Exception("GLM failed")

        with patch("vibe.claude.executor.get_git_diff", return_value=("diff", False)):
            with pytest.raises(Exception, match="GLM failed"):
                await reviewer.review(task, mock_result)

        # Should still have stored a failed review result
        stored = reviewer.get_last_review("task-1")
        assert stored is not None
        assert not stored.approved
        assert "Review failed" in stored.issues[0]
