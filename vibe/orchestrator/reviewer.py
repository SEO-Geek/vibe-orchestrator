"""
GLM Reviewer - Code review gate

Has GLM review Claude's output before accepting changes.
Tracks review attempts per task and provides retry context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from vibe.glm.client import GLMClient
from vibe.state import Task

# TYPE_CHECKING guard prevents runtime circular import:
# reviewer.py -> executor.py -> reviewer.py
# At runtime, TaskResult is only used in type annotations (not instantiated here)
if TYPE_CHECKING:
    from vibe.claude.executor import TaskResult

logger = logging.getLogger(__name__)

# Maximum retry attempts before giving up
MAX_REVIEW_ATTEMPTS = 3

# Maximum number of tasks to track in memory (prevents unbounded growth)
# Uses LRU eviction - oldest tasks are removed when limit is exceeded
MAX_TRACKED_TASKS = 100


@dataclass
class ReviewResult:
    """Result from GLM code review."""

    approved: bool
    issues: list[str] = field(default_factory=list)
    feedback: str = ""
    task_id: str = ""
    attempt: int = 1
    reviewed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "approved": self.approved,
            "issues": self.issues,
            "feedback": self.feedback,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "reviewed_at": self.reviewed_at.isoformat(),
        }


class Reviewer:
    """
    Reviewer that uses GLM to evaluate Claude's work.

    Review criteria:
    - Task completion: Did Claude complete the requested task?
    - Code quality: Sustainable solution vs bush fix?
    - Documentation: Inline comments present?
    - Scope: Did Claude stay within task boundaries?

    Tracks attempts per task to enable retry logic with feedback.
    """

    def __init__(
        self,
        glm_client: GLMClient,
        project_path: str,
        max_attempts: int = MAX_REVIEW_ATTEMPTS,
    ):
        """
        Initialize reviewer.

        Args:
            glm_client: GLM client for API calls
            project_path: Path to project for git diff operations
            max_attempts: Maximum review attempts per task (default: 3)
        """
        self.glm_client = glm_client
        self.project_path = project_path
        self.max_attempts = max_attempts

        # Track attempts per task: task_id -> attempt count
        # Separate from _last_reviews because we need count even if review failed mid-way
        self._attempt_counts: dict[str, int] = {}

        # Store last review result per task for retry context
        # Enables Claude to receive specific feedback on what to fix in next attempt
        self._last_reviews: dict[str, ReviewResult] = {}

    async def review(
        self,
        task: Task,
        claude_result: TaskResult,
        diff: str | None = None,
    ) -> ReviewResult:
        """
        Review Claude's task output.

        Args:
            task: The task that was executed
            claude_result: Claude's execution result
            diff: Git diff of changes (if None, will be fetched)

        Returns:
            ReviewResult with approval status, issues, and feedback

        Raises:
            ReviewRejectedError: If review is rejected after max attempts
        """
        # Increment attempt count for this task
        attempt = self._increment_attempt(task.id)

        logger.info(f"Reviewing task '{task.id}' (attempt {attempt}/{self.max_attempts})")

        # Get git diff if not provided
        # Caller can pass pre-computed diff to avoid redundant git operations
        was_truncated = False
        if diff is None:
            # Deferred import avoids circular dependency at module load time
            from vibe.claude.executor import get_git_diff

            # Scope diff to only files Claude reported changing, avoiding noise from
            # unrelated uncommitted changes in the working directory
            files_to_diff = claude_result.file_changes if claude_result.file_changes else None
            diff, was_truncated = get_git_diff(self.project_path, files=files_to_diff)

        # Prepend truncation warning for GLM if diff was truncated
        # GLM needs to know it's reviewing partial data to make informed approval decisions
        review_diff = diff
        if was_truncated:
            truncation_warning = (
                "⚠️ IMPORTANT: This diff was TRUNCATED due to size limits. You are only "
                "reviewing a portion of the changes. Additional modifications may exist "
                "that are not shown here. Consider:\n"
                "1. Approving with caution if the visible changes look correct\n"
                "2. The Claude summary may reference changes not visible in this diff\n"
                "3. Files changed but not shown: check the Claude summary for completeness\n"
                "---\n\n"
            )
            review_diff = truncation_warning + diff
            logger.warning("Diff was truncated - GLM review will include warning notice")

        # Get Claude's summary from result
        claude_summary = claude_result.result or "No summary provided"

        # Call GLM to review the changes
        try:
            glm_result = await self.glm_client.review_changes(
                task_description=task.description,
                changes_diff=review_diff,
                claude_summary=claude_summary,
            )
        except Exception as e:
            logger.error(f"GLM review failed: {e}")
            # Store failed review state before re-raising so retry logic can still
            # access attempt count and provide appropriate context to caller
            review_result = ReviewResult(
                approved=False,
                issues=[f"Review failed: {str(e)}"],
                feedback="Unable to complete review due to GLM error. Please retry.",
                task_id=task.id,
                attempt=attempt,
            )
            self._last_reviews[task.id] = review_result
            raise  # Let caller decide how to handle GLM failures

        # Build ReviewResult from GLM response
        # Default to rejected (False) if GLM response is malformed - fail safe
        review_result = ReviewResult(
            approved=glm_result.get("approved", False),
            issues=glm_result.get("issues", []),
            feedback=glm_result.get("feedback", ""),
            task_id=task.id,
            attempt=attempt,
        )

        # Store result regardless of approval status - needed for:
        # 1. Building retry context if rejected
        # 2. Stats tracking
        # 3. LRU eviction ordering (by reviewed_at timestamp)
        self._last_reviews[task.id] = review_result

        if review_result.approved:
            logger.info(f"Task '{task.id}' APPROVED on attempt {attempt}")
        else:
            logger.warning(
                f"Task '{task.id}' REJECTED on attempt {attempt}: "
                f"{len(review_result.issues)} issues"
            )
            for issue in review_result.issues:
                logger.warning(f"  - {issue}")

        return review_result

    def should_retry(self, task_id: str) -> bool:
        """
        Check if a task should be retried based on attempt count.

        Args:
            task_id: The task identifier

        Returns:
            True if under max attempts and should retry, False otherwise
        """
        # Note: This checks current count, which is already incremented by review()
        # So if max_attempts=3 and current_attempts=3, we've used all attempts
        current_attempts = self._attempt_counts.get(task_id, 0)
        return current_attempts < self.max_attempts

    def build_retry_context(self, task_id: str) -> str:
        """
        Build context string for Claude's retry attempt.

        Formats the previous review feedback to help Claude
        address the identified issues.

        Args:
            task_id: The task identifier

        Returns:
            Formatted retry context string for Claude
        """
        last_review = self._last_reviews.get(task_id)

        if not last_review:
            return ""

        parts = [
            "PREVIOUS REVIEW FEEDBACK:",
            "========================",
            "",
            f"Your previous attempt (#{last_review.attempt}) was rejected.",
            "",
        ]

        if last_review.issues:
            parts.append("ISSUES TO ADDRESS:")
            for i, issue in enumerate(last_review.issues, 1):
                parts.append(f"{i}. {issue}")
            parts.append("")

        if last_review.feedback:
            parts.append("REVIEWER FEEDBACK:")
            parts.append(last_review.feedback)
            parts.append("")

        # Explicit instruction helps prevent Claude from making unrelated changes
        # while fixing issues, which could introduce new problems
        parts.extend(
            [
                "Please address ALL issues above in your next attempt.",
                "Focus on fixing the identified problems without introducing new ones.",
                "",
            ]
        )

        return "\n".join(parts)

    def get_attempt_count(self, task_id: str) -> int:
        """
        Get the current attempt count for a task.

        Args:
            task_id: The task identifier

        Returns:
            Number of attempts made (0 if never attempted)
        """
        return self._attempt_counts.get(task_id, 0)

    def reset_attempts(self, task_id: str) -> None:
        """
        Reset the attempt counter for a task.

        Called when a task is approved or explicitly abandoned.

        Args:
            task_id: The task identifier
        """
        # Use pop() with default to avoid KeyError if task was never tracked
        # (defensive - allows calling reset on any task without checking existence)
        self._attempt_counts.pop(task_id, None)
        self._last_reviews.pop(task_id, None)
        logger.debug(f"Reset attempt counter for task '{task_id}'")

    def reset_all(self) -> None:
        """Reset all attempt counters and review history."""
        self._attempt_counts.clear()
        self._last_reviews.clear()
        logger.debug("Reset all reviewer state")

    def _enforce_max_size(self) -> None:
        """
        Enforce maximum tracked tasks limit using LRU eviction.

        Removes the oldest tasks (by review timestamp) when the number
        of tracked tasks exceeds MAX_TRACKED_TASKS. This prevents
        unbounded memory growth in long-running sessions.
        """
        if len(self._last_reviews) <= MAX_TRACKED_TASKS:
            return

        # Sort tasks by review timestamp (oldest first)
        # O(n log n) but only runs when limit exceeded, which is rare
        sorted_tasks = sorted(self._last_reviews.items(), key=lambda x: x[1].reviewed_at)

        # Calculate how many to evict (keep newest MAX_TRACKED_TASKS)
        evict_count = len(sorted_tasks) - MAX_TRACKED_TASKS

        # Evict oldest tasks - these are likely completed/abandoned tasks
        # whose retry context is no longer needed
        for task_id, _ in sorted_tasks[:evict_count]:
            self._attempt_counts.pop(task_id, None)
            self._last_reviews.pop(task_id, None)

        logger.debug(f"Evicted {evict_count} old tasks from reviewer (LRU)")

    def _increment_attempt(self, task_id: str) -> int:
        """
        Increment and return the attempt count for a task.

        Also enforces memory bounds by evicting old tasks if limit exceeded.

        Args:
            task_id: The task identifier

        Returns:
            New attempt count after increment
        """
        current = self._attempt_counts.get(task_id, 0)
        self._attempt_counts[task_id] = current + 1

        # Enforce memory bounds after adding new task
        # Placed here (not in review()) to ensure bounds are checked on every new task
        self._enforce_max_size()

        return current + 1

    def get_last_review(self, task_id: str) -> ReviewResult | None:
        """
        Get the last review result for a task.

        Args:
            task_id: The task identifier

        Returns:
            Last ReviewResult or None if never reviewed
        """
        return self._last_reviews.get(task_id)

    def get_stats(self) -> dict[str, Any]:
        """
        Get reviewer statistics.

        Returns:
            Dictionary with review statistics
        """
        # total_reviews counts all attempts across all tasks (sum of attempt counts)
        # vs len(_last_reviews) which is unique tasks reviewed
        total_reviews = sum(self._attempt_counts.values())
        approved_count = sum(1 for r in self._last_reviews.values() if r.approved)
        rejected_count = len(self._last_reviews) - approved_count

        return {
            "active_tasks": len(self._attempt_counts),
            "total_reviews": total_reviews,
            "approved": approved_count,
            "rejected": rejected_count,
            "max_attempts": self.max_attempts,
        }

    def cleanup_completed_task(self, task_id: str) -> None:
        """
        Clean up tracking data for a completed task.

        Call this after a task is approved OR after max retries exhausted
        to prevent memory leak from unbounded dict growth.

        Args:
            task_id: The task identifier to clean up
        """
        self._attempt_counts.pop(task_id, None)
        self._last_reviews.pop(task_id, None)
        logger.debug(f"Cleaned up tracking data for task '{task_id}'")

    def cleanup_stale_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up tasks that have been idle for too long.

        Prevents memory leak from abandoned tasks that were never completed.
        Default cleanup age is 1 hour.

        Args:
            max_age_seconds: Maximum age in seconds before cleanup (default: 1 hour)

        Returns:
            Number of stale tasks cleaned up
        """
        now = datetime.now()
        stale_tasks = []

        # Find tasks with reviews older than max_age
        # This catches tasks where the orchestrator crashed or user abandoned mid-retry
        for task_id, review in self._last_reviews.items():
            age = (now - review.reviewed_at).total_seconds()
            if age > max_age_seconds:
                stale_tasks.append(task_id)

        # Clean up stale tasks
        for task_id in stale_tasks:
            self._attempt_counts.pop(task_id, None)
            self._last_reviews.pop(task_id, None)

        if stale_tasks:
            logger.info(f"Cleaned up {len(stale_tasks)} stale tasks from reviewer")

        return len(stale_tasks)
