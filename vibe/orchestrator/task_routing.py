"""
Task Routing - Intelligent task-type-aware configuration.

Provides per-task-type settings for:
- Timeout tiers (how long to wait for Claude)
- Review requirements (skip for investigation tasks)
- Retry strategies (different max retries per type)
- Test requirements (should tests run after?)

This replaces hardcoded values with smart, task-aware defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vibe.orchestrator.task_enforcer import TaskType, get_smart_detector


class TimeoutTier(Enum):
    """Timeout tiers for Claude execution."""

    SHORT = "short"  # 60s - quick tasks
    MEDIUM = "medium"  # 120s - standard tasks
    LONG = "long"  # 300s - complex tasks
    CODE = "code"  # 180s - code writing (existing tier)


# Timeout durations in seconds for each tier
TIMEOUT_DURATIONS: dict[TimeoutTier, int] = {
    TimeoutTier.SHORT: 60,
    TimeoutTier.MEDIUM: 120,
    TimeoutTier.LONG: 300,
    TimeoutTier.CODE: 180,
}


@dataclass
class TaskRoutingConfig:
    """
    Configuration for how a task type should be handled.

    This enables smart defaults based on task type, reducing
    wasted time on unnecessary steps (e.g., no review for research).
    """

    task_type: TaskType

    # Timeout configuration
    timeout_tier: TimeoutTier = TimeoutTier.MEDIUM
    timeout_seconds: int | None = None  # Override tier if set

    # Review configuration
    require_review: bool = True  # Should GLM review output?
    skip_review_if_no_changes: bool = True  # Auto-skip if no file changes

    # Retry configuration
    max_retries: int = 3
    retry_on_timeout: bool = True  # Retry if Claude times out?

    # Test configuration
    run_tests_after: bool = False  # Run project tests after task?
    test_on_failure_only: bool = False  # Only run tests if task failed?

    # Workflow configuration
    expand_to_phases: bool = True  # Use WorkflowEngine to expand?

    # Priority (higher = more important)
    priority: int = 5  # 1-10 scale

    def get_timeout(self) -> int:
        """Get effective timeout in seconds."""
        if self.timeout_seconds:
            return self.timeout_seconds
        return TIMEOUT_DURATIONS.get(self.timeout_tier, 120)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_type": self.task_type.value,
            "timeout_tier": self.timeout_tier.value,
            "timeout_seconds": self.get_timeout(),
            "require_review": self.require_review,
            "skip_review_if_no_changes": self.skip_review_if_no_changes,
            "max_retries": self.max_retries,
            "retry_on_timeout": self.retry_on_timeout,
            "run_tests_after": self.run_tests_after,
            "expand_to_phases": self.expand_to_phases,
            "priority": self.priority,
        }


# Default routing configurations per task type
DEFAULT_ROUTING_CONFIGS: dict[TaskType, TaskRoutingConfig] = {
    # Investigation tasks - no review needed, quick timeout
    TaskType.RESEARCH: TaskRoutingConfig(
        task_type=TaskType.RESEARCH,
        timeout_tier=TimeoutTier.MEDIUM,
        require_review=False,  # Research doesn't need code review
        max_retries=1,  # Don't retry research tasks
        expand_to_phases=False,  # Keep research simple
        priority=3,
    ),
    # Debug tasks - longer timeout, review required
    TaskType.DEBUG: TaskRoutingConfig(
        task_type=TaskType.DEBUG,
        timeout_tier=TimeoutTier.LONG,
        require_review=True,
        max_retries=3,
        run_tests_after=True,  # Verify fix with tests
        expand_to_phases=True,  # Use debug workflow phases
        priority=8,
    ),
    # Code writing - standard timeout, review required
    TaskType.CODE_WRITE: TaskRoutingConfig(
        task_type=TaskType.CODE_WRITE,
        timeout_tier=TimeoutTier.CODE,
        require_review=True,
        max_retries=3,
        run_tests_after=True,
        expand_to_phases=True,
        priority=7,
    ),
    # Code refactoring - longer timeout, strict review
    TaskType.CODE_REFACTOR: TaskRoutingConfig(
        task_type=TaskType.CODE_REFACTOR,
        timeout_tier=TimeoutTier.LONG,
        require_review=True,
        max_retries=2,  # Fewer retries - refactoring is delicate
        run_tests_after=True,  # Always verify refactoring
        expand_to_phases=True,
        priority=6,
    ),
    # UI testing - quick timeout, no code review
    TaskType.UI_TEST: TaskRoutingConfig(
        task_type=TaskType.UI_TEST,
        timeout_tier=TimeoutTier.MEDIUM,
        require_review=False,  # UI tests are self-verifying
        max_retries=2,
        expand_to_phases=False,
        priority=5,
    ),
    # API testing - quick timeout, no code review
    TaskType.API_TEST: TaskRoutingConfig(
        task_type=TaskType.API_TEST,
        timeout_tier=TimeoutTier.SHORT,
        require_review=False,  # API tests are self-verifying
        max_retries=2,
        expand_to_phases=False,
        priority=5,
    ),
    # Browser work - medium timeout
    TaskType.BROWSER_WORK: TaskRoutingConfig(
        task_type=TaskType.BROWSER_WORK,
        timeout_tier=TimeoutTier.MEDIUM,
        require_review=False,  # Browser work is exploratory
        max_retries=1,
        expand_to_phases=False,
        priority=4,
    ),
    # Database - longer timeout, careful review
    TaskType.DATABASE: TaskRoutingConfig(
        task_type=TaskType.DATABASE,
        timeout_tier=TimeoutTier.MEDIUM,
        require_review=True,  # DB changes are dangerous
        max_retries=1,  # Don't retry DB operations carelessly
        run_tests_after=True,
        expand_to_phases=False,
        priority=7,
    ),
    # Docker - longer timeout
    TaskType.DOCKER: TaskRoutingConfig(
        task_type=TaskType.DOCKER,
        timeout_tier=TimeoutTier.LONG,
        require_review=True,
        max_retries=2,
        expand_to_phases=False,
        priority=5,
    ),
    # General - default settings
    TaskType.GENERAL: TaskRoutingConfig(
        task_type=TaskType.GENERAL,
        timeout_tier=TimeoutTier.MEDIUM,
        require_review=True,
        max_retries=3,
        expand_to_phases=False,
        priority=5,
    ),
}


class TaskRouter:
    """
    Routes tasks to appropriate configurations based on detected type.

    Uses SmartTaskDetector for type detection, then applies
    task-type-specific settings for optimal execution.
    """

    def __init__(self, custom_configs: dict[TaskType, TaskRoutingConfig] | None = None):
        """
        Initialize router with optional custom configurations.

        Args:
            custom_configs: Override default configs for specific task types
        """
        self._configs = DEFAULT_ROUTING_CONFIGS.copy()
        if custom_configs:
            self._configs.update(custom_configs)

    def get_config(self, task_description: str) -> TaskRoutingConfig:
        """
        Get routing configuration for a task based on its description.

        Uses SmartTaskDetector to determine task type, then returns
        the appropriate routing configuration.

        Args:
            task_description: The task description to analyze

        Returns:
            TaskRoutingConfig for the detected task type
        """
        detector = get_smart_detector()
        detection = detector.detect(task_description)

        config = self._configs.get(
            detection.task_type,
            self._configs[TaskType.GENERAL],
        )

        return config

    def get_config_for_type(self, task_type: TaskType) -> TaskRoutingConfig:
        """
        Get routing configuration for a specific task type.

        Args:
            task_type: The TaskType enum value

        Returns:
            TaskRoutingConfig for the task type
        """
        return self._configs.get(task_type, self._configs[TaskType.GENERAL])

    def should_skip_review(self, task_description: str, has_file_changes: bool) -> bool:
        """
        Determine if GLM review should be skipped for a task.

        Args:
            task_description: The task description
            has_file_changes: Whether the task produced file changes

        Returns:
            True if review should be skipped
        """
        config = self.get_config(task_description)

        # Skip if review not required for this task type
        if not config.require_review:
            return True

        # Skip if no changes and configured to skip
        if config.skip_review_if_no_changes and not has_file_changes:
            return True

        return False

    def get_timeout(self, task_description: str) -> int:
        """
        Get timeout in seconds for a task.

        Args:
            task_description: The task description

        Returns:
            Timeout in seconds
        """
        config = self.get_config(task_description)
        return config.get_timeout()

    def get_max_retries(self, task_description: str) -> int:
        """
        Get maximum retry attempts for a task.

        Args:
            task_description: The task description

        Returns:
            Maximum retry attempts
        """
        config = self.get_config(task_description)
        return config.max_retries

    def should_run_tests(self, task_description: str, task_failed: bool = False) -> bool:
        """
        Determine if tests should run after task completion.

        Args:
            task_description: The task description
            task_failed: Whether the task failed

        Returns:
            True if tests should run
        """
        config = self.get_config(task_description)

        if config.test_on_failure_only:
            return task_failed

        return config.run_tests_after


# Module-level cached instance
_cached_router: TaskRouter | None = None


def get_task_router() -> TaskRouter:
    """
    Get cached TaskRouter instance.

    Returns:
        Cached TaskRouter for performance
    """
    global _cached_router
    if _cached_router is None:
        _cached_router = TaskRouter()
    return _cached_router


def reset_task_router() -> None:
    """Reset cached router instance."""
    global _cached_router
    _cached_router = None
