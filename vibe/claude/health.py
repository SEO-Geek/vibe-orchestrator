"""
Claude Health Monitor

Monitors Claude subprocess health and provides restart capabilities.
Ensures GLM always has control over Claude execution state.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.claude.executor import ClaudeExecutor

logger = logging.getLogger(__name__)


class ClaudeStatus(Enum):
    """Claude subprocess health status."""

    IDLE = auto()         # No active task
    RUNNING = auto()      # Task in progress
    HUNG = auto()         # No output for too long
    FAILED = auto()       # Execution failed
    RESTARTING = auto()   # Being restarted


@dataclass
class HealthStats:
    """Health statistics for monitoring."""

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    restarts: int = 0
    hung_detections: int = 0
    last_activity: datetime | None = None
    last_success: datetime | None = None
    last_failure: datetime | None = None
    current_task_start: datetime | None = None

    def success_rate(self) -> float:
        """Calculate success rate (0.0 - 1.0)."""
        if self.total_tasks == 0:
            return 1.0
        return self.successful_tasks / self.total_tasks


@dataclass
class HealthCheck:
    """Result of a health check."""

    status: ClaudeStatus
    healthy: bool
    message: str
    stats: HealthStats
    seconds_since_activity: float | None = None
    recommendation: str = ""


class ClaudeHealthMonitor:
    """
    Monitors Claude executor health and manages restarts.

    GLM uses this to:
    1. Check if Claude is responsive before sending tasks
    2. Detect hung processes (no output for hang_timeout seconds)
    3. Restart Claude if needed
    4. Get health statistics for decision making
    """

    def __init__(
        self,
        hang_timeout: float = 120.0,  # 2 minutes without output = hung
        max_restarts: int = 3,         # Max restarts before giving up
        restart_cooldown: float = 10.0, # Wait between restarts
    ):
        """
        Initialize health monitor.

        Args:
            hang_timeout: Seconds without output before considering hung
            max_restarts: Maximum restart attempts before failing
            restart_cooldown: Seconds to wait between restart attempts
        """
        self.hang_timeout = hang_timeout
        self.max_restarts = max_restarts
        self.restart_cooldown = restart_cooldown

        self._status = ClaudeStatus.IDLE
        self._stats = HealthStats()
        self._last_output_time: float | None = None
        self._restart_count = 0
        self._last_restart_time: float | None = None
        self._executor: ClaudeExecutor | None = None

    def attach_executor(self, executor: "ClaudeExecutor") -> None:
        """Attach an executor to monitor."""
        self._executor = executor
        logger.info("Health monitor attached to executor")

    def detach_executor(self) -> None:
        """Detach the current executor."""
        self._executor = None
        self._status = ClaudeStatus.IDLE

    def record_task_start(self) -> None:
        """Record that a task has started."""
        self._status = ClaudeStatus.RUNNING
        self._stats.total_tasks += 1
        self._stats.current_task_start = datetime.now()
        self._last_output_time = time.time()
        logger.debug("Task started, monitoring health")

    def record_output(self) -> None:
        """Record that output was received (heartbeat)."""
        self._last_output_time = time.time()
        self._stats.last_activity = datetime.now()

    def record_task_success(self) -> None:
        """Record successful task completion."""
        self._status = ClaudeStatus.IDLE
        self._stats.successful_tasks += 1
        self._stats.last_success = datetime.now()
        self._stats.current_task_start = None
        self._restart_count = 0  # Reset restart count on success
        logger.debug("Task succeeded, health OK")

    def record_task_failure(self, error: str = "") -> None:
        """Record task failure."""
        self._status = ClaudeStatus.FAILED
        self._stats.failed_tasks += 1
        self._stats.last_failure = datetime.now()
        self._stats.current_task_start = None
        logger.warning(f"Task failed: {error}")

    def check_health(self) -> HealthCheck:
        """
        Check current health status.

        Returns:
            HealthCheck with status, stats, and recommendations
        """
        seconds_since_activity = None
        recommendation = ""

        # Check for hung process
        if self._status == ClaudeStatus.RUNNING and self._last_output_time:
            seconds_since_activity = time.time() - self._last_output_time

            if seconds_since_activity > self.hang_timeout:
                self._status = ClaudeStatus.HUNG
                self._stats.hung_detections += 1
                logger.warning(
                    f"Claude appears hung - no output for {seconds_since_activity:.1f}s"
                )

        # Determine if healthy
        healthy = self._status in (ClaudeStatus.IDLE, ClaudeStatus.RUNNING)

        # Generate message and recommendation
        if self._status == ClaudeStatus.IDLE:
            message = "Claude is idle and ready for tasks"
        elif self._status == ClaudeStatus.RUNNING:
            message = "Claude is executing a task"
            if seconds_since_activity and seconds_since_activity > 60:
                recommendation = "Task taking long, consider increasing timeout"
        elif self._status == ClaudeStatus.HUNG:
            message = f"Claude appears hung (no output for {seconds_since_activity:.0f}s)"
            recommendation = "Recommend restart"
        elif self._status == ClaudeStatus.FAILED:
            message = "Last task failed"
            recommendation = "Check error and retry or restart"
        elif self._status == ClaudeStatus.RESTARTING:
            message = "Claude is being restarted"
        else:
            message = f"Unknown status: {self._status}"

        return HealthCheck(
            status=self._status,
            healthy=healthy,
            message=message,
            stats=self._stats,
            seconds_since_activity=seconds_since_activity,
            recommendation=recommendation,
        )

    def can_restart(self) -> bool:
        """Check if restart is allowed."""
        # Check restart count
        if self._restart_count >= self.max_restarts:
            logger.error(f"Max restarts ({self.max_restarts}) exceeded")
            return False

        # Check cooldown
        if self._last_restart_time:
            elapsed = time.time() - self._last_restart_time
            if elapsed < self.restart_cooldown:
                logger.debug(f"Restart cooldown: {self.restart_cooldown - elapsed:.1f}s remaining")
                return False

        return True

    async def restart_executor(self, project_path: str) -> "ClaudeExecutor | None":
        """
        Restart the Claude executor.

        Args:
            project_path: Project path for new executor

        Returns:
            New executor if successful, None if restart not allowed
        """
        from vibe.claude.executor import ClaudeExecutor

        if not self.can_restart():
            return None

        self._status = ClaudeStatus.RESTARTING
        self._restart_count += 1
        self._last_restart_time = time.time()
        self._stats.restarts += 1

        logger.info(f"Restarting Claude executor (attempt {self._restart_count}/{self.max_restarts})")

        # Close existing executor if present
        if self._executor:
            try:
                self._executor.close()
            except Exception as e:
                logger.debug(f"Error closing old executor: {e}")

        # Small delay to ensure cleanup
        await asyncio.sleep(0.5)

        # Create new executor
        try:
            new_executor = ClaudeExecutor(
                project_path=project_path,
                timeout_tier="code",
                permission_mode="bypassPermissions",
            )
            self.attach_executor(new_executor)
            self._status = ClaudeStatus.IDLE
            logger.info("Claude executor restarted successfully")
            return new_executor
        except Exception as e:
            logger.error(f"Failed to restart executor: {e}")
            self._status = ClaudeStatus.FAILED
            return None

    def get_status_for_glm(self) -> dict:
        """
        Get health status formatted for GLM consumption.

        Returns:
            Dict with health info GLM can use for decisions
        """
        check = self.check_health()
        return {
            "claude_status": check.status.name,
            "claude_healthy": check.healthy,
            "claude_message": check.message,
            "success_rate": f"{check.stats.success_rate():.1%}",
            "total_tasks": check.stats.total_tasks,
            "restarts": check.stats.restarts,
            "recommendation": check.recommendation,
        }

    def should_glm_intervene(self) -> bool:
        """
        Check if GLM should intervene (restart, abort, etc).

        Returns:
            True if GLM should take action
        """
        check = self.check_health()

        # Intervene if hung
        if check.status == ClaudeStatus.HUNG:
            return True

        # Intervene if too many failures
        if check.stats.failed_tasks >= 3 and check.stats.success_rate() < 0.5:
            return True

        return False


# Global health monitor instance (singleton)
_health_monitor: ClaudeHealthMonitor | None = None


def get_health_monitor() -> ClaudeHealthMonitor:
    """Get or create the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ClaudeHealthMonitor()
    return _health_monitor


def reset_health_monitor() -> None:
    """Reset the global health monitor (for testing)."""
    global _health_monitor
    _health_monitor = None
