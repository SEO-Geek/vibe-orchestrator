"""
Task History - Bulletproof task tracking for Vibe.

Maintains task history in-memory with optional database persistence.
ALWAYS works, even if database fails.

Thread-safe via lock for all mutations.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    """A completed task record."""

    description: str
    status: str  # "completed" or "failed"
    summary: str
    files_changed: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_context_line(self) -> str:
        """Format for GLM context."""
        status_icon = "✓" if self.status == "completed" else "✗"
        return f"[{status_icon}] {self.description[:80]}"


@dataclass
class RequestRecord:
    """A user request record."""

    request: str
    result: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_context_line(self) -> str:
        """Format for GLM context."""
        return f"- {self.request[:100]}"


class TaskHistory:
    """
    In-memory task history with database backup.

    This class ALWAYS works. If database fails, it still tracks
    tasks in memory for the current session.

    Thread-safe: all mutations protected by lock.
    """

    # Class-level storage (survives across function calls)
    _tasks: list[TaskRecord] = []
    _requests: list[RequestRecord] = []
    _max_history: int = 50
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def add_task(
        cls,
        description: str,
        success: bool,
        summary: str = "",
        files_changed: list[str] | None = None,
    ) -> None:
        """
        Record a completed task.

        Args:
            description: Task description
            success: Whether task succeeded
            summary: Result summary
            files_changed: List of modified files
        """
        record = TaskRecord(
            description=description,
            status="completed" if success else "failed",
            summary=summary,
            files_changed=files_changed or [],
        )

        with cls._lock:
            cls._tasks.append(record)
            # Trim to max history
            if len(cls._tasks) > cls._max_history:
                cls._tasks = cls._tasks[-cls._max_history :]

        logger.debug(f"TaskHistory: Added task '{description[:50]}' ({record.status})")

    @classmethod
    def add_request(cls, request: str, result: str = "") -> None:
        """
        Record a user request.

        Args:
            request: The user's request text
            result: Outcome summary
        """
        record = RequestRecord(request=request, result=result)

        with cls._lock:
            cls._requests.append(record)
            # Trim to max history
            if len(cls._requests) > cls._max_history:
                cls._requests = cls._requests[-cls._max_history :]

        logger.debug(f"TaskHistory: Added request '{request[:50]}'")

    @classmethod
    def get_recent_tasks(cls, limit: int = 10) -> list[TaskRecord]:
        """Get recent tasks, newest first."""
        with cls._lock:
            return list(reversed(cls._tasks[-limit:]))

    @classmethod
    def get_recent_requests(cls, limit: int = 5) -> list[RequestRecord]:
        """Get recent requests, newest first."""
        with cls._lock:
            return list(reversed(cls._requests[-limit:]))

    @classmethod
    def get_context_for_glm(cls) -> str:
        """
        Get formatted task history for GLM context.

        Returns:
            Formatted string ready to inject into GLM context.
            Returns empty string if no history.
        """
        parts = []

        # Recent tasks
        tasks = cls.get_recent_tasks(10)
        if tasks:
            parts.append("## Recent Tasks Executed:")
            for task in tasks:
                parts.append(task.to_context_line())
            parts.append("")
            parts.append("(If user says 'redo', 'retry', or 'the tasks' - they mean these)")
            parts.append("")

        # Recent requests
        requests = cls.get_recent_requests(5)
        if requests:
            parts.append("## Recent User Requests:")
            for req in requests:
                parts.append(req.to_context_line())
            parts.append("")

        return "\n".join(parts)

    @classmethod
    def has_history(cls) -> bool:
        """Check if any history exists."""
        with cls._lock:
            return bool(cls._tasks or cls._requests)

    @classmethod
    def clear(cls) -> None:
        """Clear all history (for testing only)."""
        with cls._lock:
            cls._tasks = []
            cls._requests = []

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        """Get history statistics."""
        with cls._lock:
            completed = sum(1 for t in cls._tasks if t.status == "completed")
            failed = sum(1 for t in cls._tasks if t.status == "failed")
            return {
                "total_tasks": len(cls._tasks),
                "completed": completed,
                "failed": failed,
                "requests": len(cls._requests),
            }

    @classmethod
    def load_from_memory(cls, memory: Any) -> None:
        """
        Load history from VibeMemory database (if available).

        This is called at session start to restore previous history.
        Failures are silently ignored - we'll just start fresh.

        Args:
            memory: VibeMemory instance (or None)
        """
        if not memory:
            return

        try:
            # Load task items from database
            items = memory.load_project_context(limit=50)

            with cls._lock:
                for item in items:
                    # Task items
                    if item.key.startswith("task-") or item.category == "task":
                        lines = item.value.split("\n")
                        desc = ""
                        status = "completed"
                        summary = ""

                        for line in lines:
                            if line.startswith("Task:"):
                                desc = line[5:].strip()
                            elif line.startswith("Status:"):
                                status = line[7:].strip()
                            elif line.startswith("Summary:"):
                                summary = line[8:].strip()

                        if desc and desc not in [t.description for t in cls._tasks]:
                            cls._tasks.append(
                                TaskRecord(
                                    description=desc,
                                    status=status,
                                    summary=summary,
                                    timestamp=item.created_at,
                                )
                            )

                    # Request items
                    elif item.key.startswith("request-"):
                        lines = item.value.split("\n")
                        req = ""
                        result = ""

                        for line in lines:
                            if line.startswith("Request:"):
                                req = line[8:].strip()
                            elif line.startswith("Result:"):
                                result = line[7:].strip()

                        if req and req not in [r.request for r in cls._requests]:
                            cls._requests.append(
                                RequestRecord(
                                    request=req,
                                    result=result,
                                    timestamp=item.created_at,
                                )
                            )

                # Sort by timestamp
                cls._tasks.sort(key=lambda t: t.timestamp)
                cls._requests.sort(key=lambda r: r.timestamp)

            task_count = len(cls._tasks)
            req_count = len(cls._requests)
            logger.info(f"TaskHistory: Loaded {task_count} tasks, {req_count} requests")

        except Exception as e:
            logger.warning(f"TaskHistory: Could not load from database: {e}")
            # Continue with empty history - not a critical failure


# Convenience functions for easy access
def add_task(
    description: str, success: bool, summary: str = "", files_changed: list[str] | None = None
) -> None:
    """Record a completed task."""
    TaskHistory.add_task(description, success, summary, files_changed)


def add_request(request: str, result: str = "") -> None:
    """Record a user request."""
    TaskHistory.add_request(request, result)


def get_context_for_glm() -> str:
    """Get formatted task history for GLM."""
    return TaskHistory.get_context_for_glm()


def load_from_memory(memory: Any) -> None:
    """Load history from database."""
    TaskHistory.load_from_memory(memory)
