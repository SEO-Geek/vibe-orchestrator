"""
Task Queue - Manages pending tasks for Claude

Placeholder for Phase 5 implementation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from vibe.exceptions import TaskQueueFullError
from vibe.state import Task


@dataclass
class TaskQueue:
    """
    Async task queue for Claude executions.

    Limits concurrent Claude processes and provides ordering.
    """

    max_concurrent: int = 1  # Usually 1 for sequential execution
    max_pending: int = 50

    _queue: asyncio.Queue[Task] = field(default_factory=lambda: asyncio.Queue())
    _active_tasks: int = 0
    _results: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def add(self, task: Task) -> None:
        """
        Add a task to the queue.

        Args:
            task: Task to queue

        Raises:
            TaskQueueFullError: If queue is at capacity
        """
        if self._queue.qsize() >= self.max_pending:
            raise TaskQueueFullError(
                f"Task queue full (max {self.max_pending})",
                {"current_size": self._queue.qsize()},
            )

        await self._queue.put(task)

    async def process(
        self,
        executor: Callable[[Task], Any],
        on_complete: Callable[[Task, dict[str, Any]], None] | None = None,
    ) -> None:
        """
        Process all tasks in queue.

        Args:
            executor: Function to execute each task
            on_complete: Callback when task completes
        """
        while not self._queue.empty():
            task = await self._queue.get()

            try:
                self._active_tasks += 1
                result = await executor(task)
                self._results[task.id] = result

                if on_complete:
                    on_complete(task, result)

            finally:
                self._active_tasks -= 1
                self._queue.task_done()

    def get_result(self, task_id: str) -> dict[str, Any] | None:
        """Get result for a completed task."""
        return self._results.get(task_id)

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        """Number of currently executing tasks."""
        return self._active_tasks

    def clear(self) -> None:
        """Clear all pending tasks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._results.clear()
