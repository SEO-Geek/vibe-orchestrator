"""
Task Board - Kanban-style task visualization and management.

Provides visual task tracking with columns:
- BACKLOG: Tasks queued for execution
- IN_PROGRESS: Currently being executed by Claude
- REVIEW: Awaiting GLM code review
- DONE: Successfully completed
- REJECTED: Failed after max retries

Features:
- SQLite persistence for cross-session tracking
- WIP limits (configurable per column)
- Task history with timestamps
- Web UI export (JSON for rendering)
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from vibe.config import CONFIG_DIR

logger = logging.getLogger(__name__)

# Database path for task board
TASK_BOARD_DB = CONFIG_DIR / "task_board.db"


class TaskColumn(Enum):
    """Kanban board columns for task states."""

    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    REJECTED = "rejected"
    BLOCKED = "blocked"


# Default WIP (Work In Progress) limits per column
# Limits how many tasks can be in each state simultaneously
DEFAULT_WIP_LIMITS: dict[TaskColumn, int] = {
    TaskColumn.BACKLOG: 50,  # No effective limit for backlog
    TaskColumn.IN_PROGRESS: 1,  # Only one task executing at a time
    TaskColumn.REVIEW: 3,  # Can queue up to 3 for review
    TaskColumn.DONE: 100,  # Archive limit
    TaskColumn.REJECTED: 20,  # Keep recent failures
    TaskColumn.BLOCKED: 10,  # Blocked tasks
}


@dataclass
class BoardTask:
    """A task on the kanban board."""

    id: str
    description: str
    column: TaskColumn
    project: str
    session_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempts: int = 0
    last_feedback: str = ""
    files_changed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "column": self.column.value,
            "project": self.project,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempts": self.attempts,
            "last_feedback": self.last_feedback,
            "files_changed": self.files_changed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "BoardTask":
        """Create BoardTask from database row."""
        return cls(
            id=row["id"],
            description=row["description"],
            column=TaskColumn(row["column"]),
            project=row["project"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now(),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            attempts=row["attempts"] or 0,
            last_feedback=row["last_feedback"] or "",
            files_changed=json.loads(row["files_changed"]) if row["files_changed"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


@dataclass
class BoardStats:
    """Statistics for the task board."""

    total_tasks: int = 0
    by_column: dict[str, int] = field(default_factory=dict)
    completed_today: int = 0
    failed_today: int = 0
    avg_completion_time_seconds: float = 0.0


class TaskBoard:
    """
    Kanban-style task board for visualizing Vibe workflow.

    Thread-safe SQLite-backed task management with WIP limits.
    """

    def __init__(
        self,
        project: str,
        wip_limits: dict[TaskColumn, int] | None = None,
        db_path: Path | None = None,
    ):
        """
        Initialize task board for a project.

        Args:
            project: Project name for isolation
            wip_limits: Custom WIP limits per column
            db_path: Custom database path (default: ~/.config/vibe/task_board.db)
        """
        self.project = project
        self.wip_limits = wip_limits or DEFAULT_WIP_LIMITS.copy()
        self._db_path = db_path or TASK_BOARD_DB
        self._lock = threading.Lock()

        # Ensure database exists
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS board_tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    column TEXT NOT NULL,
                    project TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    attempts INTEGER DEFAULT 0,
                    last_feedback TEXT,
                    files_changed TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_board_tasks_project
                ON board_tasks(project)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_board_tasks_column
                ON board_tasks(column)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    from_column TEXT,
                    to_column TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT
                )
            """)

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def add_task(
        self,
        task_id: str,
        description: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoardTask:
        """
        Add a task to the backlog.

        Args:
            task_id: Unique task identifier
            description: Task description
            session_id: Optional session ID
            metadata: Optional metadata

        Returns:
            Created BoardTask

        Raises:
            ValueError: If WIP limit exceeded
        """
        with self._lock:
            # Check WIP limit for backlog
            if self._get_column_count(TaskColumn.BACKLOG) >= self.wip_limits[TaskColumn.BACKLOG]:
                raise ValueError(f"Backlog WIP limit ({self.wip_limits[TaskColumn.BACKLOG]}) exceeded")

            now = datetime.now()
            task = BoardTask(
                id=task_id,
                description=description,
                column=TaskColumn.BACKLOG,
                project=self.project,
                session_id=session_id,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )

            with self._get_connection() as conn:
                # Use INSERT OR REPLACE to handle duplicate task IDs gracefully
                # This can happen if a task is re-queued or tests reuse IDs
                conn.execute(
                    """
                    INSERT OR REPLACE INTO board_tasks
                    (id, description, column, project, session_id, created_at, updated_at,
                     metadata, attempts, last_feedback, files_changed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, '', '[]')
                """,
                    (
                        task.id,
                        task.description,
                        task.column.value,
                        task.project,
                        task.session_id,
                        task.created_at.isoformat(),
                        task.updated_at.isoformat(),
                        json.dumps(task.metadata),
                    ),
                )

                self._record_transition(conn, task.id, None, TaskColumn.BACKLOG, "Task created")
                conn.commit()

            logger.info(f"TaskBoard: Added task {task_id} to backlog")
            return task

    def move_to(
        self,
        task_id: str,
        column: TaskColumn,
        reason: str = "",
        feedback: str = "",
        files_changed: list[str] | None = None,
    ) -> BoardTask | None:
        """
        Move a task to a different column.

        Args:
            task_id: Task to move
            column: Target column
            reason: Reason for move
            feedback: Optional feedback (for rejections)
            files_changed: Files changed (for completions)

        Returns:
            Updated BoardTask or None if not found

        Raises:
            ValueError: If WIP limit exceeded for target column
        """
        with self._lock:
            # Check WIP limit for target column
            current_count = self._get_column_count(column)
            if current_count >= self.wip_limits[column]:
                raise ValueError(f"{column.value} WIP limit ({self.wip_limits[column]}) exceeded")

            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM board_tasks WHERE id = ? AND project = ?",
                    (task_id, self.project),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                old_column = TaskColumn(row["column"])
                now = datetime.now()

                # Update timestamps based on column
                updates = {
                    "column": column.value,
                    "updated_at": now.isoformat(),
                }

                if column == TaskColumn.IN_PROGRESS:
                    updates["started_at"] = now.isoformat()
                    updates["attempts"] = (row["attempts"] or 0) + 1

                if column in (TaskColumn.DONE, TaskColumn.REJECTED):
                    updates["completed_at"] = now.isoformat()

                if feedback:
                    updates["last_feedback"] = feedback

                if files_changed:
                    updates["files_changed"] = json.dumps(files_changed)

                # Build update query
                set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                values = list(updates.values()) + [task_id, self.project]

                conn.execute(
                    f"UPDATE board_tasks SET {set_clause} WHERE id = ? AND project = ?",
                    values,
                )

                self._record_transition(conn, task_id, old_column, column, reason)
                conn.commit()

                # Fetch updated task
                cursor = conn.execute(
                    "SELECT * FROM board_tasks WHERE id = ?",
                    (task_id,),
                )
                updated_row = cursor.fetchone()

            logger.info(f"TaskBoard: Moved task {task_id} from {old_column.value} to {column.value}")
            return BoardTask.from_row(updated_row) if updated_row else None

    def start_task(self, task_id: str) -> BoardTask | None:
        """Move task to IN_PROGRESS."""
        return self.move_to(task_id, TaskColumn.IN_PROGRESS, "Task execution started")

    def send_to_review(self, task_id: str) -> BoardTask | None:
        """Move task to REVIEW."""
        return self.move_to(task_id, TaskColumn.REVIEW, "Sent to GLM review")

    def complete_task(
        self,
        task_id: str,
        files_changed: list[str] | None = None,
    ) -> BoardTask | None:
        """Move task to DONE."""
        return self.move_to(
            task_id,
            TaskColumn.DONE,
            "Task completed successfully",
            files_changed=files_changed,
        )

    def reject_task(self, task_id: str, feedback: str) -> BoardTask | None:
        """Move task to REJECTED."""
        return self.move_to(
            task_id,
            TaskColumn.REJECTED,
            "Task rejected after max retries",
            feedback=feedback,
        )

    def retry_task(self, task_id: str, feedback: str) -> BoardTask | None:
        """Move task back to BACKLOG for retry."""
        return self.move_to(
            task_id,
            TaskColumn.BACKLOG,
            "Retrying after rejection",
            feedback=feedback,
        )

    def get_task(self, task_id: str) -> BoardTask | None:
        """Get a task by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM board_tasks WHERE id = ? AND project = ?",
                (task_id, self.project),
            )
            row = cursor.fetchone()
            return BoardTask.from_row(row) if row else None

    def get_tasks_by_column(self, column: TaskColumn) -> list[BoardTask]:
        """Get all tasks in a column."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM board_tasks WHERE column = ? AND project = ? ORDER BY created_at ASC",
                (column.value, self.project),
            )
            return [BoardTask.from_row(row) for row in cursor.fetchall()]

    def get_all_tasks(self) -> dict[str, list[BoardTask]]:
        """Get all tasks organized by column."""
        result = {column.value: [] for column in TaskColumn}

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM board_tasks WHERE project = ? ORDER BY created_at ASC",
                (self.project,),
            )
            for row in cursor.fetchall():
                task = BoardTask.from_row(row)
                result[task.column.value].append(task)

        return result

    def get_board_json(self) -> str:
        """
        Get board state as JSON for web UI rendering.

        Returns:
            JSON string with all columns and tasks
        """
        tasks = self.get_all_tasks()
        stats = self.get_stats()

        board_data = {
            "project": self.project,
            "columns": [
                {
                    "id": column.value,
                    "name": column.value.replace("_", " ").title(),
                    "wip_limit": self.wip_limits[column],
                    "task_count": len(tasks[column.value]),
                    "tasks": [t.to_dict() for t in tasks[column.value]],
                }
                for column in TaskColumn
            ],
            "stats": {
                "total_tasks": stats.total_tasks,
                "completed_today": stats.completed_today,
                "failed_today": stats.failed_today,
                "avg_completion_time_seconds": stats.avg_completion_time_seconds,
            },
            "generated_at": datetime.now().isoformat(),
        }

        return json.dumps(board_data, indent=2)

    def get_stats(self) -> BoardStats:
        """Get board statistics."""
        stats = BoardStats()
        today = datetime.now().date().isoformat()

        with self._get_connection() as conn:
            # Total tasks
            cursor = conn.execute(
                "SELECT COUNT(*) FROM board_tasks WHERE project = ?",
                (self.project,),
            )
            stats.total_tasks = cursor.fetchone()[0]

            # By column
            cursor = conn.execute(
                "SELECT column, COUNT(*) as cnt FROM board_tasks WHERE project = ? GROUP BY column",
                (self.project,),
            )
            stats.by_column = {row["column"]: row["cnt"] for row in cursor.fetchall()}

            # Completed today
            cursor = conn.execute(
                "SELECT COUNT(*) FROM board_tasks WHERE project = ? AND column = ? AND completed_at LIKE ?",
                (self.project, TaskColumn.DONE.value, f"{today}%"),
            )
            stats.completed_today = cursor.fetchone()[0]

            # Failed today
            cursor = conn.execute(
                "SELECT COUNT(*) FROM board_tasks WHERE project = ? AND column = ? AND completed_at LIKE ?",
                (self.project, TaskColumn.REJECTED.value, f"{today}%"),
            )
            stats.failed_today = cursor.fetchone()[0]

            # Average completion time
            cursor = conn.execute(
                """
                SELECT AVG(
                    (julianday(completed_at) - julianday(started_at)) * 86400
                ) as avg_time
                FROM board_tasks
                WHERE project = ? AND completed_at IS NOT NULL AND started_at IS NOT NULL
            """,
                (self.project,),
            )
            row = cursor.fetchone()
            stats.avg_completion_time_seconds = row[0] or 0.0

        return stats

    def _get_column_count(self, column: TaskColumn) -> int:
        """Get current task count for a column."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM board_tasks WHERE column = ? AND project = ?",
                (column.value, self.project),
            )
            return cursor.fetchone()[0]

    def _record_transition(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        from_column: TaskColumn | None,
        to_column: TaskColumn,
        reason: str,
    ) -> None:
        """Record a task state transition in history."""
        conn.execute(
            """
            INSERT INTO task_history (task_id, from_column, to_column, timestamp, reason)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                task_id,
                from_column.value if from_column else None,
                to_column.value,
                datetime.now().isoformat(),
                reason,
            ),
        )

    def get_task_history(self, task_id: str) -> list[dict[str, Any]]:
        """Get transition history for a task."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM task_history WHERE task_id = ? ORDER BY timestamp ASC",
                (task_id,),
            )
            return [
                {
                    "from_column": row["from_column"],
                    "to_column": row["to_column"],
                    "timestamp": row["timestamp"],
                    "reason": row["reason"],
                }
                for row in cursor.fetchall()
            ]

    def cleanup_old_tasks(self, days: int = 7) -> int:
        """
        Remove tasks older than specified days from DONE/REJECTED.

        Args:
            days: Number of days to retain

        Returns:
            Number of tasks deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM board_tasks
                    WHERE project = ?
                      AND column IN (?, ?)
                      AND completed_at < ?
                """,
                    (
                        self.project,
                        TaskColumn.DONE.value,
                        TaskColumn.REJECTED.value,
                        cutoff,
                    ),
                )
                deleted = cursor.rowcount
                conn.commit()

        logger.info(f"TaskBoard: Cleaned up {deleted} old tasks")
        return deleted


# Module-level cached instance per project
_cached_boards: dict[str, TaskBoard] = {}
_boards_lock = threading.Lock()


def get_task_board(project: str) -> TaskBoard:
    """
    Get or create TaskBoard for a project.

    Args:
        project: Project name

    Returns:
        Cached TaskBoard instance
    """
    with _boards_lock:
        if project not in _cached_boards:
            _cached_boards[project] = TaskBoard(project)
        return _cached_boards[project]


def reset_task_boards() -> None:
    """Reset all cached task boards."""
    global _cached_boards
    with _boards_lock:
        _cached_boards = {}
