"""
Vibe Repository - Database access layer

Provides all database operations for the Vibe Orchestrator.
Single connection per repository instance, with context manager support.

Thread Safety:
- SQLite in WAL mode for concurrent reads
- Write operations are serialized
- Use separate VibeRepository instances per thread

Crash Recovery:
- Heartbeat-based orphan detection
- Session status tracking
- Checkpoint/rollback support
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from vibe.persistence.models import (
    AttemptResult,
    Checkpoint,
    ContextCategory,
    ContextItem,
    Convention,
    DebugIteration,
    DebugSession,
    FileChange,
    Message,
    MessageRole,
    MessageType,
    Priority,
    # Models
    Project,
    Request,
    Review,
    Session,
    # Enums
    SessionStatus,
    Task,
    TaskAttempt,
    TaskStatus,
    ToolUsage,
    # Helpers
    generate_id,
    now_iso,
    to_json,
)

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".config" / "vibe" / "vibe.db"


class VibeRepository:
    """
    Repository for all Vibe persistence operations.

    Usage:
        repo = VibeRepository()
        repo.initialize()

        # Create a session
        project = repo.get_or_create_project("myproject", "/path/to/project")
        session = repo.start_session(project.id)

        # Use in context manager for auto-cleanup
        with repo:
            ...

        # Or close manually
        repo.close()
    """

    def __init__(self, db_path: Path | str | None = None):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def __enter__(self) -> VibeRepository:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection, initializing if needed."""
        if self._conn is None:
            self.initialize()
        return self._conn  # type: ignore

    def initialize(self) -> None:
        """
        Initialize database connection and schema.

        Creates database file and parent directories if they don't exist.
        Applies schema if not already present.
        """
        if self._initialized and self._conn:
            return

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with optimal settings for our use case
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # We handle thread safety manually
            isolation_level=None,  # Autocommit mode, we use explicit transactions
        )

        # Enable WAL mode for concurrent reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys=ON")
        # Faster writes (data still durable with WAL)
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Apply schema
        self._apply_schema()
        self._initialized = True

        logger.info(f"Initialized Vibe database at {self.db_path}")

    def _apply_schema(self) -> None:
        """Apply the database schema from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path) as f:
            schema_sql = f.read()

        # Execute schema (CREATE IF NOT EXISTS makes this idempotent)
        self._conn.executescript(schema_sql)  # type: ignore
        logger.debug("Database schema applied")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Execute operations in a transaction.

        Usage:
            with repo.transaction() as cursor:
                cursor.execute(...)
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            yield cursor
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    # =========================================================================
    # PROJECT OPERATIONS
    # =========================================================================

    def get_or_create_project(
        self,
        name: str,
        path: str,
        starmap: str = "STARMAP.md",
        claude_md: str = "CLAUDE.md",
        test_command: str = "pytest -v",
        description: str | None = None,
    ) -> Project:
        """
        Get existing project by name or create new one.

        Uses atomic upsert to avoid race conditions.

        Args:
            name: Project name (unique)
            path: Absolute filesystem path
            starmap: Starmap filename
            claude_md: CLAUDE.md filename
            test_command: Default test command
            description: Optional description

        Returns:
            Project instance
        """
        cursor = self.conn.cursor()
        now = now_iso()

        # Use atomic upsert to handle race conditions
        # INSERT OR IGNORE + UPDATE pattern for SQLite compatibility
        project_id = generate_id()
        try:
            cursor.execute(
                """INSERT INTO projects
                   (id, name, path, starmap, claude_md, test_command, description,
                    created_at, updated_at, last_accessed_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id,
                    name,
                    path,
                    starmap,
                    claude_md,
                    test_command,
                    description,
                    now,
                    now,
                    now,
                    1,
                ),
            )
            logger.info(f"Created new project: {name}")
        except sqlite3.IntegrityError:
            # Project exists - update path if changed
            cursor.execute(
                """UPDATE projects
                   SET path = ?, updated_at = ?, last_accessed_at = ?
                   WHERE name = ? AND is_active = 1 AND path != ?""",
                (path, now, now, name, path),
            )
            logger.debug(f"Project exists: {name}")

        # Fetch and return the project
        cursor.execute("SELECT * FROM projects WHERE name = ? AND is_active = 1", (name,))
        row = cursor.fetchone()
        return Project.from_row(row)

    def get_project(self, project_id: str) -> Project | None:
        """Get project by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        return Project.from_row(row) if row else None

    def get_project_by_name(self, name: str) -> Project | None:
        """Get project by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE name = ? AND is_active = 1", (name,))
        row = cursor.fetchone()
        return Project.from_row(row) if row else None

    def list_projects(self, include_inactive: bool = False) -> list[Project]:
        """List all projects."""
        cursor = self.conn.cursor()
        if include_inactive:
            cursor.execute("SELECT * FROM projects ORDER BY last_accessed_at DESC")
        else:
            cursor.execute(
                "SELECT * FROM projects WHERE is_active = 1 ORDER BY last_accessed_at DESC"
            )
        return [Project.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================

    def start_session(
        self,
        project_id: str,
        detect_orphans: bool = True,
    ) -> Session:
        """
        Start a new session for a project.

        Uses transaction for atomicity (orphan detection + session creation).

        Args:
            project_id: Project ID
            detect_orphans: Whether to mark orphaned sessions

        Returns:
            New Session instance
        """
        session = Session(
            project_id=project_id,
            status=SessionStatus.ACTIVE,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            last_heartbeat_at=datetime.now(),
        )

        with self.transaction() as cursor:
            if detect_orphans:
                self._detect_and_mark_orphans(project_id)

            cursor.execute(
                """INSERT INTO sessions
                   (id, project_id, status, pid, hostname, started_at, ended_at,
                    last_heartbeat_at, summary, error_message, total_tasks_completed,
                    total_tasks_failed, total_cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                session.to_row(),
            )

        logger.info(f"Started session {session.id[:8]} for project {project_id[:8]}")
        return session

    def _detect_and_mark_orphans(
        self,
        project_id: str,
        heartbeat_threshold_minutes: int = 5,
    ) -> int:
        """
        Detect and mark orphaned sessions.

        A session is orphaned if:
        - Status is 'active'
        - No heartbeat in threshold_minutes

        Returns:
            Number of orphaned sessions marked
        """
        threshold = datetime.now() - timedelta(minutes=heartbeat_threshold_minutes)

        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE sessions
               SET status = 'crashed',
                   ended_at = ?,
                   error_message = 'Session detected as orphaned (no heartbeat)'
               WHERE project_id = ?
                 AND status = 'active'
                 AND last_heartbeat_at < ?""",
            (now_iso(), project_id, threshold.isoformat()),
        )

        count = cursor.rowcount
        if count > 0:
            logger.warning(f"Marked {count} orphaned sessions for project {project_id[:8]}")

        return count

    def update_heartbeat(self, session_id: str) -> None:
        """Update session heartbeat timestamp."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET last_heartbeat_at = ? WHERE id = ?", (now_iso(), session_id)
        )

    def end_session(
        self,
        session_id: str,
        summary: str | None = None,
        status: SessionStatus = SessionStatus.COMPLETED,
        error_message: str | None = None,
    ) -> Session | None:
        """End a session with given status. Returns updated session."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE sessions
               SET status = ?, ended_at = ?, summary = ?, error_message = ?
               WHERE id = ?""",
            (status.value, now_iso(), summary, error_message, session_id),
        )
        logger.info(f"Ended session {session_id[:8]} with status {status.value}")
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return Session.from_row(row) if row else None

    def get_active_session(self, project_id: str) -> Session | None:
        """Get the active session for a project (if any)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions WHERE project_id = ? AND status = 'active' LIMIT 1",
            (project_id,),
        )
        row = cursor.fetchone()
        return Session.from_row(row) if row else None

    def list_sessions(
        self,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[Session]:
        """List sessions, optionally filtered by project."""
        cursor = self.conn.cursor()
        if project_id:
            cursor.execute(
                """SELECT * FROM sessions
                   WHERE project_id = ?
                   ORDER BY started_at DESC LIMIT ?""",
                (project_id, limit),
            )
        else:
            cursor.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))
        return [Session.from_row(row) for row in cursor.fetchall()]

    def get_orphaned_sessions(self) -> list[dict[str, Any]]:
        """Get list of orphaned sessions with project info."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM orphaned_sessions")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_session_recovery_context(self, session_id: str) -> dict[str, Any]:
        """
        Get full recovery context for a crashed/orphaned session.

        Returns all messages, tasks, and relevant state for resumption.
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        # Get all messages from the session
        messages = self.get_messages(session_id, limit=500)

        # Get all tasks (pending, completed, failed)
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT * FROM tasks WHERE session_id = ?
               ORDER BY sequence_num ASC""",
            (session_id,),
        )
        tasks = [Task.from_row(row) for row in cursor.fetchall()]

        # Get pending tasks specifically
        pending_tasks = [
            t
            for t in tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.EXECUTING)
        ]
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in tasks if t.status == TaskStatus.FAILED]

        # Get the last user request
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        last_request = user_messages[-1].content if user_messages else None

        # Get project info
        project = self.get_project_by_id(session.project_id)

        return {
            "session": session,
            "project": project,
            "messages": messages,
            "last_request": last_request,
            "tasks": {
                "pending": pending_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "total": len(tasks),
            },
            "summary": {
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "pending_tasks": len(pending_tasks),
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "started_at": session.started_at,
                "last_heartbeat": session.last_heartbeat_at,
            },
        }

    def recover_session(
        self,
        session_id: str,
        new_session_id: str | None = None,
    ) -> Session | None:
        """
        Mark a crashed session as recovered and optionally link to a new session.

        Args:
            session_id: The crashed session to recover from
            new_session_id: Optional new session that is continuing the work

        Returns:
            The updated (marked as recovered) session, or None if not found
        """
        cursor = self.conn.cursor()

        # Update the crashed session to mark it as recovered
        cursor.execute(
            """UPDATE sessions
               SET status = 'recovered',
                   summary = COALESCE(summary, '') || ' [Recovered]',
                   ended_at = ?
               WHERE id = ?""",
            (now_iso(), session_id),
        )

        if cursor.rowcount == 0:
            return None

        logger.info(f"Marked session {session_id[:8]} as recovered")

        # If we have a new session, record the link
        if new_session_id:
            # Add a system message linking the sessions
            self.add_message(
                new_session_id,
                MessageRole.SYSTEM,
                f"Recovered from crashed session {session_id[:8]}",
                MessageType.CHAT,
            )

        return self.get_session(session_id)

    def get_project_by_id(self, project_id: str) -> Project | None:
        """Get project by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        return Project.from_row(row) if row else None

    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        message_type: MessageType | None = None,
        parent_message_id: str | None = None,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to conversation history."""
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            message_type=message_type,
            parent_message_id=parent_message_id,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO messages
               (id, session_id, role, content, message_type, parent_message_id,
                created_at, tokens_used, cost_usd, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            message.to_row(),
        )

        return message

    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        role: MessageRole | None = None,
    ) -> list[Message]:
        """Get messages for a session."""
        cursor = self.conn.cursor()
        if role:
            cursor.execute(
                """SELECT * FROM messages
                   WHERE session_id = ? AND role = ?
                   ORDER BY created_at ASC LIMIT ?""",
                (session_id, role.value, limit),
            )
        else:
            cursor.execute(
                """SELECT * FROM messages
                   WHERE session_id = ?
                   ORDER BY created_at ASC LIMIT ?""",
                (session_id, limit),
            )
        return [Message.from_row(row) for row in cursor.fetchall()]

    def get_conversation_for_glm(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """
        Get conversation history formatted for GLM API.

        Returns list of {role, content} dicts for OpenAI-compatible API.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT role, content FROM messages
               WHERE session_id = ? AND role IN ('user', 'glm', 'assistant')
               ORDER BY created_at ASC LIMIT ?""",
            (session_id, limit),
        )

        messages = []
        for role, content in cursor.fetchall():
            # Map GLM/assistant to 'assistant' for OpenAI API
            api_role = "assistant" if role in ("glm", "assistant") else role
            messages.append({"role": api_role, "content": content})

        return messages

    # =========================================================================
    # TASK OPERATIONS
    # =========================================================================

    def create_task(
        self,
        session_id: str,
        description: str,
        files: list[str] | None = None,
        constraints: list[str] | None = None,
        success_criteria: str | None = None,
        parent_task_id: str | None = None,
        created_by: str = "glm_decomposition",
        original_request: str | None = None,
        priority: int = 0,
    ) -> Task:
        """Create a new task."""
        # Get next sequence number
        cursor = self.conn.cursor()
        if parent_task_id:
            cursor.execute(
                "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM tasks WHERE parent_task_id = ?",
                (parent_task_id,),
            )
        else:
            cursor.execute(
                "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM tasks WHERE session_id = ? AND parent_task_id IS NULL",
                (session_id,),
            )
        sequence_num = cursor.fetchone()[0]

        task = Task(
            session_id=session_id,
            parent_task_id=parent_task_id,
            sequence_num=sequence_num,
            description=description,
            files=files or [],
            constraints=constraints or [],
            success_criteria=success_criteria,
            priority=priority,
            created_by=created_by,
            original_request=original_request,
        )

        cursor.execute(
            """INSERT INTO tasks
               (id, session_id, parent_task_id, sequence_num, description, files,
                constraints, success_criteria, status, priority, created_at,
                started_at, completed_at, created_by, original_request)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            task.to_row(),
        )

        # Record initial status transition
        self._record_task_transition(task.id, None, TaskStatus.PENDING, "created", created_by)

        return task

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        reason: str | None = None,
        triggered_by: str = "system",
    ) -> Task | None:
        """Update task status with transition tracking. Returns updated task."""
        cursor = self.conn.cursor()

        # Get current status
        cursor.execute("SELECT status FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if not row:
            return None
        from_status = TaskStatus(row[0])

        # Update status
        now = now_iso()
        if status == TaskStatus.EXECUTING:
            cursor.execute(
                "UPDATE tasks SET status = ?, started_at = ? WHERE id = ?",
                (status.value, now, task_id),
            )
        elif status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.SKIPPED,
        ):
            cursor.execute(
                "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
                (status.value, now, task_id),
            )
        else:
            cursor.execute("UPDATE tasks SET status = ? WHERE id = ?", (status.value, task_id))

        # Record transition
        self._record_task_transition(task_id, from_status, status, reason, triggered_by)

        # Return updated task
        return self.get_task(task_id)

    def _record_task_transition(
        self,
        task_id: str,
        from_status: TaskStatus | None,
        to_status: TaskStatus,
        reason: str | None,
        triggered_by: str,
    ) -> None:
        """Record a task status transition."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO task_status_transitions
               (task_id, from_status, to_status, reason, transitioned_at, triggered_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                from_status.value if from_status else None,
                to_status.value,
                reason,
                now_iso(),
                triggered_by,
            ),
        )

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        return Task.from_row(row) if row else None

    def get_pending_tasks(self, session_id: str) -> list[Task]:
        """Get all pending/queued tasks for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT * FROM tasks
               WHERE session_id = ? AND status IN ('pending', 'queued')
               ORDER BY priority DESC, sequence_num ASC""",
            (session_id,),
        )
        return [Task.from_row(row) for row in cursor.fetchall()]

    def get_task_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[Task]:
        """Get completed/failed tasks for GLM context."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT * FROM tasks
               WHERE session_id = ? AND status IN ('completed', 'failed')
               ORDER BY completed_at DESC LIMIT ?""",
            (session_id, limit),
        )
        return [Task.from_row(row) for row in cursor.fetchall()]

    def get_task_context_for_glm(self, session_id: str, limit: int = 10) -> str:
        """Get formatted task history for GLM context injection."""
        tasks = self.get_task_history(session_id, limit)
        if not tasks:
            return ""

        lines = ["## Recent Tasks Executed:"]
        for task in tasks:
            status_icon = "+" if task.status == TaskStatus.COMPLETED else "x"
            lines.append(f"[{status_icon}] {task.description[:80]}")

        lines.append("")
        lines.append("(If user says 'redo', 'retry', or 'the tasks' - they mean these)")
        return "\n".join(lines)

    # =========================================================================
    # ATTEMPT OPERATIONS
    # =========================================================================

    def create_attempt(
        self,
        task_id: str,
        prompt: str,
        timeout_tier: str = "code",
        allowed_tools: list[str] | None = None,
    ) -> TaskAttempt:
        """Create a new task attempt."""
        cursor = self.conn.cursor()

        # Get next attempt number
        cursor.execute(
            "SELECT COALESCE(MAX(attempt_num), 0) + 1 FROM task_attempts WHERE task_id = ?",
            (task_id,),
        )
        attempt_num = cursor.fetchone()[0]

        attempt = TaskAttempt(
            task_id=task_id,
            attempt_num=attempt_num,
            prompt=prompt,
            timeout_tier=timeout_tier,
            allowed_tools=allowed_tools or [],
        )

        cursor.execute(
            """INSERT INTO task_attempts
               (id, task_id, attempt_num, prompt, timeout_tier, allowed_tools,
                result, response_text, error_message, summary, started_at,
                completed_at, duration_ms, cost_usd, tokens_used, num_turns,
                claude_session_id, tool_calls)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            attempt.to_row(),
        )

        return attempt

    def complete_attempt(
        self,
        attempt_id: str,
        result: AttemptResult,
        response_text: str | None = None,
        error_message: str | None = None,
        summary: str | None = None,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        tokens_used: int = 0,
        num_turns: int = 0,
        claude_session_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> TaskAttempt | None:
        """Complete a task attempt with results. Returns updated attempt."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE task_attempts
               SET result = ?, response_text = ?, error_message = ?, summary = ?,
                   completed_at = ?, duration_ms = ?, cost_usd = ?, tokens_used = ?,
                   num_turns = ?, claude_session_id = ?, tool_calls = ?
               WHERE id = ?""",
            (
                result.value,
                response_text,
                error_message,
                summary,
                now_iso(),
                duration_ms,
                cost_usd,
                tokens_used,
                num_turns,
                claude_session_id,
                to_json(tool_calls) if tool_calls else None,
                attempt_id,
            ),
        )
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id: str) -> TaskAttempt | None:
        """Get attempt by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM task_attempts WHERE id = ?", (attempt_id,))
        row = cursor.fetchone()
        return TaskAttempt.from_row(row) if row else None

    def get_task_attempts(self, task_id: str) -> list[TaskAttempt]:
        """Get all attempts for a task."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM task_attempts WHERE task_id = ? ORDER BY attempt_num ASC", (task_id,)
        )
        return [TaskAttempt.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # FILE CHANGE OPERATIONS
    # =========================================================================

    def record_file_change(
        self,
        attempt_id: str,
        file_path: str,
        change_type: str,
        old_path: str | None = None,
        diff_content: str | None = None,
        lines_added: int | None = None,
        lines_removed: int | None = None,
    ) -> FileChange:
        """Record a file change from an attempt."""
        from vibe.persistence.models import ChangeType

        change = FileChange(
            attempt_id=attempt_id,
            file_path=file_path,
            change_type=ChangeType(change_type),
            old_path=old_path,
            diff_content=diff_content,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO file_changes
               (attempt_id, file_path, change_type, old_path, diff_content,
                lines_added, lines_removed, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            change.to_row(),
        )

        return change

    def get_file_changes(self, attempt_id: str) -> list[FileChange]:
        """Get file changes for an attempt."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM file_changes WHERE attempt_id = ? ORDER BY created_at ASC", (attempt_id,)
        )
        return [FileChange.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # REVIEW OPERATIONS
    # =========================================================================

    def create_review(
        self,
        attempt_id: str,
        approved: bool,
        feedback: str,
        issues: list[str] | None = None,
        suggested_next_steps: list[str] | None = None,
        review_duration_ms: int | None = None,
        tokens_used: int | None = None,
    ) -> Review:
        """Create a review for an attempt."""
        review = Review(
            attempt_id=attempt_id,
            approved=approved,
            issues=issues or [],
            feedback=feedback,
            suggested_next_steps=suggested_next_steps or [],
            review_duration_ms=review_duration_ms,
            tokens_used=tokens_used,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO reviews
               (id, attempt_id, approved, issues, feedback, suggested_next_steps,
                reviewed_at, review_duration_ms, tokens_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            review.to_row(),
        )

        return review

    def get_review(self, attempt_id: str) -> Review | None:
        """Get review for an attempt."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM reviews WHERE attempt_id = ?", (attempt_id,))
        row = cursor.fetchone()
        return Review.from_row(row) if row else None

    # =========================================================================
    # DEBUG SESSION OPERATIONS
    # =========================================================================

    def create_debug_session(
        self,
        session_id: str,
        problem: str,
        hypothesis: str | None = None,
        must_preserve: list[str] | None = None,
        initial_git_commit: str | None = None,
    ) -> DebugSession:
        """Create a new debug session."""
        debug_session = DebugSession(
            session_id=session_id,
            problem=problem,
            hypothesis=hypothesis,
            must_preserve=must_preserve or [],
            initial_git_commit=initial_git_commit,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO debug_sessions
               (id, session_id, problem, hypothesis, must_preserve, is_active,
                is_solved, initial_git_commit, created_at, updated_at, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            debug_session.to_row(),
        )

        return debug_session

    def get_active_debug_session(self, session_id: str) -> DebugSession | None:
        """Get the active debug session for a session (if any)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM debug_sessions WHERE session_id = ? AND is_active = 1 LIMIT 1",
            (session_id,),
        )
        row = cursor.fetchone()
        return DebugSession.from_row(row) if row else None

    def update_debug_session(
        self,
        debug_session_id: str,
        hypothesis: str | None = None,
        is_active: bool | None = None,
        is_solved: bool | None = None,
    ) -> None:
        """Update debug session fields."""
        cursor = self.conn.cursor()
        updates = ["updated_at = ?"]
        values = [now_iso()]

        if hypothesis is not None:
            updates.append("hypothesis = ?")
            values.append(hypothesis)
        if is_active is not None:
            updates.append("is_active = ?")
            values.append(1 if is_active else 0)
        if is_solved is not None:
            updates.append("is_solved = ?")
            values.append(1 if is_solved else 0)
            if is_solved:
                updates.append("resolved_at = ?")
                values.append(now_iso())

        values.append(debug_session_id)
        cursor.execute(f"UPDATE debug_sessions SET {', '.join(updates)} WHERE id = ?", values)

    def create_debug_iteration(
        self,
        debug_session_id: str,
        task_description: str,
        starting_points: list[str] | None = None,
        what_to_look_for: str | None = None,
        success_criteria: str | None = None,
        git_checkpoint: str | None = None,
    ) -> DebugIteration:
        """Create a new debug iteration."""
        cursor = self.conn.cursor()

        # Get next iteration number
        cursor.execute(
            "SELECT COALESCE(MAX(iteration_num), 0) + 1 FROM debug_iterations WHERE debug_session_id = ?",
            (debug_session_id,),
        )
        iteration_num = cursor.fetchone()[0]

        iteration = DebugIteration(
            debug_session_id=debug_session_id,
            iteration_num=iteration_num,
            task_description=task_description,
            starting_points=starting_points or [],
            what_to_look_for=what_to_look_for,
            success_criteria=success_criteria,
            git_checkpoint=git_checkpoint,
        )

        cursor.execute(
            """INSERT INTO debug_iterations
               (id, debug_session_id, iteration_num, task_description, starting_points,
                what_to_look_for, success_criteria, output, files_examined, files_changed,
                structured_findings, review_approved, review_is_solved, review_feedback,
                review_next_task, started_at, completed_at, duration_ms, git_checkpoint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            iteration.to_row(),
        )

        return iteration

    def complete_debug_iteration(
        self,
        iteration_id: str,
        output: str,
        files_examined: list[str] | None = None,
        files_changed: list[str] | None = None,
        structured_findings: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Complete a debug iteration with Claude's output."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE debug_iterations
               SET output = ?, files_examined = ?, files_changed = ?,
                   structured_findings = ?, completed_at = ?, duration_ms = ?
               WHERE id = ?""",
            (
                output,
                to_json(files_examined),
                to_json(files_changed),
                to_json(structured_findings),
                now_iso(),
                duration_ms,
                iteration_id,
            ),
        )

    def review_debug_iteration(
        self,
        iteration_id: str,
        approved: bool,
        is_solved: bool,
        feedback: str,
        next_task: str | None = None,
    ) -> None:
        """Add GLM review to a debug iteration."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE debug_iterations
               SET review_approved = ?, review_is_solved = ?,
                   review_feedback = ?, review_next_task = ?
               WHERE id = ?""",
            (
                1 if approved else 0,
                1 if is_solved else 0,
                feedback,
                next_task,
                iteration_id,
            ),
        )

    def get_debug_iterations(self, debug_session_id: str) -> list[DebugIteration]:
        """Get all iterations for a debug session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM debug_iterations WHERE debug_session_id = ? ORDER BY iteration_num ASC",
            (debug_session_id,),
        )
        return [DebugIteration.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # CONTEXT OPERATIONS
    # =========================================================================

    def save_context(
        self,
        project_id: str,
        key: str,
        value: str,
        category: ContextCategory = ContextCategory.NOTE,
        priority: Priority = Priority.NORMAL,
        session_id: str | None = None,
    ) -> ContextItem:
        """Save or update a context item."""
        cursor = self.conn.cursor()

        # Check if exists
        cursor.execute(
            "SELECT id FROM context_items WHERE project_id = ? AND key = ?", (project_id, key)
        )
        existing = cursor.fetchone()

        item = ContextItem(
            id=existing[0] if existing else generate_id(),
            project_id=project_id,
            session_id=session_id,
            key=key,
            value=value,
            category=category,
            priority=priority,
        )

        if existing:
            cursor.execute(
                """UPDATE context_items
                   SET value = ?, category = ?, priority = ?, updated_at = ?
                   WHERE id = ?""",
                (value, category.value, priority.value, now_iso(), item.id),
            )
        else:
            cursor.execute(
                """INSERT INTO context_items
                   (id, project_id, session_id, key, value, category, priority,
                    created_at, updated_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                item.to_row(),
            )

        return item

    def get_context(self, project_id: str, key: str) -> ContextItem | None:
        """Get a context item by key."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM context_items WHERE project_id = ? AND key = ?", (project_id, key)
        )
        row = cursor.fetchone()
        return ContextItem.from_row(row) if row else None

    def list_context(
        self,
        project_id: str,
        category: ContextCategory | None = None,
        limit: int = 50,
    ) -> list[ContextItem]:
        """List context items for a project."""
        cursor = self.conn.cursor()
        if category:
            cursor.execute(
                """SELECT * FROM context_items
                   WHERE project_id = ? AND category = ?
                   ORDER BY priority DESC, updated_at DESC LIMIT ?""",
                (project_id, category.value, limit),
            )
        else:
            cursor.execute(
                """SELECT * FROM context_items
                   WHERE project_id = ?
                   ORDER BY priority DESC, updated_at DESC LIMIT ?""",
                (project_id, limit),
            )
        return [ContextItem.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # CONVENTION OPERATIONS
    # =========================================================================

    def save_convention(
        self,
        key: str,
        convention: str,
        applies_to: str = "all",
        created_by_project: str | None = None,
    ) -> Convention:
        """Save or update a global convention."""
        cursor = self.conn.cursor()

        # Check if exists
        cursor.execute("SELECT id FROM conventions WHERE key = ?", (key,))
        existing = cursor.fetchone()

        conv = Convention(
            id=existing[0] if existing else generate_id(),
            key=key,
            convention=convention,
            applies_to=applies_to,
            created_by_project=created_by_project,
        )

        if existing:
            cursor.execute(
                """UPDATE conventions
                   SET convention = ?, applies_to = ?, updated_at = ?
                   WHERE key = ?""",
                (convention, applies_to, now_iso(), key),
            )
        else:
            cursor.execute(
                """INSERT INTO conventions
                   (id, key, convention, applies_to, created_by_project,
                    created_at, updated_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                conv.to_row(),
            )

        return conv

    def list_conventions(self, applies_to: str = "all") -> list[Convention]:
        """List active conventions."""
        cursor = self.conn.cursor()
        if applies_to == "all":
            cursor.execute("SELECT * FROM conventions WHERE is_active = 1 ORDER BY created_at DESC")
        else:
            cursor.execute(
                """SELECT * FROM conventions
                   WHERE is_active = 1 AND (applies_to = 'all' OR applies_to = ?)
                   ORDER BY created_at DESC""",
                (applies_to,),
            )
        return [Convention.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # CHECKPOINT OPERATIONS
    # =========================================================================

    def create_checkpoint(
        self,
        session_id: str,
        name: str,
        description: str | None = None,
        git_branch: str | None = None,
        git_commit: str | None = None,
        git_status: str | None = None,
    ) -> Checkpoint:
        """Create a recovery checkpoint."""
        checkpoint = Checkpoint(
            session_id=session_id,
            name=name,
            description=description,
            git_branch=git_branch,
            git_commit=git_commit,
            git_status=git_status,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO checkpoints
               (id, session_id, name, description, git_branch, git_commit,
                git_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            checkpoint.to_row(),
        )

        return checkpoint

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """List checkpoints for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
        )
        return [Checkpoint.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # TOOL USAGE OPERATIONS
    # =========================================================================

    def record_tool_usage(
        self,
        session_id: str,
        tool_name: str,
        success: bool,
        duration_ms: int = 0,
    ) -> None:
        """Record a tool invocation."""
        cursor = self.conn.cursor()

        # Upsert tool usage record
        cursor.execute(
            """INSERT INTO tool_usage (session_id, tool_name, invocation_count,
                   success_count, failure_count, total_duration_ms, last_used_at)
               VALUES (?, ?, 1, ?, ?, ?, ?)
               ON CONFLICT(session_id, tool_name) DO UPDATE SET
                   invocation_count = invocation_count + 1,
                   success_count = success_count + ?,
                   failure_count = failure_count + ?,
                   total_duration_ms = total_duration_ms + ?,
                   last_used_at = ?""",
            (
                session_id,
                tool_name,
                1 if success else 0,
                0 if success else 1,
                duration_ms,
                now_iso(),
                1 if success else 0,
                0 if success else 1,
                duration_ms,
                now_iso(),
            ),
        )

    def get_tool_usage(self, session_id: str) -> list[ToolUsage]:
        """Get tool usage statistics for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM tool_usage WHERE session_id = ? ORDER BY invocation_count DESC",
            (session_id,),
        )
        return [ToolUsage.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # REQUEST OPERATIONS
    # =========================================================================

    def create_request(self, session_id: str, request_text: str) -> Request:
        """Create a new user request."""
        request = Request(
            session_id=session_id,
            request_text=request_text,
        )

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO requests
               (id, session_id, request_text, result_summary, tasks_created,
                status, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            request.to_row(),
        )

        return request

    def complete_request(
        self,
        request_id: str,
        status: str,
        result_summary: str,
        tasks_created: int = 0,
    ) -> None:
        """Complete a user request."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE requests
               SET status = ?, result_summary = ?, tasks_created = ?, completed_at = ?
               WHERE id = ?""",
            (status, result_summary, tasks_created, now_iso(), request_id),
        )

    def get_recent_requests(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[Request]:
        """Get recent requests for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM requests WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        return [Request.from_row(row) for row in cursor.fetchall()]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get comprehensive statistics for a session."""
        cursor = self.conn.cursor()

        # Task counts
        cursor.execute(
            """SELECT status, COUNT(*) FROM tasks WHERE session_id = ? GROUP BY status""",
            (session_id,),
        )
        task_counts = dict(cursor.fetchall())

        # Message count
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        message_count = cursor.fetchone()[0]

        # Total cost
        cursor.execute(
            "SELECT SUM(cost_usd) FROM task_attempts WHERE task_id IN (SELECT id FROM tasks WHERE session_id = ?)",
            (session_id,),
        )
        total_cost = cursor.fetchone()[0] or 0.0

        # Request count
        cursor.execute("SELECT COUNT(*) FROM requests WHERE session_id = ?", (session_id,))
        request_count = cursor.fetchone()[0]

        return {
            "task_counts": task_counts,
            "message_count": message_count,
            "total_cost_usd": total_cost,
            "request_count": request_count,
        }
