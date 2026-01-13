"""
Memory-keeper Integration - Direct SQLite Access

Provides session and context management via direct SQLite access
to the memory-keeper database.

Features:
- Session management (start, end, list recent)
- Context items (save, load, search)
- Checkpoints with git status capture
- Journal entries for notes
"""

import json
import logging
import sqlite3
import subprocess
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from vibe.config import MEMORY_DB_PATH
from vibe.exceptions import MemoryConnectionError, MemoryNotFoundError

logger = logging.getLogger(__name__)

# Global channel for cross-project conventions
GLOBAL_CHANNEL = "_vibe_global"


# =============================================================================
# CONNECTION POOL
# Thread-local connection pool for better performance
# =============================================================================

class ConnectionPool:
    """
    Thread-local SQLite connection pool.

    Each thread gets its own connection to avoid SQLite threading issues.
    Connections are reused within the same thread for better performance.
    """

    def __init__(self, db_path: Path, max_idle_time: float = 300.0):
        """
        Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            max_idle_time: Close connections idle longer than this (seconds)
        """
        self.db_path = db_path
        self.max_idle_time = max_idle_time
        self._local = threading.local()
        self._lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """
        Get or create a connection for the current thread.

        Returns:
            SQLite connection for current thread
        """
        # Check if current thread has a connection
        conn = getattr(self._local, 'connection', None)

        # Check if connection exists and is still valid
        if conn is not None:
            try:
                # Test connection is still alive
                conn.execute("SELECT 1")
                self._local.last_used = datetime.now()
                return conn
            except sqlite3.Error:
                # Connection is dead, close it
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.connection = None

        # Create new connection for this thread
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,  # We manage thread safety ourselves
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA foreign_keys=ON")

        self._local.connection = conn
        self._local.last_used = datetime.now()

        return conn

    def close_all(self) -> None:
        """Close all connections (call at shutdown)."""
        conn = getattr(self._local, 'connection', None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.connection = None


# Module-level connection pool (lazy initialization)
_connection_pool: ConnectionPool | None = None


def get_connection_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool(MEMORY_DB_PATH)
    return _connection_pool


@dataclass
class ContextItem:
    """A saved context item from memory-keeper."""

    key: str
    value: str
    category: str
    priority: str
    created_at: datetime
    channel: str
    id: str = ""


@dataclass
class SessionInfo:
    """Information about a Vibe session."""

    id: str
    name: str
    description: str
    channel: str
    created_at: datetime
    item_count: int = 0


class VibeMemory:
    """
    Direct SQLite access to memory-keeper data.

    Uses channel = project_name for isolation between projects.
    """

    def __init__(self, project_name: str, use_pool: bool = True):
        """
        Initialize memory connection.

        Args:
            project_name: Project name (used as channel for isolation)
            use_pool: Use connection pooling for better performance (default: True)
        """
        self.project_name = project_name
        self.session_id: str | None = None
        self._db_path = MEMORY_DB_PATH
        self._use_pool = use_pool

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection.

        Uses connection pooling if enabled for better performance.
        Pool connections are thread-local and reused.
        """
        if not self._db_path.exists():
            raise MemoryConnectionError(
                f"Memory-keeper database not found at {self._db_path}"
            )

        if self._use_pool:
            # Use global connection pool for thread-local reuse
            return get_connection_pool().get_connection()

        # Fallback to creating a new connection each time
        return sqlite3.connect(self._db_path)

    @contextmanager
    def _db_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        When using pool: connection is reused, only commits/rollbacks
        When not using pool: connection is created and closed each time
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            # Only close if not using pool (pool connections are reused)
            if not self._use_pool:
                conn.close()

    def start_session(self, description: str = "") -> str:
        """
        Start a new Vibe session.

        Args:
            description: Optional session description

        Returns:
            New session ID
        """
        self.session_id = str(uuid.uuid4())

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (id, name, description, default_channel, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    f"vibe-{self.project_name}",
                    description,
                    self.project_name,
                    datetime.now().isoformat(),
                ),
            )

        return self.session_id

    def save(
        self,
        key: str,
        value: str,
        category: str = "progress",
        priority: str = "normal",
    ) -> None:
        """
        Save a context item (upsert semantics).

        If an item with the same key and channel already exists, it will be
        updated with the new value while preserving the original id and created_at.
        Otherwise, a new item is created.

        Args:
            key: Unique key for the item
            value: Content to save
            category: One of: task, decision, progress, note, error, warning
            priority: One of: high, normal, low
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session - call start_session first")

        now = datetime.now().isoformat()

        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Check if item with same key+channel exists (proper upsert logic)
            cursor.execute(
                """
                SELECT id, created_at FROM context_items
                WHERE key = ? AND channel = ?
                """,
                (key, self.project_name),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing item, preserve original id and created_at
                cursor.execute(
                    """
                    UPDATE context_items
                    SET value = ?, category = ?, priority = ?,
                        session_id = ?, updated_at = ?
                    WHERE key = ? AND channel = ?
                    """,
                    (value, category, priority, self.session_id, now, key, self.project_name),
                )
                logger.debug(f"Updated existing context item: {key}")
            else:
                # Insert new item
                cursor.execute(
                    """
                    INSERT INTO context_items
                    (id, session_id, key, value, category, priority, channel, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        self.session_id,
                        key,
                        value,
                        category,
                        priority,
                        self.project_name,
                        now,
                        now,
                    ),
                )
                logger.debug(f"Created new context item: {key}")

    def load_project_context(self, limit: int = 50) -> list[ContextItem]:
        """
        Load all context for this project.

        Args:
            limit: Maximum items to return

        Returns:
            List of ContextItem objects
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value, category, priority, created_at, channel
                FROM context_items
                WHERE channel = ?
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
                """,
                (self.project_name, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    ContextItem(
                        key=row[0],
                        value=row[1],
                        category=row[2],
                        priority=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        channel=row[5],
                    )
                )

        return results

    def get(self, key: str) -> ContextItem | None:
        """
        Get a specific context item by key.

        Args:
            key: Item key

        Returns:
            ContextItem or None if not found
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value, category, priority, created_at, channel
                FROM context_items
                WHERE key = ? AND channel = ?
                """,
                (key, self.project_name),
            )

            row = cursor.fetchone()

        if row:
            return ContextItem(
                key=row[0],
                value=row[1],
                category=row[2],
                priority=row[3],
                created_at=datetime.fromisoformat(row[4]),
                channel=row[5],
            )
        return None

    def create_checkpoint(self, name: str, description: str = "") -> str:
        """
        Create a named checkpoint for recovery.

        Args:
            name: Checkpoint name
            description: Optional description

        Returns:
            Checkpoint ID
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session")

        checkpoint_id = str(uuid.uuid4())

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints (id, session_id, name, description, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    self.session_id,
                    name,
                    description,
                    datetime.now().isoformat(),
                ),
            )

        return checkpoint_id

    def end_session(self, summary: str = "") -> None:
        """
        End the current session with a summary.

        Args:
            summary: Session summary to save
        """
        if self.session_id and summary:
            self.save(
                key=f"session-summary-{self.session_id[:8]}",
                value=summary,
                category="note",
                priority="high",
            )

        self.session_id = None

    def list_recent_sessions(self, limit: int = 10) -> list[SessionInfo]:
        """
        List recent sessions for this project.

        Args:
            limit: Maximum sessions to return

        Returns:
            List of SessionInfo objects
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.id, s.name, s.description, s.default_channel, s.created_at,
                       (SELECT COUNT(*) FROM context_items ci WHERE ci.session_id = s.id) as item_count
                FROM sessions s
                WHERE s.default_channel = ?
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (self.project_name, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    SessionInfo(
                        id=row[0],
                        name=row[1] or "",
                        description=row[2] or "",
                        channel=row[3] or "",
                        created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        item_count=row[5] or 0,
                    )
                )

        return results

    def continue_session(self, session_id: str) -> bool:
        """
        Continue an existing session.

        Args:
            session_id: Session ID to continue

        Returns:
            True if session found and activated
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM sessions WHERE id = ?",
                (session_id,),
            )
            found = cursor.fetchone() is not None

        if found:
            self.session_id = session_id
            return True
        return False

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 20,
    ) -> list[ContextItem]:
        """
        Search context items by text.

        Args:
            query: Search query (matches key or value)
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of matching ContextItem objects
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()

            if category:
                cursor.execute(
                    """
                    SELECT id, key, value, category, priority, created_at, channel
                    FROM context_items
                    WHERE channel = ?
                      AND category = ?
                      AND (key LIKE ? OR value LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self.project_name, category, f"%{query}%", f"%{query}%", limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, key, value, category, priority, created_at, channel
                    FROM context_items
                    WHERE channel = ?
                      AND (key LIKE ? OR value LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self.project_name, f"%{query}%", f"%{query}%", limit),
                )

            results = []
            for row in cursor.fetchall():
                results.append(
                    ContextItem(
                        id=row[0],
                        key=row[1],
                        value=row[2],
                        category=row[3],
                        priority=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                        channel=row[6],
                    )
                )

        return results

    def save_decision(self, key: str, decision: str, reasoning: str = "") -> None:
        """
        Save a decision with optional reasoning.

        Args:
            key: Decision identifier
            decision: The decision made
            reasoning: Why this decision was made
        """
        value = decision
        if reasoning:
            value = f"{decision}\n\nReasoning: {reasoning}"

        self.save(key=key, value=value, category="decision", priority="high")

    def save_task_result(
        self,
        task_description: str,
        success: bool,
        summary: str,
        files_changed: list[str] | None = None,
    ) -> None:
        """
        Save the result of a completed task.

        Args:
            task_description: Original task description
            success: Whether task succeeded
            summary: Result summary
            files_changed: List of modified files
        """
        status = "completed" if success else "failed"
        value = f"Task: {task_description}\nStatus: {status}\nSummary: {summary}"
        if files_changed:
            value += f"\nFiles: {', '.join(files_changed)}"

        key = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.save(
            key=key,
            value=value,
            category="task" if success else "error",
            priority="normal",
        )

    def add_journal_entry(
        self,
        entry: str,
        mood: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a journal entry.

        Args:
            entry: Journal entry text
            mood: Optional mood indicator
            tags: Optional tags

        Returns:
            Entry ID
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session")

        entry_id = str(uuid.uuid4())
        tags_json = json.dumps(tags or [])

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO journal_entries (id, session_id, entry, mood, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    self.session_id,
                    entry,
                    mood,
                    tags_json,
                    datetime.now().isoformat(),
                ),
            )

        return entry_id

    def create_checkpoint_with_git(
        self,
        name: str,
        description: str = "",
        project_path: str | None = None,
    ) -> str:
        """
        Create a checkpoint capturing current git status.

        Args:
            name: Checkpoint name
            description: Optional description
            project_path: Path to git repo (optional)

        Returns:
            Checkpoint ID
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session")

        # Capture git status if project_path provided
        git_status = ""
        git_branch = ""
        if project_path:
            try:
                # Get branch
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                git_branch = result.stdout.strip()

                # Get status
                result = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                git_status = result.stdout.strip()
            except Exception as e:
                logger.warning(f"Could not capture git status: {e}")

        checkpoint_id = str(uuid.uuid4())

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints
                (id, session_id, name, description, git_status, git_branch, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    self.session_id,
                    name,
                    description,
                    git_status,
                    git_branch,
                    datetime.now().isoformat(),
                ),
            )

            # Link current context items to checkpoint
            cursor.execute(
                """
                INSERT INTO checkpoint_items (id, checkpoint_id, context_item_id)
                SELECT ?, ?, id FROM context_items WHERE session_id = ?
                """,
                (str(uuid.uuid4()), checkpoint_id, self.session_id),
            )

        logger.info(f"Created checkpoint '{name}' with ID {checkpoint_id[:8]}")
        return checkpoint_id

    def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics for this project.

        Returns:
            Dict with counts and recent activity
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Count items by category
            cursor.execute(
                """
                SELECT category, COUNT(*) as count
                FROM context_items
                WHERE channel = ?
                GROUP BY category
                """,
                (self.project_name,),
            )
            by_category = {row[0]: row[1] for row in cursor.fetchall()}

            # Total items
            cursor.execute(
                "SELECT COUNT(*) FROM context_items WHERE channel = ?",
                (self.project_name,),
            )
            total_items = cursor.fetchone()[0]

            # Session count
            cursor.execute(
                "SELECT COUNT(*) FROM sessions WHERE default_channel = ?",
                (self.project_name,),
            )
            session_count = cursor.fetchone()[0]

            # Checkpoint count
            cursor.execute(
                """
                SELECT COUNT(*) FROM checkpoints c
                JOIN sessions s ON c.session_id = s.id
                WHERE s.default_channel = ?
                """,
                (self.project_name,),
            )
            checkpoint_count = cursor.fetchone()[0]

        return {
            "total_items": total_items,
            "by_category": by_category,
            "session_count": session_count,
            "checkpoint_count": checkpoint_count,
            "current_session": self.session_id[:8] if self.session_id else None,
        }

    # Global Conventions Methods (cross-project)

    def save_convention(
        self,
        key: str,
        convention: str,
        applies_to: str = "all",
    ) -> None:
        """
        Save a global convention that applies across projects.

        Uses proper upsert logic to preserve original id and created_at
        when updating an existing convention.

        Args:
            key: Convention identifier (e.g., "browser-testing", "code-style")
            convention: The convention text (e.g., "Always use Playwright for browser testing")
            applies_to: Project type this applies to ("all", "python", "javascript", etc.)
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session")

        now = datetime.now().isoformat()
        full_key = f"convention:{key}"
        value = json.dumps({
            "convention": convention,
            "applies_to": applies_to,
            "created_by": self.project_name,
        })

        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Check if convention exists (proper upsert logic)
            cursor.execute(
                """
                SELECT id FROM context_items
                WHERE key = ? AND channel = ?
                """,
                (full_key, GLOBAL_CHANNEL),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing convention, preserve id and created_at
                cursor.execute(
                    """
                    UPDATE context_items
                    SET value = ?, session_id = ?, updated_at = ?
                    WHERE key = ? AND channel = ?
                    """,
                    (value, self.session_id, now, full_key, GLOBAL_CHANNEL),
                )
            else:
                # Insert new convention
                cursor.execute(
                    """
                    INSERT INTO context_items
                    (id, session_id, key, value, category, priority, channel, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        self.session_id,
                        full_key,
                        value,
                        "decision",
                        "high",
                        GLOBAL_CHANNEL,
                        now,
                        now,
                    ),
                )

        logger.info(f"Saved global convention: {key}")

    def load_conventions(self, applies_to: str = "all") -> list[str]:
        """
        Load global conventions.

        Args:
            applies_to: Filter by project type ("all" returns everything)

        Returns:
            List of convention strings
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value
                FROM context_items
                WHERE channel = ?
                  AND key LIKE 'convention:%'
                ORDER BY created_at DESC
                """,
                (GLOBAL_CHANNEL,),
            )

            conventions = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    # Filter by applies_to
                    if applies_to == "all" or data.get("applies_to") in ("all", applies_to):
                        conventions.append(data.get("convention", ""))
                except json.JSONDecodeError:
                    # Fallback for plain text conventions
                    conventions.append(row[1])

        return conventions

    def list_conventions(self) -> list[dict[str, Any]]:
        """
        List all global conventions with metadata.

        Returns:
            List of convention dicts with key, convention, applies_to
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value, created_at
                FROM context_items
                WHERE channel = ?
                  AND key LIKE 'convention:%'
                ORDER BY created_at DESC
                """,
                (GLOBAL_CHANNEL,),
            )

            conventions = []
            for row in cursor.fetchall():
                key = row[0].replace("convention:", "")
                try:
                    data = json.loads(row[1])
                    conventions.append({
                        "key": key,
                        "convention": data.get("convention", ""),
                        "applies_to": data.get("applies_to", "all"),
                        "created_by": data.get("created_by", "unknown"),
                        "created_at": row[2],
                    })
                except json.JSONDecodeError:
                    conventions.append({
                        "key": key,
                        "convention": row[1],
                        "applies_to": "all",
                        "created_by": "unknown",
                        "created_at": row[2],
                    })

        return conventions

    def delete_convention(self, key: str) -> bool:
        """
        Delete a global convention.

        Args:
            key: Convention key to delete

        Returns:
            True if deleted, False if not found
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM context_items
                WHERE channel = ?
                  AND key = ?
                """,
                (GLOBAL_CHANNEL, f"convention:{key}"),
            )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted global convention: {key}")
        return deleted

    # Debug Session Methods

    def save_debug_session(self, session_data: dict[str, Any]) -> None:
        """
        Save a debug session to memory.

        Uses proper upsert logic to preserve original id and created_at
        when updating an existing debug session.

        Args:
            session_data: Serialized debug session dict
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session")

        now = datetime.now().isoformat()
        key = f"debug-session:{session_data.get('problem', 'unknown')[:50]}"
        value = json.dumps(session_data)

        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Check if debug session exists (proper upsert logic)
            cursor.execute(
                """
                SELECT id FROM context_items
                WHERE key = ? AND channel = ?
                """,
                (key, self.project_name),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing session, preserve id and created_at
                cursor.execute(
                    """
                    UPDATE context_items
                    SET value = ?, session_id = ?, updated_at = ?
                    WHERE key = ? AND channel = ?
                    """,
                    (value, self.session_id, now, key, self.project_name),
                )
            else:
                # Insert new session
                cursor.execute(
                    """
                    INSERT INTO context_items
                    (id, session_id, key, value, category, priority, channel, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        self.session_id,
                        key,
                        value,
                        "progress",
                        "high",
                        self.project_name,
                        now,
                        now,
                    ),
                )

        logger.info(f"Saved debug session: {key}")

    def load_debug_session(self) -> dict[str, Any] | None:
        """
        Load the most recent active debug session for this project.

        Returns:
            Debug session dict or None
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT value
                FROM context_items
                WHERE channel = ?
                  AND key LIKE 'debug-session:%'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (self.project_name,),
            )
            row = cursor.fetchone()

        if row:
            try:
                data = json.loads(row[0])
                # Only return if still active
                if data.get("is_active", False):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def list_debug_sessions(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        """
        List all debug sessions for this project.

        Args:
            include_inactive: Include completed/inactive sessions

        Returns:
            List of debug session summaries
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value, updated_at
                FROM context_items
                WHERE channel = ?
                  AND key LIKE 'debug-session:%'
                ORDER BY updated_at DESC
                """,
                (self.project_name,),
            )

            sessions = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    if include_inactive or data.get("is_active", False):
                        sessions.append({
                            "key": row[0],
                            "problem": data.get("problem", "Unknown"),
                            "attempts": len(data.get("attempts", [])),
                            "is_active": data.get("is_active", False),
                            "updated_at": row[2],
                        })
                except json.JSONDecodeError:
                    pass

        return sessions

    def delete_debug_session(self, key: str) -> bool:
        """
        Delete a debug session.

        Args:
            key: Debug session key to delete

        Returns:
            True if deleted
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM context_items
                WHERE channel = ?
                  AND key = ?
                """,
                (self.project_name, key),
            )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted debug session: {key}")
        return deleted
