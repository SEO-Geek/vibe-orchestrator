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
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vibe.config import MEMORY_DB_PATH
from vibe.exceptions import MemoryConnectionError, MemoryNotFoundError

logger = logging.getLogger(__name__)


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

    def __init__(self, project_name: str):
        """
        Initialize memory connection.

        Args:
            project_name: Project name (used as channel for isolation)
        """
        self.project_name = project_name
        self.session_id: str | None = None
        self._db_path = MEMORY_DB_PATH

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self._db_path.exists():
            raise MemoryConnectionError(
                f"Memory-keeper database not found at {self._db_path}"
            )

        return sqlite3.connect(self._db_path)

    def start_session(self, description: str = "") -> str:
        """
        Start a new Vibe session.

        Args:
            description: Optional session description

        Returns:
            New session ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        self.session_id = str(uuid.uuid4())

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

        conn.commit()
        conn.close()
        return self.session_id

    def save(
        self,
        key: str,
        value: str,
        category: str = "progress",
        priority: str = "normal",
    ) -> None:
        """
        Save a context item.

        Args:
            key: Unique key for the item
            value: Content to save
            category: One of: task, decision, progress, note, error, warning
            priority: One of: high, normal, low
        """
        if not self.session_id:
            raise MemoryConnectionError("No active session - call start_session first")

        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO context_items
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

        conn.commit()
        conn.close()

    def load_project_context(self, limit: int = 50) -> list[ContextItem]:
        """
        Load all context for this project.

        Args:
            limit: Maximum items to return

        Returns:
            List of ContextItem objects
        """
        conn = self._get_connection()
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

        conn.close()
        return results

    def get(self, key: str) -> ContextItem | None:
        """
        Get a specific context item by key.

        Args:
            key: Item key

        Returns:
            ContextItem or None if not found
        """
        conn = self._get_connection()
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
        conn.close()

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

        conn = self._get_connection()
        cursor = conn.cursor()

        checkpoint_id = str(uuid.uuid4())

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

        conn.commit()
        conn.close()
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
        conn = self._get_connection()
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

        conn.close()
        return results

    def continue_session(self, session_id: str) -> bool:
        """
        Continue an existing session.

        Args:
            session_id: Session ID to continue

        Returns:
            True if session found and activated
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        )

        if cursor.fetchone():
            self.session_id = session_id
            conn.close()
            return True

        conn.close()
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
        conn = self._get_connection()
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

        conn.close()
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

        conn = self._get_connection()
        cursor = conn.cursor()

        entry_id = str(uuid.uuid4())
        tags_json = json.dumps(tags or [])

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

        conn.commit()
        conn.close()
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

        conn = self._get_connection()
        cursor = conn.cursor()

        checkpoint_id = str(uuid.uuid4())

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

        conn.commit()
        conn.close()

        logger.info(f"Created checkpoint '{name}' with ID {checkpoint_id[:8]}")
        return checkpoint_id

    def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics for this project.

        Returns:
            Dict with counts and recent activity
        """
        conn = self._get_connection()
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

        conn.close()

        return {
            "total_items": total_items,
            "by_category": by_category,
            "session_count": session_count,
            "checkpoint_count": checkpoint_count,
            "current_session": self.session_id[:8] if self.session_id else None,
        }
