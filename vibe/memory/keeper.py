"""
Memory-keeper Integration - Direct SQLite Access

Provides session and context management via direct SQLite access
to the memory-keeper database.

Placeholder for Phase 6 implementation.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vibe.config import MEMORY_DB_PATH
from vibe.exceptions import MemoryConnectionError, MemoryNotFoundError


@dataclass
class ContextItem:
    """A saved context item from memory-keeper."""

    key: str
    value: str
    category: str
    priority: str
    created_at: datetime
    channel: str


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
