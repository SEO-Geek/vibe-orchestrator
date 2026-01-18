"""
Pattern Learning - Save and recall successful task patterns.

Captures what works for each project/task type combination:
- Prompt styles that succeed
- Tools commonly used
- Time to completion
- Common failure patterns to avoid

This enables continuous improvement as Vibe learns from experience.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.config import CONFIG_DIR

if TYPE_CHECKING:
    from vibe.orchestrator.task_enforcer import TaskType

logger = logging.getLogger(__name__)

# Database for pattern storage
PATTERNS_DB = CONFIG_DIR / "task_patterns.db"


@dataclass
class TaskPattern:
    """A learned pattern from successful task execution."""

    id: str
    project: str
    task_type: str  # TaskType enum value
    description_template: str  # Generalized description
    tools_used: list[str] = field(default_factory=list)
    success_count: int = 1
    failure_count: int = 0
    avg_duration_seconds: float = 0.0
    avg_cost_usd: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100

    @property
    def is_reliable(self) -> bool:
        """Check if pattern is reliable (>75% success, >3 uses)."""
        return self.success_rate > 75 and (self.success_count + self.failure_count) >= 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "project": self.project,
            "task_type": self.task_type,
            "description_template": self.description_template,
            "tools_used": self.tools_used,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_duration_seconds": self.avg_duration_seconds,
            "avg_cost_usd": self.avg_cost_usd,
            "success_rate": self.success_rate,
            "is_reliable": self.is_reliable,
            "last_used": self.last_used.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FailurePattern:
    """A pattern that consistently fails - to avoid repeating."""

    id: str
    project: str
    task_type: str
    error_pattern: str  # Common error message pattern
    failure_count: int = 1
    last_feedback: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project": self.project,
            "task_type": self.task_type,
            "error_pattern": self.error_pattern,
            "failure_count": self.failure_count,
            "last_feedback": self.last_feedback,
            "created_at": self.created_at.isoformat(),
        }


class PatternLearner:
    """
    Learns from task execution patterns for continuous improvement.

    Stores successful patterns per project and task type, and tracks
    failure patterns to avoid repeating mistakes.
    """

    def __init__(self, project: str, db_path: Path | None = None):
        """
        Initialize pattern learner for a project.

        Args:
            project: Project name for isolation
            db_path: Custom database path
        """
        self.project = project
        self._db_path = db_path or PATTERNS_DB
        self._lock = threading.Lock()

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Create database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_patterns (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    description_template TEXT NOT NULL,
                    tools_used TEXT,
                    success_count INTEGER DEFAULT 1,
                    failure_count INTEGER DEFAULT 0,
                    avg_duration_seconds REAL DEFAULT 0,
                    avg_cost_usd REAL DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_project_type
                ON task_patterns(project, task_type)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    error_pattern TEXT NOT NULL,
                    failure_count INTEGER DEFAULT 1,
                    last_feedback TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failures_project
                ON failure_patterns(project)
            """)

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with error handling."""
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"PatternLearner: Database connection failed: {e}")
            raise

    def _generalize_description(self, description: str) -> str:
        """
        Generalize a task description to create a reusable pattern.

        Removes specific names, paths, and values to create a template.

        Args:
            description: Specific task description

        Returns:
            Generalized template
        """
        import re

        template = description

        # Replace quoted strings with placeholder
        template = re.sub(r'"[^"]*"', '"<VALUE>"', template)
        template = re.sub(r"'[^']*'", "'<VALUE>'", template)

        # Replace file paths
        template = re.sub(r"/[\w/.-]+\.\w+", "<FILE_PATH>", template)

        # Replace numbers
        template = re.sub(r"\b\d+\b", "<NUM>", template)

        # Replace URLs
        template = re.sub(r"https?://\S+", "<URL>", template)

        # Limit length
        if len(template) > 200:
            template = template[:197] + "..."

        return template

    def record_success(
        self,
        task_description: str,
        task_type: "TaskType",
        tools_used: list[str],
        duration_seconds: float = 0.0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> TaskPattern | None:
        """
        Record a successful task execution to learn from.

        Updates existing pattern or creates new one.

        Args:
            task_description: Original task description
            task_type: Type of task
            tools_used: Tools used in execution
            duration_seconds: How long it took
            cost_usd: Cost of execution
            metadata: Additional metadata

        Returns:
            Updated or created TaskPattern, or None if database error
        """
        template = self._generalize_description(task_description)
        pattern_id = f"{self.project}:{task_type.value}:{hash(template)}"

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Check for existing pattern
                    cursor = conn.execute(
                        "SELECT * FROM task_patterns WHERE id = ?",
                        (pattern_id,),
                    )
                    existing = cursor.fetchone()

                    now = datetime.now()

                    if existing:
                        # Update existing pattern with new data
                        old_count = existing["success_count"]
                        new_count = old_count + 1

                        # Running average for duration and cost
                        new_avg_duration = (
                            (existing["avg_duration_seconds"] * old_count + duration_seconds)
                            / new_count
                        )
                        new_avg_cost = (
                            (existing["avg_cost_usd"] * old_count + cost_usd)
                            / new_count
                        )

                        # Merge tools
                        old_tools = json.loads(existing["tools_used"]) if existing["tools_used"] else []
                        merged_tools = list(set(old_tools + tools_used))

                        conn.execute("""
                            UPDATE task_patterns SET
                                success_count = ?,
                                tools_used = ?,
                                avg_duration_seconds = ?,
                                avg_cost_usd = ?,
                                last_used = ?
                            WHERE id = ?
                        """, (
                            new_count,
                            json.dumps(merged_tools),
                            new_avg_duration,
                            new_avg_cost,
                            now.isoformat(),
                            pattern_id,
                        ))
                        conn.commit()

                        return TaskPattern(
                            id=pattern_id,
                            project=self.project,
                            task_type=task_type.value,
                            description_template=template,
                            tools_used=merged_tools,
                            success_count=new_count,
                            failure_count=existing["failure_count"],
                            avg_duration_seconds=new_avg_duration,
                            avg_cost_usd=new_avg_cost,
                            last_used=now,
                            created_at=datetime.fromisoformat(existing["created_at"]),
                        )

                    else:
                        # Create new pattern
                        conn.execute("""
                            INSERT INTO task_patterns
                            (id, project, task_type, description_template, tools_used,
                             success_count, failure_count, avg_duration_seconds, avg_cost_usd,
                             last_used, created_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pattern_id,
                            self.project,
                            task_type.value,
                            template,
                            json.dumps(tools_used),
                            1, 0,
                            duration_seconds,
                            cost_usd,
                            now.isoformat(),
                            now.isoformat(),
                            json.dumps(metadata or {}),
                        ))
                        conn.commit()

                        logger.info(f"PatternLearner: Created new pattern for {task_type.value}")

                        return TaskPattern(
                            id=pattern_id,
                            project=self.project,
                            task_type=task_type.value,
                            description_template=template,
                            tools_used=tools_used,
                            success_count=1,
                            failure_count=0,
                            avg_duration_seconds=duration_seconds,
                            avg_cost_usd=cost_usd,
                            last_used=now,
                            created_at=now,
                        )
        except sqlite3.Error as e:
            logger.error(f"PatternLearner: Failed to record success: {e}")
            return None

    def record_failure(
        self,
        task_description: str,
        task_type: "TaskType",
        feedback: str,
    ) -> FailurePattern | TaskPattern | None:
        """
        Record a task failure to learn what to avoid.

        Either increments failure count on existing pattern or
        creates new failure pattern.

        Args:
            task_description: Failed task description
            task_type: Type of task
            feedback: GLM feedback explaining failure

        Returns:
            Updated TaskPattern or new FailurePattern, or None on error
        """
        template = self._generalize_description(task_description)
        pattern_id = f"{self.project}:{task_type.value}:{hash(template)}"
        failure_id = f"fail:{pattern_id}"

        # Log full feedback before truncation (for debugging)
        if len(feedback) > 500:
            logger.debug(f"PatternLearner: Full feedback ({len(feedback)} chars): {feedback}")
            feedback_stored = feedback[:497] + "..."
        else:
            feedback_stored = feedback

        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Check if we have a success pattern to update
                    cursor = conn.execute(
                        "SELECT * FROM task_patterns WHERE id = ?",
                        (pattern_id,),
                    )
                    success_pattern = cursor.fetchone()

                    if success_pattern:
                        # Increment failure count on existing success pattern
                        conn.execute("""
                            UPDATE task_patterns SET failure_count = failure_count + 1
                            WHERE id = ?
                        """, (pattern_id,))
                        conn.commit()

                        return TaskPattern(
                            id=pattern_id,
                            project=self.project,
                            task_type=task_type.value,
                            description_template=template,
                            tools_used=json.loads(success_pattern["tools_used"]) if success_pattern["tools_used"] else [],
                            success_count=success_pattern["success_count"],
                            failure_count=success_pattern["failure_count"] + 1,
                            avg_duration_seconds=success_pattern["avg_duration_seconds"],
                            avg_cost_usd=success_pattern["avg_cost_usd"],
                        )

                    # Check for existing failure pattern
                    cursor = conn.execute(
                        "SELECT * FROM failure_patterns WHERE id = ?",
                        (failure_id,),
                    )
                    existing_failure = cursor.fetchone()

                    now = datetime.now()

                    if existing_failure:
                        conn.execute("""
                            UPDATE failure_patterns SET
                                failure_count = failure_count + 1,
                                last_feedback = ?
                            WHERE id = ?
                        """, (feedback_stored, failure_id))
                        conn.commit()

                        return FailurePattern(
                            id=failure_id,
                            project=self.project,
                            task_type=task_type.value,
                            error_pattern=template,
                            failure_count=existing_failure["failure_count"] + 1,
                            last_feedback=feedback_stored,
                        )

                    # Create new failure pattern
                    conn.execute("""
                        INSERT INTO failure_patterns
                        (id, project, task_type, error_pattern, failure_count, last_feedback, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        failure_id,
                        self.project,
                        task_type.value,
                        template,
                        1,
                        feedback_stored,
                        now.isoformat(),
                    ))
                    conn.commit()

                    logger.info(f"PatternLearner: Recorded failure pattern for {task_type.value}")

                    return FailurePattern(
                        id=failure_id,
                        project=self.project,
                        task_type=task_type.value,
                        error_pattern=template,
                        failure_count=1,
                        last_feedback=feedback_stored,
                        created_at=now,
                    )
        except sqlite3.Error as e:
            logger.error(f"PatternLearner: Failed to record failure: {e}")
            return None

    def get_relevant_patterns(
        self,
        task_type: "TaskType",
        limit: int = 5,
    ) -> list[TaskPattern]:
        """
        Get reliable patterns for a task type.

        Returns patterns with high success rates to guide future tasks.

        Args:
            task_type: Type of task
            limit: Maximum patterns to return

        Returns:
            List of reliable TaskPatterns
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM task_patterns
                WHERE project = ? AND task_type = ?
                  AND (success_count * 1.0 / (success_count + failure_count + 0.001)) > 0.75
                ORDER BY success_count DESC
                LIMIT ?
            """, (self.project, task_type.value, limit))

            patterns = []
            for row in cursor.fetchall():
                patterns.append(TaskPattern(
                    id=row["id"],
                    project=row["project"],
                    task_type=row["task_type"],
                    description_template=row["description_template"],
                    tools_used=json.loads(row["tools_used"]) if row["tools_used"] else [],
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                    avg_duration_seconds=row["avg_duration_seconds"] or 0,
                    avg_cost_usd=row["avg_cost_usd"] or 0,
                    last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else datetime.now(),
                    created_at=datetime.fromisoformat(row["created_at"]),
                ))

            return patterns

    def get_failure_warnings(
        self,
        task_type: "TaskType",
        limit: int = 3,
    ) -> list[FailurePattern]:
        """
        Get common failure patterns to warn about.

        Returns patterns that frequently fail for this task type.

        Args:
            task_type: Type of task
            limit: Maximum warnings to return

        Returns:
            List of FailurePatterns to avoid
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM failure_patterns
                WHERE project = ? AND task_type = ?
                  AND failure_count >= 2
                ORDER BY failure_count DESC
                LIMIT ?
            """, (self.project, task_type.value, limit))

            failures = []
            for row in cursor.fetchall():
                failures.append(FailurePattern(
                    id=row["id"],
                    project=row["project"],
                    task_type=row["task_type"],
                    error_pattern=row["error_pattern"],
                    failure_count=row["failure_count"],
                    last_feedback=row["last_feedback"] or "",
                    created_at=datetime.fromisoformat(row["created_at"]),
                ))

            return failures

    def generate_learnings_context(self, task_type: "TaskType") -> str:
        """
        Generate context string with learnings for Claude prompt injection.

        Includes recommended tools and warnings about common failures.

        Args:
            task_type: Type of task being executed

        Returns:
            Formatted string to inject into task context
        """
        parts = []

        # Get successful patterns
        patterns = self.get_relevant_patterns(task_type, limit=3)
        if patterns:
            parts.append("## LEARNED PATTERNS (from previous successful tasks):")
            for p in patterns:
                tools_str = ", ".join(p.tools_used[:5]) if p.tools_used else "various"
                parts.append(
                    f"- {p.task_type}: Used tools [{tools_str}], "
                    f"avg time {p.avg_duration_seconds:.0f}s, "
                    f"success rate {p.success_rate:.0f}%"
                )
            parts.append("")

        # Get failure warnings
        failures = self.get_failure_warnings(task_type, limit=2)
        if failures:
            parts.append("## COMMON PITFALLS TO AVOID:")
            for f in failures:
                parts.append(f"- Failed {f.failure_count}x: {f.last_feedback[:100]}")
            parts.append("")

        return "\n".join(parts) if parts else ""

    def get_decomposition_hints(self, request_text: str = "") -> str:
        """
        Generate hints for Gemini's task decomposition based on historical patterns.

        This allows Gemini to learn from past successes and failures when
        breaking down new requests.

        Args:
            request_text: The user's request (used for keyword-based type guessing)

        Returns:
            Formatted string with decomposition hints for Gemini
        """
        parts = []

        # Get all task type statistics
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT task_type,
                       SUM(success_count) as successes,
                       SUM(failure_count) as failures,
                       AVG(avg_duration_seconds) as avg_duration
                FROM task_patterns
                WHERE project = ?
                GROUP BY task_type
            """, (self.project,))
            stats = cursor.fetchall()

        if stats:
            parts.append("Task type success rates for this project:")
            for row in stats:
                task_type = row["task_type"]
                successes = row["successes"] or 0
                failures = row["failures"] or 0
                total = successes + failures
                if total > 0:
                    rate = (successes / total) * 100
                    avg_dur = row["avg_duration"] or 0
                    status = "✓" if rate >= 75 else "⚠" if rate >= 50 else "✗"
                    parts.append(
                        f"- {task_type}: {rate:.0f}% success ({successes}/{total}), "
                        f"avg {avg_dur:.0f}s {status}"
                    )

        # Get top failure patterns across all types
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT task_type, error_pattern, failure_count, last_feedback
                FROM failure_patterns
                WHERE project = ?
                ORDER BY failure_count DESC
                LIMIT 5
            """, (self.project,))
            failures = cursor.fetchall()

        if failures:
            parts.append("\nCommon failure patterns to avoid:")
            for row in failures:
                feedback = (row["last_feedback"] or "")[:80]
                parts.append(f"- {row['task_type']}: {feedback}...")

        # Add specific guidance based on stats
        if parts:
            parts.append("\nDecomposition guidance:")
            parts.append("- Break high-failure task types into smaller steps")
            parts.append("- Add verification tasks after code changes")
            parts.append("- For debug tasks: investigate first, then fix in separate task")

        return "\n".join(parts) if parts else "(No historical patterns yet)"

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics for this project."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM task_patterns WHERE project = ?",
                (self.project,),
            )
            pattern_count = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM failure_patterns WHERE project = ?",
                (self.project,),
            )
            failure_count = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT task_type, SUM(success_count) as successes, SUM(failure_count) as failures
                FROM task_patterns WHERE project = ?
                GROUP BY task_type
            """, (self.project,))

            by_type = {}
            for row in cursor.fetchall():
                by_type[row["task_type"]] = {
                    "successes": row["successes"],
                    "failures": row["failures"],
                }

        return {
            "total_patterns": pattern_count,
            "total_failure_patterns": failure_count,
            "by_task_type": by_type,
        }


# Module-level cached instances per project
_cached_learners: dict[str, PatternLearner] = {}
_learners_lock = threading.Lock()


def get_pattern_learner(project: str) -> PatternLearner:
    """
    Get or create PatternLearner for a project.

    Args:
        project: Project name

    Returns:
        Cached PatternLearner instance
    """
    with _learners_lock:
        if project not in _cached_learners:
            _cached_learners[project] = PatternLearner(project)
        return _cached_learners[project]


def reset_pattern_learners() -> None:
    """Reset all cached learners."""
    global _cached_learners
    with _learners_lock:
        _cached_learners = {}
