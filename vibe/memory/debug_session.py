"""
Debug Session Tracking

Tracks debugging sessions to prevent Claude from:
- Repeating failed attempts
- Losing context on what was already tried
- Making "bush fixes" that break existing functionality
- Forgetting the original hypothesis

Provides:
- Attempt history with pass/fail and reasoning
- Feature preservation checklists
- Checkpoint/rollback capabilities
- Context injection for Claude prompts
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AttemptResult(Enum):
    """Result of a debugging attempt."""

    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL = "partial"  # Helped but didn't fully solve
    FAILED = "failed"
    MADE_WORSE = "made_worse"


@dataclass
class DebugAttempt:
    """A single debugging attempt."""

    id: int
    description: str
    hypothesis: str
    result: AttemptResult = AttemptResult.PENDING
    reason: str = ""  # Why it failed/succeeded
    files_modified: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    rollback_commit: str = ""  # Git commit to rollback to

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "result": self.result.value,
            "reason": self.reason,
            "files_modified": self.files_modified,
            "timestamp": self.timestamp.isoformat(),
            "rollback_commit": self.rollback_commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebugAttempt":
        return cls(
            id=data["id"],
            description=data["description"],
            hypothesis=data["hypothesis"],
            result=AttemptResult(data["result"]),
            reason=data.get("reason", ""),
            files_modified=data.get("files_modified", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            rollback_commit=data.get("rollback_commit", ""),
        )


@dataclass
class DebugSession:
    """
    Tracks a debugging session with attempts, hypotheses, and rollback points.

    Usage:
        session = DebugSession(
            project_path="/home/brian/myproject",
            problem="API returns 500 on /users endpoint"
        )
        session.add_must_preserve("User authentication must still work")
        session.add_must_preserve("Rate limiting must remain active")

        # Before each attempt
        attempt = session.start_attempt(
            description="Add null check in user handler",
            hypothesis="Null user object causing crash"
        )

        # After attempt
        session.complete_attempt(
            attempt_id=attempt.id,
            result=AttemptResult.FAILED,
            reason="Still crashes, null check didn't help"
        )

        # Get context for Claude
        context = session.get_context_for_claude()
    """

    project_path: str
    problem: str
    current_hypothesis: str = ""
    must_preserve: list[str] = field(default_factory=list)
    attempts: list[DebugAttempt] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    initial_commit: str = ""  # Git commit when session started
    is_active: bool = True
    next_steps: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Capture initial git state."""
        if not self.initial_commit:
            self.initial_commit = self._get_current_commit()

    def _get_current_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else ""
        except Exception:
            return ""

    def _create_checkpoint(self, message: str) -> str:
        """Create a git commit as checkpoint, return commit hash."""
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_path,
                capture_output=True,
                timeout=10,
            )

            # Commit with message
            result = subprocess.run(
                ["git", "commit", "-m", f"[DEBUG CHECKPOINT] {message}"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return self._get_current_commit()
            return ""
        except Exception:
            return ""

    def add_must_preserve(self, feature: str) -> None:
        """Add a feature that must be preserved during debugging."""
        if feature not in self.must_preserve:
            self.must_preserve.append(feature)

    def remove_must_preserve(self, feature: str) -> bool:
        """Remove a feature from preservation list."""
        if feature in self.must_preserve:
            self.must_preserve.remove(feature)
            return True
        return False

    def set_hypothesis(self, hypothesis: str) -> None:
        """Set the current working hypothesis."""
        self.current_hypothesis = hypothesis

    def add_next_step(self, step: str) -> None:
        """Add a next step to try."""
        if step not in self.next_steps:
            self.next_steps.append(step)

    def start_attempt(
        self,
        description: str,
        hypothesis: str | None = None,
        create_checkpoint: bool = True,
    ) -> DebugAttempt:
        """
        Start a new debugging attempt.

        Creates a git checkpoint before the attempt so we can rollback.
        """
        attempt_id = len(self.attempts) + 1

        # Use provided hypothesis or current session hypothesis
        hyp = hypothesis or self.current_hypothesis

        # Create checkpoint before attempt
        rollback_commit = ""
        if create_checkpoint:
            rollback_commit = self._create_checkpoint(
                f"Before attempt #{attempt_id}: {description[:50]}"
            )

        attempt = DebugAttempt(
            id=attempt_id,
            description=description,
            hypothesis=hyp,
            rollback_commit=rollback_commit or self._get_current_commit(),
        )

        self.attempts.append(attempt)
        return attempt

    def complete_attempt(
        self,
        attempt_id: int,
        result: AttemptResult,
        reason: str,
        files_modified: list[str] | None = None,
    ) -> None:
        """Complete an attempt with result and reason."""
        for attempt in self.attempts:
            if attempt.id == attempt_id:
                attempt.result = result
                attempt.reason = reason
                if files_modified:
                    attempt.files_modified = files_modified

                # Remove from next steps if it was there
                if attempt.description in self.next_steps:
                    self.next_steps.remove(attempt.description)
                break

    def rollback_to_attempt(self, attempt_id: int) -> bool:
        """
        Rollback to the state before a specific attempt.

        Returns True if successful.
        """
        for attempt in self.attempts:
            if attempt.id == attempt_id and attempt.rollback_commit:
                try:
                    result = subprocess.run(
                        ["git", "checkout", attempt.rollback_commit, "--", "."],
                        cwd=self.project_path,
                        capture_output=True,
                        timeout=30,
                    )
                    return result.returncode == 0
                except Exception:
                    return False
        return False

    def rollback_to_start(self) -> bool:
        """Rollback to the initial state when debug session started."""
        if not self.initial_commit:
            return False
        try:
            result = subprocess.run(
                ["git", "checkout", self.initial_commit, "--", "."],
                cwd=self.project_path,
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_failed_attempts(self) -> list[DebugAttempt]:
        """Get all failed attempts."""
        return [
            a for a in self.attempts if a.result in (AttemptResult.FAILED, AttemptResult.MADE_WORSE)
        ]

    def get_successful_attempts(self) -> list[DebugAttempt]:
        """Get all successful attempts."""
        return [a for a in self.attempts if a.result == AttemptResult.SUCCESS]

    def get_partial_attempts(self) -> list[DebugAttempt]:
        """Get attempts that partially helped."""
        return [a for a in self.attempts if a.result == AttemptResult.PARTIAL]

    def was_already_tried(self, description: str) -> DebugAttempt | None:
        """Check if something similar was already tried."""
        desc_lower = description.lower()
        for attempt in self.attempts:
            # Simple similarity check - could be enhanced with embeddings
            if (
                desc_lower in attempt.description.lower()
                or attempt.description.lower() in desc_lower
            ):
                return attempt
        return None

    def get_context_for_claude(self) -> str:
        """
        Generate context string to inject into Claude's prompt.

        This is the key function that prevents Claude from:
        - Repeating failed attempts
        - Forgetting what was already tried
        - Breaking preserved functionality
        """
        lines = [
            "=" * 60,
            "DEBUG SESSION STATE - READ CAREFULLY",
            "=" * 60,
            "",
            f"PROBLEM: {self.problem}",
            "",
        ]

        if self.current_hypothesis:
            lines.extend(
                [
                    f"CURRENT HYPOTHESIS: {self.current_hypothesis}",
                    "",
                ]
            )

        # Must preserve features
        if self.must_preserve:
            lines.extend(
                [
                    "MUST PRESERVE (do NOT break these):",
                ]
            )
            for i, feature in enumerate(self.must_preserve, 1):
                lines.append(f"  {i}. {feature}")
            lines.append("")

        # Failed attempts - CRITICAL
        failed = self.get_failed_attempts()
        if failed:
            lines.extend(
                [
                    "ALREADY TRIED - FAILED (do NOT repeat):",
                ]
            )
            for attempt in failed:
                lines.append(f"  ✗ Attempt #{attempt.id}: {attempt.description}")
                lines.append(f"    Why it failed: {attempt.reason}")
            lines.append("")

        # Partial successes
        partial = self.get_partial_attempts()
        if partial:
            lines.extend(
                [
                    "PARTIALLY WORKED (may be useful):",
                ]
            )
            for attempt in partial:
                lines.append(f"  ~ Attempt #{attempt.id}: {attempt.description}")
                lines.append(f"    Result: {attempt.reason}")
            lines.append("")

        # Successful attempts
        successful = self.get_successful_attempts()
        if successful:
            lines.extend(
                [
                    "SUCCESSFUL FIXES (already applied):",
                ]
            )
            for attempt in successful:
                lines.append(f"  ✓ Attempt #{attempt.id}: {attempt.description}")
            lines.append("")

        # Next steps to try
        if self.next_steps:
            lines.extend(
                [
                    "SUGGESTED NEXT STEPS:",
                ]
            )
            for i, step in enumerate(self.next_steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        # Warnings
        lines.extend(
            [
                "WARNINGS:",
                "  - Do NOT simplify the solution in ways that break preserved features",
                "  - Do NOT try failed approaches again without NEW evidence",
                "  - EXPLAIN your reasoning before making changes",
                "  - If stuck, propose a DIFFERENT approach instead of variations",
                "=" * 60,
                "",
            ]
        )

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a short summary of the debug session."""
        total = len(self.attempts)
        failed = len(self.get_failed_attempts())
        partial = len(self.get_partial_attempts())
        success = len(self.get_successful_attempts())
        pending = total - failed - partial - success

        return (
            f"Debug Session: {self.problem[:50]}...\n"
            f"Attempts: {total} total ({success} success, {partial} partial, "
            f"{failed} failed, {pending} pending)\n"
            f"Features to preserve: {len(self.must_preserve)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "project_path": self.project_path,
            "problem": self.problem,
            "current_hypothesis": self.current_hypothesis,
            "must_preserve": self.must_preserve,
            "attempts": [a.to_dict() for a in self.attempts],
            "created_at": self.created_at.isoformat(),
            "initial_commit": self.initial_commit,
            "is_active": self.is_active,
            "next_steps": self.next_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebugSession":
        """Deserialize from dictionary."""
        session = cls(
            project_path=data["project_path"],
            problem=data["problem"],
            current_hypothesis=data.get("current_hypothesis", ""),
            must_preserve=data.get("must_preserve", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            initial_commit=data.get("initial_commit", ""),
            is_active=data.get("is_active", True),
            next_steps=data.get("next_steps", []),
        )
        session.attempts = [DebugAttempt.from_dict(a) for a in data.get("attempts", [])]
        return session

    def save(self, filepath: str | Path) -> None:
        """Save session to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "DebugSession":
        """Load session from JSON file."""
        with open(filepath) as f:
            return cls.from_dict(json.load(f))
