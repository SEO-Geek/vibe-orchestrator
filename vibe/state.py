"""
Vibe Orchestrator - Session State Machine

Tracks the current state of a Vibe session to ensure proper
transitions and prevent invalid operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class SessionState(Enum):
    """
    Possible states for a Vibe session.

    State transitions:
    INITIALIZING -> IDLE (startup complete)
    IDLE -> AWAITING_INPUT (user prompt)
    AWAITING_INPUT -> PLANNING (GLM decomposing task)
    PLANNING -> EXECUTING (Claude working)
    EXECUTING -> REVIEWING (GLM checking output)
    REVIEWING -> IDLE (approved) or EXECUTING (rejected, retry)
    Any -> ERROR (on failure)
    Any -> SHUTDOWN (on exit)
    """

    INITIALIZING = auto()  # Startup validation in progress
    IDLE = auto()  # Ready for user input
    AWAITING_INPUT = auto()  # Waiting for user to type
    PLANNING = auto()  # GLM decomposing task into subtasks
    EXECUTING = auto()  # Claude executing a task
    REVIEWING = auto()  # GLM reviewing Claude's output
    ERROR = auto()  # Error state, needs recovery
    SHUTDOWN = auto()  # Graceful shutdown in progress


# Valid state transitions
VALID_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.INITIALIZING: {SessionState.IDLE, SessionState.ERROR, SessionState.SHUTDOWN},
    SessionState.IDLE: {SessionState.AWAITING_INPUT, SessionState.ERROR, SessionState.SHUTDOWN},
    SessionState.AWAITING_INPUT: {
        SessionState.PLANNING,
        SessionState.IDLE,
        SessionState.ERROR,
        SessionState.SHUTDOWN,
    },
    SessionState.PLANNING: {
        SessionState.EXECUTING,
        SessionState.IDLE,
        SessionState.ERROR,
        SessionState.SHUTDOWN,
    },
    SessionState.EXECUTING: {
        SessionState.REVIEWING,
        SessionState.ERROR,
        SessionState.SHUTDOWN,
    },
    SessionState.REVIEWING: {
        SessionState.IDLE,
        SessionState.EXECUTING,
        SessionState.ERROR,
        SessionState.SHUTDOWN,
    },
    SessionState.ERROR: {SessionState.IDLE, SessionState.SHUTDOWN},
    SessionState.SHUTDOWN: set(),  # Terminal state
}


@dataclass
class Task:
    """A single task to be executed by Claude."""

    id: str
    description: str
    files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None


@dataclass
class SessionContext:
    """
    Context maintained throughout a Vibe session.

    This includes the current state, loaded project, task queue,
    and conversation history with GLM.
    """

    state: SessionState = SessionState.INITIALIZING
    project_name: str = ""
    project_path: str = ""
    session_id: str = ""
    repo_session_id: str = ""  # New unified persistence session ID

    # Task management
    current_task: Task | None = None
    task_queue: list[Task] = field(default_factory=list)
    completed_tasks: list[Task] = field(default_factory=list)

    # Conversation history with GLM
    glm_messages: list[dict[str, str]] = field(default_factory=list)

    # Session timestamps
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Error tracking
    last_error: str | None = None
    error_count: int = 0

    # Clarification tracking - reset when task decomposition starts
    clarification_count: int = 0

    def transition_to(self, new_state: SessionState) -> bool:
        """
        Attempt to transition to a new state.

        Args:
            new_state: The target state

        Returns:
            True if transition was valid and performed, False otherwise
        """
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            self.state = new_state
            self.last_activity = datetime.now()
            return True
        return False

    def require_transition(self, new_state: SessionState) -> None:
        """
        Transition to a new state, raising an exception if invalid.

        Use this instead of transition_to() when the transition MUST succeed
        and failure indicates a bug in the state machine logic.

        Args:
            new_state: The target state

        Raises:
            StateTransitionError: If the transition is not valid
        """
        from vibe.exceptions import StateTransitionError

        if not self.transition_to(new_state):
            valid_targets = VALID_TRANSITIONS.get(self.state, set())
            valid_names = ", ".join(s.name for s in valid_targets) or "none"
            raise StateTransitionError(
                f"Invalid state transition: {self.state.name} -> {new_state.name}. "
                f"Valid transitions from {self.state.name}: {valid_names}",
                from_state=self.state.name,
                to_state=new_state.name,
            )

    def can_transition_to(self, new_state: SessionState) -> bool:
        """Check if transition to new_state is valid from current state."""
        return new_state in VALID_TRANSITIONS.get(self.state, set())

    def add_glm_message(self, role: str, content: str) -> None:
        """Add a message to GLM conversation history."""
        self.glm_messages.append({"role": role, "content": content})
        self.last_activity = datetime.now()

    def queue_task(self, task: Task) -> None:
        """Add a task to the queue."""
        self.task_queue.append(task)

    def next_task(self) -> Task | None:
        """Get the next pending task from queue."""
        if self.task_queue:
            task = self.task_queue.pop(0)
            task.status = "in_progress"
            self.current_task = task
            return task
        return None

    def complete_current_task(self, result: dict[str, Any]) -> None:
        """Mark current task as completed."""
        if self.current_task:
            self.current_task.status = "completed"
            self.current_task.result = result
            self.current_task.completed_at = datetime.now()
            self.completed_tasks.append(self.current_task)
            self.current_task = None

    def fail_current_task(self, error: str) -> None:
        """Mark current task as failed."""
        if self.current_task:
            self.current_task.status = "failed"
            self.current_task.result = {"error": error}
            self.current_task.completed_at = datetime.now()
            self.completed_tasks.append(self.current_task)
            self.current_task = None
            self.error_count += 1
            self.last_error = error

    def add_error(self, error: str) -> None:
        """Record an error without requiring a current task."""
        self.error_count += 1
        self.last_error = error
        self.last_activity = datetime.now()

    def add_completed_task(self, description: str) -> None:
        """Record a completed task by description (simple tracking)."""
        task = Task(
            id=f"task-{len(self.completed_tasks) + 1}",
            description=description,
            status="completed",
            completed_at=datetime.now(),
        )
        self.completed_tasks.append(task)
        self.last_activity = datetime.now()

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "state": self.state.name,
            "project": self.project_name,
            "pending_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "error_count": self.error_count,
            "duration_seconds": (datetime.now() - self.started_at).total_seconds(),
        }
