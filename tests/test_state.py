"""Tests for state module - session state machine."""

from datetime import datetime

import pytest

from vibe.exceptions import StateTransitionError
from vibe.state import (
    VALID_TRANSITIONS,
    SessionContext,
    SessionState,
    Task,
)


class TestSessionState:
    """Tests for SessionState enum and transitions."""

    def test_all_states_have_transitions(self):
        """Every state should have defined transitions."""
        for state in SessionState:
            assert state in VALID_TRANSITIONS

    def test_shutdown_is_terminal(self):
        """SHUTDOWN state should have no outgoing transitions."""
        assert VALID_TRANSITIONS[SessionState.SHUTDOWN] == set()

    def test_all_states_can_reach_shutdown(self):
        """All states except SHUTDOWN should transition to SHUTDOWN."""
        for state in SessionState:
            if state != SessionState.SHUTDOWN:
                assert SessionState.SHUTDOWN in VALID_TRANSITIONS[state]

    def test_all_states_can_reach_error(self):
        """All states except SHUTDOWN and ERROR should transition to ERROR."""
        for state in SessionState:
            if state not in (SessionState.SHUTDOWN, SessionState.ERROR):
                assert SessionState.ERROR in VALID_TRANSITIONS[state]


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation_defaults(self):
        """Test task creation with default values."""
        task = Task(id="task-1", description="Test task")
        assert task.id == "task-1"
        assert task.description == "Test task"
        assert task.status == "pending"
        assert task.files == []
        assert task.constraints == []
        assert task.result is None
        assert task.completed_at is None

    def test_task_with_all_fields(self):
        """Test task creation with all fields."""
        task = Task(
            id="task-2",
            description="Full task",
            files=["file1.py", "file2.py"],
            constraints=["no breaking changes"],
            status="in_progress",
        )
        assert task.files == ["file1.py", "file2.py"]
        assert task.constraints == ["no breaking changes"]
        assert task.status == "in_progress"


class TestSessionContext:
    """Tests for SessionContext state machine."""

    def test_initial_state(self):
        """Context starts in INITIALIZING state."""
        ctx = SessionContext()
        assert ctx.state == SessionState.INITIALIZING

    def test_valid_transition(self):
        """Valid transitions should succeed."""
        ctx = SessionContext()
        assert ctx.transition_to(SessionState.IDLE)
        assert ctx.state == SessionState.IDLE

    def test_invalid_transition(self):
        """Invalid transitions should fail."""
        ctx = SessionContext()
        # INITIALIZING cannot go directly to EXECUTING
        assert not ctx.transition_to(SessionState.EXECUTING)
        assert ctx.state == SessionState.INITIALIZING

    def test_transition_updates_last_activity(self):
        """Transition should update last_activity timestamp."""
        ctx = SessionContext()
        old_activity = ctx.last_activity

        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)

        ctx.transition_to(SessionState.IDLE)
        assert ctx.last_activity > old_activity

    def test_full_workflow_transitions(self):
        """Test complete workflow: INIT -> IDLE -> INPUT -> PLAN -> EXEC -> REVIEW -> IDLE."""
        ctx = SessionContext()

        assert ctx.transition_to(SessionState.IDLE)
        assert ctx.transition_to(SessionState.AWAITING_INPUT)
        assert ctx.transition_to(SessionState.PLANNING)
        assert ctx.transition_to(SessionState.EXECUTING)
        assert ctx.transition_to(SessionState.REVIEWING)
        assert ctx.transition_to(SessionState.IDLE)

        assert ctx.state == SessionState.IDLE

    def test_review_rejection_retry(self):
        """Test retry path: REVIEWING -> EXECUTING on rejection."""
        ctx = SessionContext()

        ctx.transition_to(SessionState.IDLE)
        ctx.transition_to(SessionState.AWAITING_INPUT)
        ctx.transition_to(SessionState.PLANNING)
        ctx.transition_to(SessionState.EXECUTING)
        ctx.transition_to(SessionState.REVIEWING)

        # Rejection should allow retry
        assert ctx.transition_to(SessionState.EXECUTING)
        assert ctx.state == SessionState.EXECUTING

    def test_can_transition_to(self):
        """Test can_transition_to helper method."""
        ctx = SessionContext()

        assert ctx.can_transition_to(SessionState.IDLE)
        assert ctx.can_transition_to(SessionState.ERROR)
        assert ctx.can_transition_to(SessionState.SHUTDOWN)
        assert not ctx.can_transition_to(SessionState.EXECUTING)

    def test_require_transition_valid(self):
        """require_transition should succeed for valid transitions."""
        ctx = SessionContext()
        ctx.require_transition(SessionState.IDLE)
        assert ctx.state == SessionState.IDLE

    def test_require_transition_invalid(self):
        """require_transition should raise StateTransitionError for invalid transitions."""
        ctx = SessionContext()

        with pytest.raises(StateTransitionError) as exc_info:
            ctx.require_transition(SessionState.EXECUTING)

        assert "INITIALIZING" in str(exc_info.value)
        assert "EXECUTING" in str(exc_info.value)
        assert exc_info.value.from_state == "INITIALIZING"
        assert exc_info.value.to_state == "EXECUTING"

    def test_error_recovery(self):
        """Test ERROR state can recover to IDLE."""
        ctx = SessionContext()
        ctx.transition_to(SessionState.ERROR)
        assert ctx.transition_to(SessionState.IDLE)
        assert ctx.state == SessionState.IDLE


class TestSessionContextTaskManagement:
    """Tests for task management in SessionContext."""

    def test_queue_task(self):
        """Test adding tasks to queue."""
        ctx = SessionContext()
        task = Task(id="t1", description="Test")

        ctx.queue_task(task)

        assert len(ctx.task_queue) == 1
        assert ctx.task_queue[0] == task

    def test_next_task(self):
        """Test getting next task from queue."""
        ctx = SessionContext()
        task1 = Task(id="t1", description="First")
        task2 = Task(id="t2", description="Second")

        ctx.queue_task(task1)
        ctx.queue_task(task2)

        next_task = ctx.next_task()

        assert next_task == task1
        assert next_task.status == "in_progress"
        assert ctx.current_task == task1
        assert len(ctx.task_queue) == 1

    def test_next_task_empty_queue(self):
        """Test next_task returns None for empty queue."""
        ctx = SessionContext()
        assert ctx.next_task() is None

    def test_complete_current_task(self):
        """Test completing the current task."""
        ctx = SessionContext()
        task = Task(id="t1", description="Test")
        ctx.queue_task(task)
        ctx.next_task()

        result = {"output": "success"}
        ctx.complete_current_task(result)

        assert ctx.current_task is None
        assert len(ctx.completed_tasks) == 1
        assert ctx.completed_tasks[0].status == "completed"
        assert ctx.completed_tasks[0].result == result
        assert ctx.completed_tasks[0].completed_at is not None

    def test_fail_current_task(self):
        """Test failing the current task."""
        ctx = SessionContext()
        task = Task(id="t1", description="Test")
        ctx.queue_task(task)
        ctx.next_task()

        ctx.fail_current_task("Something went wrong")

        assert ctx.current_task is None
        assert len(ctx.completed_tasks) == 1
        assert ctx.completed_tasks[0].status == "failed"
        assert ctx.error_count == 1
        assert ctx.last_error == "Something went wrong"

    def test_add_completed_task(self):
        """Test adding completed task by description."""
        ctx = SessionContext()
        ctx.add_completed_task("Did something")

        assert len(ctx.completed_tasks) == 1
        assert ctx.completed_tasks[0].description == "Did something"
        assert ctx.completed_tasks[0].status == "completed"

    def test_add_error(self):
        """Test recording errors without current task."""
        ctx = SessionContext()
        ctx.add_error("Error 1")
        ctx.add_error("Error 2")

        assert ctx.error_count == 2
        assert ctx.last_error == "Error 2"


class TestSessionContextGLMMessages:
    """Tests for GLM message handling."""

    def test_add_glm_message(self):
        """Test adding GLM messages."""
        ctx = SessionContext()
        ctx.add_glm_message("user", "Hello")
        ctx.add_glm_message("assistant", "Hi there")

        assert len(ctx.glm_messages) == 2
        assert ctx.glm_messages[0] == {"role": "user", "content": "Hello"}
        assert ctx.glm_messages[1] == {"role": "assistant", "content": "Hi there"}


class TestSessionContextStats:
    """Tests for session statistics."""

    def test_get_stats(self):
        """Test getting session statistics."""
        ctx = SessionContext()
        ctx.project_name = "test-project"

        task1 = Task(id="t1", description="Task 1")
        task2 = Task(id="t2", description="Task 2")
        ctx.queue_task(task1)
        ctx.completed_tasks.append(task2)
        ctx.error_count = 1

        stats = ctx.get_stats()

        assert stats["state"] == "INITIALIZING"
        assert stats["project"] == "test-project"
        assert stats["pending_tasks"] == 1
        assert stats["completed_tasks"] == 1
        assert stats["error_count"] == 1
        assert "duration_seconds" in stats
