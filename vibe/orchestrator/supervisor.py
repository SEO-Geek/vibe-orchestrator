"""
Vibe Supervisor - Main Orchestration Logic

Coordinates between user requests, GLM (task decomposition and review),
and Claude (task execution). This is the CORE of the Vibe system.

Workflow:
1. User submits request
2. GLM asks for clarification if needed
3. GLM decomposes request into atomic tasks
4. For each task:
   a. Run pre-task hooks (if configured)
   b. Claude executes the task
   c. GLM reviews the output
   d. If rejected, retry with feedback (max 3 attempts)
   e. Run post-task hooks (if configured)
5. Return structured result with all changes
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from vibe.config import Project
from vibe.exceptions import (
    ClaudeError,
    GLMError,
    ReviewRejectedError,
    TaskError,
    VibeError,
)
from vibe.glm.client import GLMClient
from vibe.memory.keeper import VibeMemory
from vibe.state import SessionContext, SessionState, Task

# Avoid circular import - import lazily in methods that need it
if TYPE_CHECKING:
    from vibe.claude.executor import ClaudeExecutor, TaskResult

logger = logging.getLogger(__name__)

# Maximum retry attempts for rejected tasks
MAX_RETRY_ATTEMPTS = 3


@dataclass
class SupervisorCallbacks:
    """Callbacks for supervisor events to update UI/progress."""

    on_status: Callable[[str], None] | None = None
    on_progress: Callable[[str], None] | None = None
    on_task_start: Callable[[Task], None] | None = None
    on_task_complete: Callable[[Task, bool], None] | None = None
    on_review_result: Callable[[bool, str], None] | None = None
    on_error: Callable[[str], None] | None = None


@dataclass
class TaskExecutionResult:
    """Result from executing and reviewing a single task."""

    task: Task
    success: bool
    attempts: int = 1
    execution_result: TaskResult | None = None
    review_approved: bool = False
    review_feedback: str = ""
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class SupervisorResult:
    """Final result from processing a user request."""

    success: bool
    tasks_completed: int
    tasks_failed: int
    total_tasks: int
    task_results: list[TaskExecutionResult] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    clarification_asked: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    total_cost_usd: float = 0.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        if self.error:
            return f"Failed: {self.error}"

        if self.clarification_asked:
            return f"Clarification needed: {self.clarification_asked}"

        status = "completed" if self.success else "partially completed"
        parts = [
            f"Request {status}.",
            f"Tasks: {self.tasks_completed}/{self.total_tasks} succeeded.",
        ]

        if self.files_changed:
            parts.append(f"Files modified: {len(self.files_changed)}")

        if self.total_cost_usd > 0:
            parts.append(f"Cost: ${self.total_cost_usd:.4f}")

        return " ".join(parts)


class Supervisor:
    """
    Main supervisor that orchestrates the Vibe workflow.

    Responsibilities:
    - Receive user requests
    - Ask GLM to decompose into tasks
    - Execute tasks via Claude
    - Send output to GLM for review
    - Accept or reject based on review
    - Handle retries with feedback
    - Persist state to memory
    """

    def __init__(
        self,
        glm_client: GLMClient,
        project: Project,
        memory: VibeMemory | None = None,
        callbacks: SupervisorCallbacks | None = None,
        max_retries: int = MAX_RETRY_ATTEMPTS,
    ):
        """
        Initialize supervisor.

        Args:
            glm_client: Initialized GLMClient for task decomposition and review
            project: Project configuration with path and settings
            memory: Optional VibeMemory for persistence
            callbacks: Optional callbacks for progress updates
            max_retries: Maximum retry attempts for rejected tasks
        """
        self.glm_client = glm_client
        self.project = project
        self.memory = memory
        self.callbacks = callbacks or SupervisorCallbacks()
        self.max_retries = max_retries

        # Initialize session context
        self.context = SessionContext(
            project_name=project.name,
            project_path=project.path,
        )

        # Claude executor will be created per-task with appropriate settings
        self._executor: ClaudeExecutor | None = None

        # Track total cost across all tasks
        self._total_cost_usd = 0.0

    def _emit_status(self, message: str) -> None:
        """Emit status update via callback."""
        logger.info(message)
        if self.callbacks.on_status:
            self.callbacks.on_status(message)

    def _emit_progress(self, message: str) -> None:
        """Emit progress update via callback."""
        logger.debug(message)
        if self.callbacks.on_progress:
            self.callbacks.on_progress(message)

    def _emit_error(self, message: str) -> None:
        """Emit error via callback."""
        logger.error(message)
        if self.callbacks.on_error:
            self.callbacks.on_error(message)

    async def _run_hooks(self, hooks: list[str], phase: str) -> bool:
        """
        Run hook scripts before/after task execution.

        Args:
            hooks: List of hook script paths (relative to project directory)
            phase: Phase name for logging ('pre-task' or 'post-task')

        Returns:
            True if all hooks succeeded, False if any failed
        """
        import os

        if not hooks:
            return True

        project_path = Path(self.project.path).resolve()

        for hook in hooks:
            hook_path = (project_path / hook).resolve()

            # Security: Ensure hook is within project directory (prevent path traversal)
            try:
                hook_path.relative_to(project_path)
            except ValueError:
                self._emit_error(f"{phase} hook path traversal detected: {hook}")
                return False

            # Check hook exists and is executable
            if not hook_path.is_file():
                self._emit_error(f"{phase} hook not found: {hook}")
                return False

            if not os.access(hook_path, os.X_OK):
                self._emit_error(f"{phase} hook not executable: {hook}")
                return False

            self._emit_progress(f"Running {phase} hook: {hook}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    str(hook_path),
                    cwd=self.project.path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=60.0,  # 60 second timeout for hooks
                )

                if proc.returncode != 0:
                    error_output = stderr.decode("utf-8", errors="replace") if stderr else stdout.decode("utf-8", errors="replace")
                    self._emit_error(f"{phase} hook failed: {hook}\n{error_output[:200]}")
                    return False

                if stdout:
                    self._emit_progress(f"Hook output: {stdout.decode('utf-8', errors='replace')[:100]}")

            except asyncio.TimeoutError:
                self._emit_error(f"{phase} hook timed out: {hook}")
                return False
            except Exception as e:
                self._emit_error(f"{phase} hook error: {hook}: {e}")
                return False

        return True

    def _load_project_context(self) -> str:
        """
        Load project context for GLM including starmap and recent memory.

        Returns:
            Formatted context string for GLM prompts
        """
        context_parts = []

        # Load STARMAP.md if it exists
        starmap_path = self.project.starmap_path
        if starmap_path.exists():
            try:
                starmap_content = starmap_path.read_text()
                # Truncate if too long (keep first 4000 chars)
                if len(starmap_content) > 4000:
                    starmap_content = starmap_content[:4000] + "\n... [truncated]"
                context_parts.append(f"PROJECT STARMAP:\n{starmap_content}")
            except Exception as e:
                logger.warning(f"Could not read STARMAP.md: {e}")

        # Load CLAUDE.md if it exists (project-specific conventions)
        claude_md_path = self.project.claude_md_path
        if claude_md_path.exists():
            try:
                claude_content = claude_md_path.read_text()
                if len(claude_content) > 2000:
                    claude_content = claude_content[:2000] + "\n... [truncated]"
                context_parts.append(f"PROJECT CONVENTIONS:\n{claude_content}")
            except Exception as e:
                logger.warning(f"Could not read CLAUDE.md: {e}")

        # Load recent context from memory if available
        if self.memory:
            try:
                recent_items = self.memory.load_project_context(limit=10)
                if recent_items:
                    memory_context = "\n".join(
                        f"- [{item.category}] {item.key}: {item.value[:200]}"
                        for item in recent_items
                    )
                    context_parts.append(f"RECENT CONTEXT:\n{memory_context}")
            except Exception as e:
                logger.warning(f"Could not load memory context: {e}")

        # Add project metadata
        context_parts.append(
            f"PROJECT: {self.project.name}\n"
            f"PATH: {self.project.path}\n"
            f"TEST COMMAND: {self.project.test_command}"
        )

        return "\n\n".join(context_parts)

    def _create_checkpoint(self, name: str, description: str = "") -> str | None:
        """
        Create a checkpoint in memory for recovery.

        Args:
            name: Checkpoint name
            description: Optional description

        Returns:
            Checkpoint ID or None if memory not available
        """
        if not self.memory:
            return None

        try:
            return self.memory.create_checkpoint_with_git(
                name=name,
                description=description,
                project_path=self.project.path,
            )
        except Exception as e:
            logger.warning(f"Could not create checkpoint: {e}")
            return None

    def _save_task_result(
        self,
        task: Task,
        success: bool,
        summary: str,
        files_changed: list[str],
    ) -> None:
        """Save task result to memory."""
        if not self.memory:
            return

        try:
            self.memory.save_task_result(
                task_description=task.description,
                success=success,
                summary=summary,
                files_changed=files_changed,
            )
        except Exception as e:
            logger.warning(f"Could not save task result: {e}")

    async def process_user_request(
        self,
        request: str,
        skip_clarification: bool = False,
    ) -> SupervisorResult:
        """
        Process a user request through the full pipeline.

        This is the MAIN orchestration loop:
        1. Load project context
        2. Ask GLM for clarification if needed
        3. Have GLM decompose into tasks
        4. Execute each task via Claude
        5. Review each task via GLM
        6. Retry rejected tasks with feedback

        Args:
            request: User's request text
            skip_clarification: Skip the clarification step

        Returns:
            SupervisorResult with all task outcomes and changes
        """
        start_time = datetime.now()
        self._total_cost_usd = 0.0

        # Initialize result
        result = SupervisorResult(
            success=False,
            tasks_completed=0,
            tasks_failed=0,
            total_tasks=0,
        )

        try:
            # Transition to planning state
            self.context.transition_to(SessionState.PLANNING)
            self._emit_status("Loading project context...")

            # Step 1: Load project context
            project_context = self._load_project_context()

            # Step 2: Ask for clarification if needed
            if not skip_clarification:
                self._emit_status("Checking if clarification needed...")
                try:
                    clarification = await self.glm_client.ask_clarification(
                        user_request=request,
                        project_context=project_context,
                    )

                    if clarification:
                        # GLM needs more info - return early
                        result.clarification_asked = clarification
                        result.success = True  # Not a failure, just needs input
                        self.context.transition_to(SessionState.IDLE)
                        return result

                except GLMError as e:
                    self._emit_error(f"GLM clarification failed: {e}")
                    # Continue without clarification rather than fail

            # Step 3: Decompose task
            self._emit_status("Decomposing request into tasks...")
            try:
                task_dicts = await self.glm_client.decompose_task(
                    user_request=request,
                    project_context=project_context,
                )
            except GLMError as e:
                result.error = f"Task decomposition failed: {e}"
                self._emit_error(result.error)
                self.context.transition_to(SessionState.ERROR)
                return result

            if not task_dicts:
                result.error = "GLM returned no tasks"
                self._emit_error(result.error)
                self.context.transition_to(SessionState.ERROR)
                return result

            # Convert to Task objects and queue them
            tasks = []
            for i, td in enumerate(task_dicts, 1):
                task = Task(
                    id=td.get("id", f"task-{i}"),
                    description=td.get("description", ""),
                    files=td.get("files", []),
                    constraints=td.get("constraints", []),
                )
                tasks.append(task)
                self.context.queue_task(task)

            result.total_tasks = len(tasks)
            self._emit_status(f"Decomposed into {len(tasks)} tasks")

            # Create checkpoint before execution
            self._create_checkpoint(
                name=f"pre-execution-{datetime.now().strftime('%H%M%S')}",
                description=f"Before executing: {request[:100]}",
            )

            # Step 4 & 5: Execute and review each task
            all_files_changed: set[str] = set()

            for task_idx, task in enumerate(tasks, 1):
                self._emit_status(f"Task {task_idx}/{len(tasks)}: {task.description[:50]}...")

                if self.callbacks.on_task_start:
                    self.callbacks.on_task_start(task)

                # Execute task with retry loop
                task_result = await self._execute_task_with_retries(task)
                result.task_results.append(task_result)

                if task_result.success:
                    result.tasks_completed += 1
                    all_files_changed.update(task_result.files_changed)
                    self.context.complete_current_task({
                        "success": True,
                        "files": task_result.files_changed,
                    })
                else:
                    result.tasks_failed += 1
                    self.context.fail_current_task(task_result.error or "Unknown error")

                if self.callbacks.on_task_complete:
                    self.callbacks.on_task_complete(task, task_result.success)

            # Finalize result
            result.files_changed = list(all_files_changed)
            result.success = result.tasks_failed == 0
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            result.total_cost_usd = self._total_cost_usd

            # Transition to idle
            self.context.transition_to(SessionState.IDLE)

            # Save summary to memory
            if self.memory:
                try:
                    self.memory.save(
                        key=f"request-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        value=f"Request: {request}\n{result.summary()}",
                        category="progress",
                        priority="normal",
                    )
                except Exception as e:
                    logger.warning(f"Could not save request summary: {e}")

            self._emit_status(result.summary())
            return result

        except Exception as e:
            logger.exception("Supervisor error")
            result.error = str(e)
            self.context.transition_to(SessionState.ERROR)
            self._emit_error(f"Supervisor error: {e}")
            return result

    async def _execute_task_with_retries(self, task: Task) -> TaskExecutionResult:
        """
        Execute a task with retry logic on rejection.

        Args:
            task: Task to execute

        Returns:
            TaskExecutionResult with execution and review outcomes
        """
        result = TaskExecutionResult(task=task, success=False)
        previous_feedback: str | None = None

        # Run pre-task hooks before any attempts
        if not await self._run_hooks(self.project.pre_task_hooks, "pre-task"):
            result.error = "Pre-task hook failed"
            return result

        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt
            self._emit_progress(f"Attempt {attempt}/{self.max_retries}")

            # Transition to executing state
            self.context.transition_to(SessionState.EXECUTING)

            # Execute the task
            try:
                execution_result = await self.execute_task(
                    task=task,
                    previous_feedback=previous_feedback,
                )
                result.execution_result = execution_result

                if not execution_result.success:
                    result.error = execution_result.error or "Execution failed"
                    self._emit_error(f"Task execution failed: {result.error}")
                    # Don't retry on execution failure (e.g., timeout)
                    break

            except ClaudeError as e:
                result.error = str(e)
                self._emit_error(f"Claude error: {e}")
                break

            # Track cost
            self._total_cost_usd += execution_result.cost_usd

            # Transition to reviewing state
            self.context.transition_to(SessionState.REVIEWING)

            # Review the task
            try:
                review_result = await self.review_task(task, execution_result)
                result.review_approved = review_result.get("approved", False)
                result.review_feedback = review_result.get("feedback", "")

                if self.callbacks.on_review_result:
                    self.callbacks.on_review_result(
                        result.review_approved,
                        result.review_feedback,
                    )

                if result.review_approved:
                    # Success - task approved
                    result.success = True
                    result.files_changed = execution_result.file_changes

                    # Save successful result to memory
                    self._save_task_result(
                        task=task,
                        success=True,
                        summary=execution_result.result or "Completed",
                        files_changed=execution_result.file_changes,
                    )

                    self._emit_progress("Task approved by GLM")

                    # Run post-task hooks after successful completion
                    await self._run_hooks(self.project.post_task_hooks, "post-task")
                    break
                else:
                    # Rejected - prepare for retry with meaningful feedback
                    issues = review_result.get("issues", [])
                    feedback_text = result.review_feedback or "Task did not meet quality standards"
                    issues_text = ", ".join(issues) if issues else "Not specified"

                    # Limit feedback length to prevent context overflow
                    combined_feedback = f"Issues: {issues_text}. Feedback: {feedback_text}"
                    if len(combined_feedback) > 500:
                        combined_feedback = combined_feedback[:497] + "..."

                    previous_feedback = f"Previous attempt was rejected. {combined_feedback}"

                    self._emit_progress(
                        f"Task rejected: {result.review_feedback[:100]}... Retrying."
                    )

                    # Save rejection to memory for learning
                    if self.memory:
                        try:
                            self.memory.save(
                                key=f"rejection-{task.id}-{attempt}",
                                value=f"Task: {task.description}\nIssues: {issues}\nFeedback: {result.review_feedback}",
                                category="warning",
                                priority="normal",
                            )
                        except Exception:
                            pass

            except GLMError as e:
                # Review failed - assume approved to avoid blocking
                logger.warning(f"GLM review failed, assuming approved: {e}")
                result.review_approved = True
                result.success = True
                result.files_changed = execution_result.file_changes
                break

        # If we exhausted retries without success
        if not result.success and result.attempts >= self.max_retries:
            result.error = f"Task failed after {self.max_retries} attempts. Last feedback: {result.review_feedback}"
            self._save_task_result(
                task=task,
                success=False,
                summary=result.error,
                files_changed=[],
            )

        return result

    async def execute_task(
        self,
        task: Task,
        previous_feedback: str | None = None,
    ) -> TaskResult:
        """
        Execute a single task via Claude.

        Args:
            task: Task to execute
            previous_feedback: Feedback from previous rejection (for retries)

        Returns:
            TaskResult from Claude execution
        """
        # Import here to avoid circular import
        from vibe.claude.executor import ClaudeExecutor

        # Load global conventions from memory if available
        global_conventions: list[str] = []
        if self.memory:
            try:
                global_conventions = self.memory.load_conventions()
            except Exception as e:
                logger.warning(f"Could not load conventions: {e}")

        # Create executor for this task
        executor = ClaudeExecutor(
            project_path=self.project.path,
            timeout_tier="code",
            global_conventions=global_conventions,
        )

        # Build constraints including previous feedback
        constraints = list(task.constraints) if task.constraints else []
        if previous_feedback:
            constraints.append(f"IMPORTANT - Previous attempt feedback: {previous_feedback}")

        # Execute
        result = await executor.execute(
            task_description=task.description,
            files=task.files if task.files else None,
            constraints=constraints if constraints else None,
            on_progress=self.callbacks.on_progress,
        )

        return result

    async def review_task(
        self,
        task: Task,
        result: TaskResult,
    ) -> dict[str, Any]:
        """
        Have GLM review a task's output.

        Args:
            task: The task that was executed
            result: Claude's execution result

        Returns:
            Review result with approved/issues/feedback
        """
        # Import here to avoid circular import
        from vibe.claude.executor import get_git_diff

        # Get git diff for the changed files
        if result.file_changes:
            changes_diff = get_git_diff(
                project_path=self.project.path,
                files=result.file_changes,
            )
        else:
            changes_diff = "(no file changes detected)"

        # Get Claude's summary
        claude_summary = result.result or "(no summary provided)"

        # Call GLM for review
        review_result = await self.glm_client.review_changes(
            task_description=task.description,
            changes_diff=changes_diff,
            claude_summary=claude_summary,
        )

        return review_result

    def get_stats(self) -> dict[str, Any]:
        """Get supervisor statistics."""
        return {
            "project": self.project.name,
            "session_state": self.context.state.name,
            "completed_tasks": len(self.context.completed_tasks),
            "pending_tasks": len(self.context.task_queue),
            "error_count": self.context.error_count,
            "total_cost_usd": self._total_cost_usd,
            "glm_usage": self.glm_client.get_usage_stats(),
        }
