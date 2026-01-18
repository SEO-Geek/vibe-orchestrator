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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.claude.circuit import CircuitBreaker
from vibe.config import Project
from vibe.exceptions import (
    ClaudeCircuitOpenError,
    ClaudeError,
    GLMError,
)
from vibe.glm.client import GLMClient
from vibe.memory.keeper import VibeMemory
from vibe.memory.pattern_learning import get_pattern_learner
from vibe.orchestrator.task_board import get_task_board
from vibe.orchestrator.task_routing import TaskRouter, get_task_router
from vibe.state import SessionContext, SessionState, Task

# TYPE_CHECKING imports are only evaluated by type checkers (mypy, pyright),
# not at runtime. This breaks the circular dependency: supervisor -> executor -> supervisor
if TYPE_CHECKING:
    from vibe.claude.executor import ClaudeExecutor, TaskResult
    from vibe.gemini.client import GeminiClient
    from vibe.orchestrator.reviewer import Reviewer

logger = logging.getLogger(__name__)


# =============================================================================
# MCP ROUTING TABLE
# Maps task types to recommended MCP tools for Claude to use.
# This table guides Claude toward the RIGHT tools for each task type,
# preventing common mistakes like using curl for UI testing instead of Playwright.
# =============================================================================

MCP_ROUTING_TABLE: dict[str, dict[str, list[str]]] = {
    "debug": {
        "recommended": [
            "mcp__chrome-devtools__list_console_messages",
            "mcp__chrome-devtools__list_network_requests",
            "mcp__sequential-thinking__sequentialthinking",
        ],
        "hint": (
            "Use Chrome DevTools for browser debugging, "
            "sequential-thinking for complex analysis"
        ),
    },
    "code_write": {
        "recommended": [
            "mcp__context7__resolve-library-id",
            "mcp__context7__query-docs",
            "mcp__github__search_code",
        ],
        "hint": "Use context7 for up-to-date library documentation before implementing",
    },
    "ui_test": {
        "recommended": [
            "mcp__playwright__playwright_navigate",
            "mcp__playwright__playwright_screenshot",
            "mcp__playwright__playwright_click",
            "mcp__chrome-devtools__take_screenshot",
        ],
        "hint": "MUST use Playwright or Chrome DevTools for browser testing - never curl",
    },
    "research": {
        "recommended": [
            "mcp__perplexity__search",
            "mcp__perplexity__reason",
            "mcp__context7__query-docs",
        ],
        "hint": "Use Perplexity for web research, context7 for library docs",
    },
    "code_refactor": {
        "recommended": [
            "mcp__git__git_status",
            "mcp__git__git_diff",
            "mcp__memory-keeper__context_save",
        ],
        "hint": "Save state before refactoring, verify with git diff after",
    },
    "database": {
        "recommended": [
            "mcp__sqlite__read_query",
            "mcp__sqlite__list_tables",
        ],
        "hint": "Use SQLite tools for database operations",
    },
}

# Maximum retry attempts for rejected tasks.
# 3 attempts balances giving Claude a fair chance vs wasting time on impossible tasks.
MAX_RETRY_ATTEMPTS = 3

# Retry backoff delays (seconds) - exponential backoff.
# Progressive delays prevent hammering the API and allow transient issues to resolve.
RETRY_BACKOFF_DELAYS = [2.0, 5.0, 10.0]

# GLM API timeout for review calls (seconds)
GLM_REVIEW_TIMEOUT = 60.0

# Token budget for GLM context (approximate).
# GLM-4 has ~128K context, but we reserve most for the response and actual task content.
# Using 25% (32K tokens) for project context ensures room for:
# - Task decomposition output
# - Review reasoning
# - Error messages and feedback
# The 4 chars/token ratio is conservative; actual ratio varies by language.
MAX_CONTEXT_TOKENS = 32000
CHARS_PER_TOKEN = 4
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN


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

    Architecture:
        User → Gemini (brain/orchestrator) → Claude (worker)
                        ↓                        ↓
                     GLM (code review only)

    Responsibilities:
    - Receive user requests
    - Ask GEMINI to decompose into tasks (brain)
    - Execute tasks via Claude (worker)
    - Send output to GLM for code review (verifier)
    - Accept or reject based on GLM review
    - Handle retries with feedback
    - Persist state to memory
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        glm_client: GLMClient,
        project: Project,
        memory: VibeMemory | None = None,
        callbacks: SupervisorCallbacks | None = None,
        max_retries: int = MAX_RETRY_ATTEMPTS,
        use_workflow_engine: bool | None = None,
        use_circuit_breaker: bool = True,
        task_router: TaskRouter | None = None,
    ):
        """
        Initialize supervisor.

        Args:
            gemini_client: GeminiClient for task decomposition (the brain)
            glm_client: GLMClient for code review only (the verifier)
            project: Project configuration with path and settings
            memory: Optional VibeMemory for persistence
            callbacks: Optional callbacks for progress updates
            max_retries: Maximum retry attempts for rejected tasks
            use_workflow_engine: Enable workflow phase expansion (default: from project config)
            use_circuit_breaker: Enable circuit breaker for Claude calls (default: True)
            task_router: Optional TaskRouter for task-type-aware configuration
        """
        self.gemini_client = gemini_client
        self.glm_client = glm_client  # Only for code review
        self.project = project
        self.memory = memory
        self.callbacks = callbacks or SupervisorCallbacks()
        self.max_retries = max_retries

        # Task router for intelligent per-task-type configuration
        # Determines timeout, review requirements, retries based on task type
        self.task_router = task_router or get_task_router()

        # Task board for kanban-style visualization
        # Tracks tasks as they move through: BACKLOG -> IN_PROGRESS -> REVIEW -> DONE/REJECTED
        self.task_board = get_task_board(project.name)

        # Pattern learner for continuous improvement
        # Learns from successful tasks and warns about common failures
        self.pattern_learner = get_pattern_learner(project.name)

        # Workflow engine setting - default from project config
        self.use_workflow_engine = (
            use_workflow_engine if use_workflow_engine is not None else project.use_workflows
        )

        # Initialize session context
        self.context = SessionContext(
            project_name=project.name,
            project_path=project.path,
        )

        # Claude executor will be created per-task with appropriate settings
        self._executor: ClaudeExecutor | None = None

        # Circuit breaker prevents cascading failures when Claude is unhealthy.
        # Pattern: After N consecutive failures, "open" the circuit and fail fast
        # instead of wasting time on doomed requests. After reset_timeout,
        # allow one "probe" request to check if Claude recovered.
        self._circuit_breaker: CircuitBreaker | None = None
        if use_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=3,
                reset_timeout=60.0,
            )

        # Reviewer reference for cleanup (set externally if using shared reviewer)
        self._reviewer: Reviewer | None = None

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

    def _is_investigation_task(self, request: str) -> bool:
        """
        Check if a request is an investigation/research task.

        Investigation tasks don't need clarification because:
        1. They're exploratory by nature (no wrong interpretation)
        2. Claude will report findings regardless of exact interpretation
        3. Asking clarification adds ~5-10s latency for no benefit

        Args:
            request: User request text

        Returns:
            True if this is an investigation/research task
        """
        import re

        request_lower = request.lower().strip()

        # Question words at START of request indicate information-seeking, not action.
        # Anchored patterns (^) ensure we only match when these are the request intent.
        question_patterns = [
            r"^what\s",  # "what does X do"
            r"^how\s",  # "how does X work"
            r"^why\s",  # "why is X happening"
            r"^where\s",  # "where is X defined"
            r"^which\s",  # "which files contain X"
            r"^who\s",  # "who wrote X"
            r"^when\s",  # "when was X changed"
            r"^can\s+you\s+(find|show|explain|tell|describe)",
            r"^please\s+(find|show|explain|tell|describe|investigate)",
        ]

        # Investigation keywords anywhere in request (word boundaries prevent false matches).
        # These verbs indicate read-only operations that don't need clarification.
        investigation_keywords = [
            r"\b(find|search|locate|look\s+for)\b",
            r"\b(investigate|analyze|examine|inspect|explore)\b",
            r"\b(explain|describe|show\s+me|tell\s+me)\b",
            r"\b(what\s+is|what\s+are|what\s+does)\b",
            r"\b(how\s+does|how\s+to|how\s+can)\b",
            r"\b(list|enumerate|identify)\s+(all|the|any)\b",
            r"\b(check|verify|confirm)\s+(if|whether)\b",
            r"\breadme|docs|documentation\b",
            r"\bunderstand|figure\s+out\b",
        ]

        # Check question patterns
        for pattern in question_patterns:
            if re.search(pattern, request_lower):
                logger.debug(f"Investigation task detected (question pattern): {request[:50]}")
                return True

        # Check investigation keywords
        for pattern in investigation_keywords:
            if re.search(pattern, request_lower):
                logger.debug(f"Investigation task detected (keyword): {request[:50]}")
                return True

        # Requests that end with "?" are questions
        if request.strip().endswith("?"):
            logger.debug(f"Investigation task detected (question mark): {request[:50]}")
            return True

        return False

    def _might_produce_large_diff(self, description: str) -> bool:
        """
        Heuristic check for tasks that might produce large changes.

        Warns about tasks that commonly produce large diffs so users can
        consider breaking them into smaller units.

        Args:
            description: Task description to check

        Returns:
            True if the task might produce a large diff
        """
        import re

        # Patterns that historically cause huge diffs that exceed review limits.
        # When diff is truncated, GLM can't verify all changes - risky.
        large_task_patterns = [
            r"refactor\s+(entire|all|whole)",
            r"rename\s+.+\s+across",
            r"update\s+all\s+",
            r"format\s+(entire|all)",
            r"migrate\s+",
            r"rewrite\s+(entire|all|whole)",
            r"add\s+.+\s+to\s+(all|every)",
            r"remove\s+.+\s+from\s+(all|every)",
        ]
        description_lower = description.lower()
        return any(re.search(p, description_lower) for p in large_task_patterns)

    def _get_mcp_hints(self, task_description: str) -> str:
        """
        Get MCP tool hints based on detected task type.

        Uses SmartTaskDetector to determine task type and returns
        appropriate MCP recommendations from the routing table.

        Args:
            task_description: The task description

        Returns:
            Formatted string with MCP hints, or empty string if none
        """
        try:
            from vibe.orchestrator.task_enforcer import get_smart_detector

            # SmartTaskDetector uses keyword analysis to classify tasks.
            # This avoids calling GLM just to determine task type.
            detector = get_smart_detector()
            detection = detector.detect(task_description)

            # Get task type name (lowercase for routing table lookup)
            task_type_name = detection.task_type.name.lower()

            # Look up in routing table
            if task_type_name in MCP_ROUTING_TABLE:
                routing = MCP_ROUTING_TABLE[task_type_name]
                hint = routing.get("hint", "")
                recommended = routing.get("recommended", [])

                if hint or recommended:
                    # Format as markdown section that Claude will parse as instructions.
                    # Limiting to 3 tools prevents overwhelming Claude with options.
                    parts = ["\n## MCP TOOL RECOMMENDATIONS"]
                    if hint:
                        parts.append(f"HINT: {hint}")
                    if recommended:
                        tools_str = ", ".join(recommended[:3])
                        parts.append(f"Consider using: {tools_str}")
                    return "\n".join(parts)

        except Exception as e:
            logger.debug(f"Could not get MCP hints: {e}")

        return ""

    def _expand_tasks_with_workflow(
        self,
        task_dicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Expand tasks using WorkflowEngine if enabled.

        Converts simple tasks into multi-phase workflows based on
        detected task type (e.g., DEBUG -> reproduce, investigate, fix, verify).

        Why expand? Single "fix bug" tasks often fail because Claude rushes
        to a solution. Multi-phase forces a methodical approach:
        1. Reproduce (confirm the bug exists)
        2. Investigate (find root cause)
        3. Fix (implement solution)
        4. Verify (confirm fix works)

        Args:
            task_dicts: List of task dictionaries from GLM

        Returns:
            Expanded list of tasks (or original if workflow disabled)
        """
        if not self.use_workflow_engine:
            return task_dicts

        try:
            from vibe.orchestrator.workflows import WorkflowEngine

            engine = WorkflowEngine(
                enable_workflows=True,
                enable_injection=self.project.inject_subtasks,
            )

            # expand_to_phases=True triggers the task type detection and phase injection
            expanded = engine.process_tasks(task_dicts, expand_to_phases=True)

            # Convert ExpandedTask objects back to dicts
            expanded_dicts = [t.to_dict() for t in expanded]

            if len(expanded_dicts) != len(task_dicts):
                self._emit_progress(
                    f"WorkflowEngine expanded {len(task_dicts)} tasks "
                    f"to {len(expanded_dicts)} phases"
                )

            return expanded_dicts

        except ImportError as e:
            logger.warning(f"WorkflowEngine not available: {e}")
            return task_dicts
        except Exception as e:
            logger.warning(f"WorkflowEngine expansion failed: {e}")
            return task_dicts

    async def _run_hooks(self, hooks: list[str], phase: str) -> bool:
        """
        Run hook scripts before/after task execution.

        Hooks enable project-specific automation:
        - pre-task: backup state, validate environment, run linters
        - post-task: run tests, update docs, notify systems

        Args:
            hooks: List of hook script paths (relative to project directory)
            phase: Phase name for logging ('pre-task' or 'post-task')

        Returns:
            True if all hooks succeeded, False if any failed
        """
        import os

        if not hooks:
            return True

        # resolve() normalizes the path and follows symlinks for consistent comparison
        project_path = Path(self.project.path).resolve()

        for hook in hooks:
            # Resolve to absolute path (handles ../ and symlinks)
            hook_path = (project_path / hook).resolve()

            # SECURITY: Prevent path traversal attacks (e.g., "../../../etc/passwd")
            # relative_to() raises ValueError if hook_path is outside project_path
            try:
                hook_path.relative_to(project_path)
            except ValueError:
                self._emit_error(f"{phase} hook path traversal detected: {hook}")
                return False

            # Verify hook exists and has execute permissions
            if not hook_path.is_file():
                self._emit_error(f"{phase} hook not found: {hook}")
                return False

            # os.X_OK checks execute permission for current user
            if not os.access(hook_path, os.X_OK):
                self._emit_error(f"{phase} hook not executable: {hook}")
                return False

            self._emit_progress(f"Running {phase} hook: {hook}")
            try:
                # create_subprocess_exec avoids shell injection vulnerabilities.
                # Unlike shell=True, it doesn't interpret special characters.
                proc = await asyncio.create_subprocess_exec(
                    str(hook_path),
                    cwd=self.project.path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Timeout prevents a hung hook from blocking the entire pipeline.
                # 60s is generous - hooks should do quick validation, not long operations.
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=60.0,
                )

                if proc.returncode != 0:
                    error_output = (
                        stderr.decode("utf-8", errors="replace")
                        if stderr
                        else stdout.decode("utf-8", errors="replace")
                    )
                    self._emit_error(f"{phase} hook failed: {hook}\n{error_output[:200]}")
                    return False

                if stdout:
                    self._emit_progress(
                        f"Hook output: {stdout.decode('utf-8', errors='replace')[:100]}"
                    )

            except TimeoutError:
                self._emit_error(f"{phase} hook timed out: {hook}")
                return False
            except Exception as e:
                self._emit_error(f"{phase} hook error: {hook}: {e}")
                return False

        return True

    def _load_project_context(self) -> str:
        """
        Load project context for GLM including starmap and recent memory.

        Context prioritization (most important first):
        1. STARMAP.md - project architecture and goals (4K chars)
        2. CLAUDE.md - coding conventions (2K chars)
        3. Recent memory - what we've been working on (3K chars)
        4. Project metadata - name, path, test command

        Enforces token budget and truncates content if needed.

        Returns:
            Formatted context string for GLM prompts
        """
        context_parts = []
        total_chars = 0

        # Closure captures total_chars for cumulative tracking across calls
        def add_context(label: str, content: str, max_chars: int) -> None:
            nonlocal total_chars
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... [truncated]"
            context_parts.append(f"{label}:\n{content}")
            total_chars += len(content)

        # Load STARMAP.md if it exists
        starmap_path = self.project.starmap_path
        if starmap_path.exists():
            try:
                starmap_content = starmap_path.read_text()
                add_context("PROJECT STARMAP", starmap_content, 4000)
            except Exception as e:
                logger.warning(f"Could not read STARMAP.md: {e}")

        # Load CLAUDE.md if it exists (project-specific conventions)
        claude_md_path = self.project.claude_md_path
        if claude_md_path.exists():
            try:
                claude_content = claude_md_path.read_text()
                add_context("PROJECT CONVENTIONS", claude_content, 2000)
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
                    add_context("RECENT CONTEXT", memory_context, 3000)
            except Exception as e:
                logger.warning(f"Could not load memory context: {e}")

        # Add project metadata
        metadata = (
            f"PROJECT: {self.project.name}\n"
            f"PATH: {self.project.path}\n"
            f"TEST COMMAND: {self.project.test_command}"
        )
        context_parts.append(metadata)
        total_chars += len(metadata)

        # Check token budget and warn if exceeded
        estimated_tokens = total_chars // CHARS_PER_TOKEN
        if total_chars > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Context size ({estimated_tokens} tokens) exceeds budget "
                f"({MAX_CONTEXT_TOKENS} tokens). Consider reducing context."
            )
        else:
            logger.debug(
                f"Context loaded: ~{estimated_tokens} tokens "
                f"({total_chars} chars, {100 * total_chars // MAX_CONTEXT_CHARS}% of budget)"
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
            # Transition to planning state (must succeed from IDLE/AWAITING_INPUT)
            self.context.require_transition(SessionState.PLANNING)
            self._emit_status("Loading project context...")

            # Step 1: Load project context
            project_context = self._load_project_context()

            # Step 2: Ask for clarification if needed.
            # Decision tree:
            # - skip_clarification=True (CLI flag) -> skip
            # - Investigation task (questions, research) -> skip (no ambiguity)
            # - Everything else -> ask GEMINI if request is ambiguous
            should_ask_clarification = not skip_clarification and not self._is_investigation_task(
                request
            )

            if should_ask_clarification:
                self._emit_status("Checking if clarification needed...")
                try:
                    # Gemini (the brain) decides if clarification is needed
                    clarification_result = await self.gemini_client.check_clarification(
                        user_request=request,
                        project_context=project_context,
                    )

                    if clarification_result.get("needs_clarification"):
                        # Gemini needs more info - return early.
                        # This is success=True because the system worked correctly;
                        # we just need user input before proceeding.
                        result.clarification_asked = clarification_result.get(
                            "question", "Please clarify your request."
                        )
                        result.success = True
                        self.context.transition_to(SessionState.IDLE)
                        return result

                except Exception as e:
                    self._emit_error(f"Gemini clarification failed: {e}")
                    # Continue without clarification rather than fail
            elif not skip_clarification:
                # Investigation task - log that we're skipping
                self._emit_progress("Investigation task detected - skipping clarification step")

            # Step 3: Decompose task - GEMINI is the brain that plans
            self._emit_status("Gemini decomposing request into tasks...")

            # Get historical patterns for adaptive decomposition
            # This allows Gemini to learn from past successes/failures
            pattern_context = ""
            try:
                pattern_context = self.pattern_learner.get_decomposition_hints(request)
                if pattern_context and pattern_context != "(No historical patterns yet)":
                    logger.debug(f"Injecting {len(pattern_context)} chars of pattern context to Gemini")
            except Exception as e:
                logger.debug(f"Could not get decomposition hints: {e}")

            try:
                task_dicts = await self.gemini_client.decompose_task(
                    user_request=request,
                    project_context=project_context,
                    pattern_context=pattern_context,
                )
            except Exception as e:
                result.error = f"Task decomposition failed: {e}"
                self._emit_error(result.error)
                self.context.transition_to(SessionState.ERROR)
                return result

            if not task_dicts:
                result.error = "Gemini returned no tasks"
                self._emit_error(result.error)
                self.context.transition_to(SessionState.ERROR)
                return result

            # WorkflowEngine transforms simple tasks into multi-phase workflows.
            # e.g., "fix login bug" -> [reproduce, investigate, fix, verify]
            task_dicts = self._expand_tasks_with_workflow(task_dicts)

            # Convert raw dicts to Task objects for type safety and queue management
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

                # Add task to kanban board (BACKLOG)
                try:
                    self.task_board.add_task(
                        task_id=task.id,
                        description=task.description,
                        session_id=self.context.session_id,
                        metadata={"files": task.files, "constraints": task.constraints},
                    )
                except ValueError as e:
                    logger.warning(f"Could not add task to board: {e}")

            result.total_tasks = len(tasks)
            self._emit_status(f"Decomposed into {len(tasks)} tasks")

            # Checkpoint enables rollback if task execution goes wrong.
            # Creates git stash + memory snapshot for recovery.
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
                    self.context.complete_current_task(
                        {
                            "success": True,
                            "files": task_result.files_changed,
                        }
                    )
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

        Uses TaskRouter to determine task-type-specific settings:
        - Max retries (fewer for delicate operations like refactoring)
        - Review requirements (skip for research/investigation tasks)
        - Test requirements (run after code changes)

        Args:
            task: Task to execute

        Returns:
            TaskExecutionResult with execution and review outcomes
        """
        result = TaskExecutionResult(task=task, success=False)
        previous_feedback: str | None = None

        # Get task-type-specific routing configuration
        routing_config = self.task_router.get_config(task.description)
        task_max_retries = min(routing_config.max_retries, self.max_retries)

        self._emit_progress(
            f"Task type: {routing_config.task_type.value} "
            f"(timeout={routing_config.get_timeout()}s, retries={task_max_retries})"
        )

        # Pre-check: Warn about tasks that might produce large diffs
        if self._might_produce_large_diff(task.description):
            logger.warning(
                f"Task '{task.description[:60]}...' may produce large changes. "
                f"Consider splitting into smaller tasks to ensure thorough review."
            )
            self._emit_progress(
                "Note: Large-scope task detected. Review may be partial if diff exceeds limit."
            )

        # Run pre-task hooks before any attempts
        if not await self._run_hooks(self.project.pre_task_hooks, "pre-task"):
            result.error = "Pre-task hook failed"
            return result

        for attempt in range(1, task_max_retries + 1):
            result.attempts = attempt
            self._emit_progress(f"Attempt {attempt}/{task_max_retries}")

            # Move task to IN_PROGRESS on kanban board
            try:
                self.task_board.start_task(task.id)
            except Exception as e:
                logger.debug(f"Could not update board: {e}")

            # Transition to executing state (must succeed)
            self.context.require_transition(SessionState.EXECUTING)

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

            except ClaudeCircuitOpenError as e:
                # Circuit open = Claude has failed too many times recently.
                # Fail fast instead of wasting time on likely-to-fail requests.
                result.error = f"Circuit breaker open: {e}"
                self._emit_error(f"Circuit breaker prevented execution: {e}")
                break

            except ClaudeError as e:
                result.error = str(e)
                self._emit_error(f"Claude error: {e}")
                break

            # Track cost
            self._total_cost_usd += execution_result.cost_usd

            # State machine enforces valid transitions: IDLE -> PLANNING -> EXECUTING -> REVIEWING
            self.context.require_transition(SessionState.REVIEWING)

            # Move task to REVIEW column on kanban board
            try:
                self.task_board.send_to_review(task.id)
            except Exception as e:
                logger.debug(f"Could not update board for review: {e}")

            # Smart review skipping based on task type and file changes.
            # Uses TaskRouter to determine if this task type requires review.
            # Research/investigation tasks skip review; code changes require it.
            should_skip_review = self.task_router.should_skip_review(
                task.description,
                has_file_changes=bool(execution_result.file_changes),
            )

            if should_skip_review:
                skip_reason = (
                    "Task type doesn't require review"
                    if routing_config.require_review is False
                    else "No file changes to review"
                )
                self._emit_progress(f"Skipping review: {skip_reason}")
                result.review_approved = True
                result.review_feedback = f"Auto-approved: {skip_reason}"
                result.success = True
                result.files_changed = execution_result.file_changes

                # Move task to DONE column on kanban board (auto-approved)
                try:
                    self.task_board.complete_task(
                        task.id,
                        files_changed=execution_result.file_changes,
                    )
                except Exception as e:
                    logger.debug(f"Could not update board for auto-approval: {e}")

                # Record success pattern for learning
                try:
                    self.pattern_learner.record_success(
                        task_description=task.description,
                        task_type=routing_config.task_type,
                        tools_used=execution_result.tool_calls if hasattr(execution_result, 'tool_calls') else [],
                        duration_seconds=0,  # Not tracked for auto-approved
                        cost_usd=execution_result.cost_usd,
                    )
                except Exception as e:
                    logger.debug(f"Could not record success pattern: {e}")

                if self.callbacks.on_review_result:
                    self.callbacks.on_review_result(True, result.review_feedback)

                # Save successful result to memory
                self._save_task_result(
                    task=task,
                    success=True,
                    summary=execution_result.result or "Completed",
                    files_changed=execution_result.file_changes,
                )

                # Run post-task hooks after successful completion
                await self._run_hooks(self.project.post_task_hooks, "post-task")

                # Run tests if configured for this task type
                if self.task_router.should_run_tests(task.description, task_failed=False):
                    self._emit_progress("Running tests after task completion...")
                    await self._run_project_tests()

                break

            # Review the task with GLM
            # Pass task_type for task-type-specific review criteria
            try:
                review_result = await self.review_task(
                    task, execution_result, task_type=routing_config.task_type.value
                )
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

                    # Move task to DONE column on kanban board
                    try:
                        self.task_board.complete_task(
                            task.id,
                            files_changed=execution_result.file_changes,
                        )
                    except Exception as e:
                        logger.debug(f"Could not update board for completion: {e}")

                    # Record success pattern for learning
                    try:
                        self.pattern_learner.record_success(
                            task_description=task.description,
                            task_type=routing_config.task_type,
                            tools_used=execution_result.tool_calls if hasattr(execution_result, 'tool_calls') else [],
                            duration_seconds=0,  # Could track if needed
                            cost_usd=execution_result.cost_usd,
                        )
                    except Exception as e:
                        logger.debug(f"Could not record success pattern: {e}")

                    # Run post-task hooks after successful completion
                    await self._run_hooks(self.project.post_task_hooks, "post-task")

                    # Run tests if configured for this task type
                    if self.task_router.should_run_tests(task.description, task_failed=False):
                        self._emit_progress("Running tests after task completion...")
                        await self._run_project_tests()

                    break
                else:
                    # Rejected - prepare retry with GLM's feedback as guidance.
                    # This feedback goes into Claude's next prompt so it knows what to fix.
                    issues = review_result.get("issues", [])
                    feedback_text = result.review_feedback or "Task did not meet quality standards"
                    issues_text = ", ".join(issues) if issues else "Not specified"

                    # Truncate to 500 chars to avoid bloating Claude's context.
                    # Long feedback can crowd out actual task instructions.
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
                            rejection_value = (
                                f"Task: {task.description}\n"
                                f"Issues: {issues}\n"
                                f"Feedback: {result.review_feedback}"
                            )
                            self.memory.save(
                                key=f"rejection-{task.id}-{attempt}",
                                value=rejection_value,
                                category="warning",
                                priority="normal",
                            )
                        except Exception:
                            pass

                    # Exponential backoff: 2s -> 5s -> 10s
                    # Gives transient issues time to resolve; prevents rate limiting.
                    if attempt < self.max_retries:
                        backoff_idx = min(attempt - 1, len(RETRY_BACKOFF_DELAYS) - 1)
                        backoff_delay = RETRY_BACKOFF_DELAYS[backoff_idx]
                        self._emit_progress(f"Waiting {backoff_delay}s before retry...")
                        await asyncio.sleep(backoff_delay)

            except TimeoutError:
                # CRITICAL: Never auto-approve on timeout. Unreviewed code could
                # contain bugs, security issues, or unintended changes.
                logger.error(f"GLM review timed out after {GLM_REVIEW_TIMEOUT}s")
                result.error = f"GLM review timed out after {GLM_REVIEW_TIMEOUT}s"
                result.review_approved = False
                self._emit_error("Review timeout - cannot approve without review")
                break

            except GLMError as e:
                # CRITICAL: Never auto-approve on GLM failure. Same reasoning as
                # timeout - we must verify changes before accepting them.
                logger.error(f"GLM review failed: {e}")
                result.error = f"Cannot approve without review: {e}"
                result.review_approved = False
                self._emit_error(f"Review failed - cannot approve: {e}")
                break

        # If we exhausted retries without success
        if not result.success and result.attempts >= task_max_retries:
            result.error = (
                f"Task failed after {task_max_retries} attempts. "
                f"Last feedback: {result.review_feedback}"
            )

            # Move task to REJECTED column on kanban board
            try:
                self.task_board.reject_task(task.id, result.review_feedback)
            except Exception as e:
                logger.debug(f"Could not update board for rejection: {e}")

            # Record failure pattern for learning
            try:
                self.pattern_learner.record_failure(
                    task_description=task.description,
                    task_type=routing_config.task_type,
                    feedback=result.review_feedback,
                )
            except Exception as e:
                logger.debug(f"Could not record failure pattern: {e}")

            self._save_task_result(
                task=task,
                success=False,
                summary=result.error,
                files_changed=[],
            )

        # Reviewer tracks per-task state (review history, rejection count).
        # Must clean up to prevent memory growth in long-running sessions.
        if self._reviewer:
            try:
                self._reviewer.cleanup_completed_task(task.id)
            except Exception as e:
                logger.debug(f"Reviewer cleanup failed: {e}")

        return result

    async def execute_task(
        self,
        task: Task,
        previous_feedback: str | None = None,
    ) -> TaskResult:
        """
        Execute a single task via Claude.

        Uses circuit breaker to prevent cascading failures and adds
        MCP tool hints based on detected task type.

        Args:
            task: Task to execute
            previous_feedback: Feedback from previous rejection (for retries)

        Returns:
            TaskResult from Claude execution

        Raises:
            ClaudeCircuitOpenError: If circuit breaker is open
        """
        # Import here to avoid circular import
        from vibe.claude.executor import ClaudeExecutor

        # Check circuit breaker before attempting execution
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise ClaudeCircuitOpenError(
                "Circuit breaker is open - too many consecutive failures",
                failures=self._circuit_breaker.stats.failed_calls,
                reset_time=self._circuit_breaker.reset_timeout,
            )

        # Load global conventions from memory if available
        global_conventions: list[str] = []
        if self.memory:
            try:
                global_conventions = self.memory.load_conventions()
            except Exception as e:
                logger.warning(f"Could not load conventions: {e}")

        # Build constraints including previous feedback and MCP hints
        constraints = list(task.constraints) if task.constraints else []
        if previous_feedback:
            constraints.append(f"IMPORTANT - Previous attempt feedback: {previous_feedback}")

        # Add MCP tool hints based on task type
        mcp_hints = self._get_mcp_hints(task.description)
        if mcp_hints:
            constraints.append(mcp_hints)

        # Inject learned patterns and warnings from previous tasks
        # This helps Claude avoid repeating past mistakes and use proven approaches
        from vibe.orchestrator.task_enforcer import get_smart_detector

        try:
            detector = get_smart_detector()
            detection = detector.detect(task.description)
            learnings = self.pattern_learner.generate_learnings_context(detection.task_type)
            if learnings:
                constraints.append(learnings)
        except Exception as e:
            logger.debug(f"Could not generate learnings context: {e}")

        # async with ensures executor cleanup (subprocess termination, temp files)
        # even if execute() raises an exception or times out.
        async with ClaudeExecutor(
            project_path=self.project.path,
            timeout_tier="code",  # "code" tier = longer timeout for complex tasks
            global_conventions=global_conventions,
        ) as executor:
            # Check for previous checkpoint (recovery from timeout)
            # This allows Claude to continue from where it left off instead of starting over
            checkpoint = executor.load_checkpoint_from_disk(task.id)
            if checkpoint:
                logger.info(f"Found checkpoint for task {task.id}: {checkpoint.summary()}")
                checkpoint_context = (
                    f"CONTINUE FROM PREVIOUS WORK (task timed out after {checkpoint.elapsed_seconds:.1f}s):\n"
                    f"- Tool calls made: {len(checkpoint.tool_calls)}\n"
                    f"- Files modified: {', '.join(checkpoint.file_changes) if checkpoint.file_changes else 'none'}\n"
                    f"- Partial progress: {checkpoint.partial_output[:500]}..."
                    if len(checkpoint.partial_output) > 500
                    else f"- Partial progress: {checkpoint.partial_output}"
                )
                constraints.append(checkpoint_context)
                self._emit_progress(f"Resuming from checkpoint: {checkpoint.summary()}")

            # Execute with circuit breaker tracking
            try:
                result = await executor.execute(
                    task_description=task.description,
                    files=task.files if task.files else None,
                    constraints=constraints if constraints else None,
                    on_progress=self.callbacks.on_progress,
                    task_id=task.id,  # Pass task_id for checkpoint persistence
                )

                # Record success/failure in circuit breaker
                if self._circuit_breaker:
                    if result.success:
                        self._circuit_breaker.record_success()
                    else:
                        self._circuit_breaker.record_failure()

                # Clear checkpoint on success (no longer needed)
                if result.success:
                    executor.clear_checkpoint_from_disk(task.id)

                return result

            except Exception:
                # Record failure in circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise

    async def review_task(
        self,
        task: Task,
        result: TaskResult,
        task_type: str = "code_write",
    ) -> dict[str, Any]:
        """
        Have GLM review a task's output.

        Args:
            task: The task that was executed
            result: Claude's execution result
            task_type: Type of task for task-type-specific review criteria

        Returns:
            Review result with approved/issues/feedback
        """
        # Import here to avoid circular import
        from vibe.claude.executor import get_git_diff

        # Git diff is the primary evidence GLM uses for review.
        # We limit size to prevent context overflow, but this means large changes
        # may not be fully reviewed - a known tradeoff for performance.
        was_truncated = False
        if result.file_changes:
            changes_diff, was_truncated = get_git_diff(
                project_path=self.project.path,
                files=result.file_changes,
                max_chars=self.project.context_settings.max_diff_chars,
                exclude_patterns=self.project.context_settings.diff_exclude_patterns,
            )
            # Truncated diff = partial review. GLM can only verify what it sees.
            if was_truncated:
                logger.warning(
                    f"Diff truncated for review. GLM will only see first "
                    f"{self.project.context_settings.max_diff_chars:,} chars. "
                    f"Changed files: {result.file_changes}"
                )
        else:
            changes_diff = "(no file changes detected)"

        # Get Claude's summary
        claude_summary = result.result or "(no summary provided)"

        # Format files changed for GLM context
        files_changed = ", ".join(result.file_changes) if result.file_changes else ""

        # wait_for wraps the coroutine with a timeout. If GLM takes longer than
        # GLM_REVIEW_TIMEOUT, raises TimeoutError (handled by _execute_task_with_retries).
        review_result = await asyncio.wait_for(
            self.glm_client.review_changes(
                task_description=task.description,
                changes_diff=changes_diff,
                claude_summary=claude_summary,
                task_type=task_type,
                files_changed=files_changed,
            ),
            timeout=GLM_REVIEW_TIMEOUT,
        )

        return review_result

    async def _run_project_tests(self) -> bool:
        """
        Run project tests if configured.

        Uses the project's test_command setting.

        Returns:
            True if tests passed, False otherwise
        """
        if not self.project.test_command:
            return True

        try:
            proc = await asyncio.create_subprocess_shell(
                self.project.test_command,
                cwd=self.project.path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)

            if proc.returncode == 0:
                self._emit_progress("Tests passed")
                return True
            else:
                error_output = stderr.decode("utf-8", errors="replace")[:500]
                self._emit_error(f"Tests failed: {error_output}")
                return False

        except TimeoutError:
            self._emit_error("Test command timed out")
            return False
        except Exception as e:
            self._emit_error(f"Test execution error: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get supervisor statistics including circuit breaker status."""
        stats = {
            "project": self.project.name,
            "session_state": self.context.state.name,
            "completed_tasks": len(self.context.completed_tasks),
            "pending_tasks": len(self.context.task_queue),
            "error_count": self.context.error_count,
            "total_cost_usd": self._total_cost_usd,
            "glm_usage": self.glm_client.get_usage_stats(),
            "workflow_engine_enabled": self.use_workflow_engine,
        }

        # Add circuit breaker stats if enabled
        if self._circuit_breaker:
            cb_stats = self._circuit_breaker.stats
            stats["circuit_breaker"] = {
                "state": self._circuit_breaker.state.name,
                "total_calls": cb_stats.total_calls,
                "successful_calls": cb_stats.successful_calls,
                "failed_calls": cb_stats.failed_calls,
                "rejected_calls": cb_stats.rejected_calls,
            }

        return stats
