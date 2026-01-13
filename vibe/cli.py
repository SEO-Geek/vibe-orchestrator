#!/usr/bin/env python3
"""
Vibe Orchestrator - CLI Entry Point

Main entry point for the `vibe` command.
Shows startup validation, project selection, and conversation interface.
"""

import asyncio
import copy
import logging
import os
import shutil
import signal
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import NoReturn

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from vibe.config import (
    CONFIG_DIR,
    MEMORY_DB_PATH,
    Project,
    VibeConfig,
    get_openrouter_key,
    load_config,
    save_config,
)
from vibe.claude.executor import ClaudeExecutor, TaskResult, ToolCall, get_git_diff
from vibe.exceptions import ConfigError, ClaudeError, GLMConnectionError, MemoryConnectionError, StartupError, ResearchError, GitHubError
from vibe.integrations import PerplexityClient, GitHubOps
from vibe.glm.client import GLMClient, ping_glm_sync
from vibe.glm.prompts import SUPERVISOR_SYSTEM_PROMPT
from vibe.memory.keeper import VibeMemory
from vibe.memory.debug_session import DebugSession, DebugAttempt, AttemptResult
from vibe.memory.task_history import TaskHistory, add_task, add_request, get_context_for_glm
# New unified persistence layer
from vibe.persistence.repository import VibeRepository
from vibe.persistence.models import (
    SessionStatus, TaskStatus, AttemptResult as PersistAttemptResult,
    MessageRole, MessageType
)
from vibe.glm.debug_state import DebugContext, ClaudeIteration
from vibe.glm.prompts import DEBUG_CLAUDE_PROMPT
from vibe.logging import (
    session_logger,
    SessionLogEntry,
    now_iso,
    set_session_id,
    set_project_name,
    get_session_id,
)
from vibe.logging.viewer import query_logs, calculate_stats, format_entry_line, format_stats, tail_logs
from vibe.orchestrator.project_updater import ProjectUpdater
from vibe.state import SessionContext, SessionState

# Rich console for terminal output
console = Console()

# Typer app for CLI
app = typer.Typer(
    name="vibe",
    help="GLM-4.7 as brain, Claude Code as worker - AI pair programming orchestrator",
    add_completion=False,
)

# Global clients (initialized after startup validation)
_glm_client: GLMClient | None = None
_memory: VibeMemory | None = None
_repository: VibeRepository | None = None  # New unified persistence
_perplexity: PerplexityClient | None = None
_github: GitHubOps | None = None
_debug_session: DebugSession | None = None
_current_repo_session_id: str | None = None  # Track current session for signal handler


def handle_shutdown(signum: int, frame: types.FrameType | None) -> NoReturn:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    console.print("\n[yellow]Shutting down Vibe...[/yellow]")
    # End sessions and save state
    if _memory and _memory.session_id:
        try:
            _memory.end_session("Session ended via signal")
        except Exception:
            pass  # Best effort on shutdown
    # End new persistence session with summary
    if _repository and _current_repo_session_id:
        try:
            _repository.end_session(
                _current_repo_session_id,
                summary="Session ended via signal (SIGINT/SIGTERM)",
                status=SessionStatus.COMPLETED,
            )
        except Exception:
            pass
    # Close repository connection
    if _repository:
        try:
            _repository.close()
        except Exception:
            pass
    sys.exit(0)


def persist_message(
    session_id: str,
    role: MessageRole,
    content: str,
    message_type: MessageType = MessageType.CHAT,
) -> None:
    """
    Persist a message to the new unified persistence layer.

    This is a helper for gradually migrating to the new system.
    Fails silently if repository is not available.
    """
    if _repository and session_id:
        try:
            _repository.add_message(session_id, role, content, message_type)
        except Exception as e:
            logger.debug(f"Failed to persist message: {e}")


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def _check_openrouter(ping_glm: bool) -> tuple[str, tuple[bool, str]]:
    """Check OpenRouter API (optionally ping GLM)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return ("OpenRouter API", (False, "OPENROUTER_API_KEY not set"))
    elif ping_glm:
        success, message = ping_glm_sync(api_key, timeout=15.0)
        if success:
            return ("OpenRouter API", (True, f"GLM-4 ({message})"))
        else:
            return ("OpenRouter API", (False, message))
    else:
        return ("OpenRouter API", (True, "key configured"))


def _check_claude_cli() -> tuple[str, tuple[bool, str]]:
    """Check Claude CLI availability."""
    claude_path = shutil.which("claude")
    if claude_path:
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
            return ("Claude Code CLI", (True, version))
        except Exception as e:
            return ("Claude Code CLI", (False, str(e)))
    else:
        return ("Claude Code CLI", (False, "not installed"))


def _check_memory_keeper() -> tuple[str, tuple[bool, str]]:
    """Check memory-keeper database."""
    if MEMORY_DB_PATH.exists():
        return ("Memory-keeper", (True, "database found"))
    else:
        return ("Memory-keeper", (False, "not found"))


def _check_github_cli() -> tuple[str, tuple[bool, str]]:
    """Check GitHub CLI authentication."""
    gh_path = shutil.which("gh")
    if gh_path:
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stderr.split("\n"):
                    if "Logged in" in line:
                        return ("GitHub CLI", (True, "authenticated"))
                return ("GitHub CLI", (True, "authenticated"))
            else:
                return ("GitHub CLI", (False, "not authenticated"))
        except Exception as e:
            return ("GitHub CLI", (False, str(e)))
    else:
        return ("GitHub CLI", (False, "not installed"))


def validate_startup(ping_glm: bool = True) -> dict[str, tuple[bool, str]]:
    """
    Validate all required systems are available.

    Runs all checks in parallel for faster startup.

    Args:
        ping_glm: Whether to actually ping GLM API (slower but more reliable)

    Returns:
        Dict mapping system name to (success, message) tuple
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, tuple[bool, str]] = {}

    # Run all checks in parallel for faster startup
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_check_openrouter, ping_glm),
            executor.submit(_check_claude_cli),
            executor.submit(_check_memory_keeper),
            executor.submit(_check_github_cli),
        ]

        for future in as_completed(futures):
            try:
                name, result = future.result(timeout=20.0)
                results[name] = result
            except Exception as e:
                # If a check itself raises, log it but don't crash
                logger.warning(f"Startup check failed: {e}")

    return results


def show_startup_panel(results: dict[str, tuple[bool, str]]) -> bool:
    """
    Display startup validation results.

    Returns:
        True if all checks passed, False otherwise
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold]VIBE ORCHESTRATOR[/bold] - Startup Check",
            border_style="blue",
        )
    )
    console.print()

    all_passed = True
    for system, (success, message) in results.items():
        if success:
            console.print(f"  [green][✓][/green] {system:<20} {message}")
        else:
            console.print(f"  [red][✗][/red] {system:<20} {message}")
            all_passed = False

    console.print()
    return all_passed


def show_project_list(config: VibeConfig) -> int | None:
    """
    Display project list and get user selection.

    Returns:
        Selected project index (1-based) or None if quit
    """
    console.print(
        Panel.fit(
            "[bold]Your Projects[/bold]",
            border_style="blue",
        )
    )
    console.print()

    if not config.projects:
        console.print("  [dim]No projects registered yet.[/dim]")
        console.print("  Use [cyan]vibe add[/cyan] to register a project.")
        console.print()
        return None

    for i, project in enumerate(config.projects, 1):
        exists = "[green]✓[/green]" if project.exists() else "[red]✗[/red]"
        console.print(f"  {exists} [{i}] [bold]{project.name}[/bold]")
        console.print(f"      [dim]{project.path}[/dim]")
        if project.description:
            console.print(f"      {project.description}")
        console.print()

    console.print("  [dim][N] New project  [Q] Quit[/dim]")
    console.print()

    while True:
        choice = Prompt.ask("Select project", default="1")

        if choice.upper() == "Q":
            return None
        if choice.upper() == "N":
            # TODO: Implement new project flow
            console.print("[yellow]New project registration not yet implemented.[/yellow]")
            continue

        try:
            idx = int(choice)
            if 1 <= idx <= len(config.projects):
                return idx
            console.print(f"[red]Invalid selection. Choose 1-{len(config.projects)}[/red]")
        except ValueError:
            console.print("[red]Enter a number or Q to quit[/red]")


def show_project_loaded(project: Project, memory_items: int = 0) -> None:
    """Display project loaded confirmation."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold]{project.name}[/bold] loaded\n"
            f"[dim]Path: {project.path}[/dim]\n"
            f"[dim]Memory: {memory_items} items restored[/dim]",
            border_style="green",
        )
    )
    console.print()


def load_project_context(project: Project) -> str:
    """
    Load project context for GLM (starmap, guidelines, task history).

    Task history comes from TaskHistory class (in-memory, always available).

    Args:
        project: The project to load context for

    Returns:
        Context string for GLM
    """
    context_parts = [f"Project: {project.name}", f"Path: {project.path}"]

    # Load STARMAP.md if exists
    starmap_path = project.starmap_path
    if starmap_path.exists():
        try:
            content = starmap_path.read_text()[:2000]
            context_parts.append(f"\n## Project Structure (STARMAP.md):\n{content}")
        except Exception:
            pass

    # Load CLAUDE.md if exists
    claude_md_path = project.claude_md_path
    if claude_md_path.exists():
        try:
            content = claude_md_path.read_text()[:2000]
            context_parts.append(f"\n## Project Guidelines (CLAUDE.md):\n{content}")
        except Exception:
            pass

    # Get task history from TaskHistory (ALWAYS available, in-memory)
    task_context = get_context_for_glm()
    if task_context:
        context_parts.append(f"\n{task_context}")

    return "\n".join(context_parts)


# =============================================================================
# DEBUG WORKFLOW - GLM as Brain, Claude as Hands
# =============================================================================

async def execute_debug_workflow(
    glm_client: GLMClient,
    executor: ClaudeExecutor,
    project: Project,
    problem: str,
    memory: VibeMemory | None = None,
) -> None:
    """
    Execute the Claude-driven debugging workflow.

    Flow:
    1. Initialize DebugContext
    2. GLM generates initial task for Claude
    3. Loop:
       a. Claude executes with FULL history
       b. GLM reviews iteration
       c. If solved: done
       d. Else: GLM generates next task
    4. Save context to memory

    Args:
        glm_client: GLM client for task generation and review
        executor: Claude executor
        project: Current project
        problem: The debugging problem description
        memory: Optional memory for persistence
    """
    MAX_ITERATIONS = 10

    # Initialize debug context
    context = DebugContext(problem=problem)

    console.print(Panel(
        f"[bold]Debug Workflow Started[/bold]\n\n"
        f"Problem: {problem}\n"
        f"Max iterations: {MAX_ITERATIONS}",
        border_style="cyan",
    ))

    # GLM generates initial task
    console.print("\n[dim]GLM generating initial task...[/dim]")
    task_info = await glm_client.generate_debug_task(
        problem=problem,
        iterations_summary=context.format_iterations_summary(),
        hypothesis=context.hypothesis,
    )
    current_task = task_info.get("task", f"Investigate: {problem}")

    for iteration_num in range(1, MAX_ITERATIONS + 1):
        console.print(f"\n[bold cyan]═══ Debug Iteration {iteration_num}/{MAX_ITERATIONS} ═══[/bold cyan]")
        console.print(f"[bold]Task:[/bold] {current_task}")

        # Build prompt with FULL context for Claude
        claude_prompt = DEBUG_CLAUDE_PROMPT.format(
            context=context.format_for_claude(),
            task=current_task,
        )

        # Claude executes
        console.print()
        with console.status("[bold blue]Claude investigating...[/bold blue]"):
            result = await executor.execute(
                task_description=claude_prompt,
                timeout_tier="research",  # 300s for investigation
            )

        # Display Claude's output
        if result.success and result.result:
            console.print(Panel(
                result.result,
                title="Claude's Findings",
                border_style="green" if result.success else "red",
            ))
            # Flush after Panel to prevent buffer bleeding
            import sys
            sys.stdout.flush()
        elif result.error:
            console.print(f"[red]Error: {result.error}[/red]")

        # Track iteration in context
        context.add_iteration(
            task=current_task,
            output=result.result or result.error or "(no output)",
            files_changed=result.file_changes,
            duration_ms=result.duration_ms,
        )

        # Show duration
        duration_sec = result.duration_ms / 1000
        console.print(f"[dim]Duration: {duration_sec:.1f}s[/dim]")

        # GLM reviews
        console.print("\n[dim]GLM reviewing...[/dim]")
        review = await glm_client.review_debug_iteration(
            problem=problem,
            task=current_task,
            output=result.result or result.error or "",
            files_changed=result.file_changes,
            must_preserve=context.must_preserve,
            previous_iterations=context.format_iterations_summary(),
        )

        # Track review
        context.add_review(
            approved=review.get("approved", False),
            is_problem_solved=review.get("is_problem_solved", False),
            feedback=review.get("feedback", ""),
            next_task=review.get("next_task"),
        )

        # Display review result
        if review.get("is_problem_solved"):
            console.print(Panel(
                f"[bold green]PROBLEM SOLVED![/bold green]\n\n{review.get('feedback', '')}",
                border_style="green",
            ))
            break

        elif review.get("approved"):
            console.print(Panel(
                f"[bold yellow]ITERATION APPROVED[/bold yellow]\n\n{review.get('feedback', '')}\n\n"
                f"[bold]Next:[/bold] {review.get('next_task', 'Continue')}",
                border_style="yellow",
            ))
        else:
            console.print(Panel(
                f"[bold red]NEEDS MORE WORK[/bold red]\n\n{review.get('feedback', '')}\n\n"
                f"[bold]Next:[/bold] {review.get('next_task', 'Continue investigation')}",
                border_style="red",
            ))

        # Get next task
        next_task = review.get("next_task")
        if next_task:
            current_task = next_task
        else:
            # GLM generates new task
            console.print("\n[dim]GLM generating next task...[/dim]")
            task_info = await glm_client.generate_debug_task(
                problem=problem,
                iterations_summary=context.format_iterations_summary(),
                hypothesis=context.hypothesis,
            )
            current_task = task_info.get("task", "Continue investigation")

    # End of loop
    if not context.is_complete:
        console.print(f"\n[yellow]Max iterations ({MAX_ITERATIONS}) reached without solving problem.[/yellow]")

    # Show summary
    glm_stats = glm_client.get_usage_stats()
    console.print(Panel(
        f"[bold]Debug Session Summary[/bold]\n\n"
        f"Problem: {problem}\n"
        f"Iterations: {len(context.iterations)}\n"
        f"Solved: {'Yes' if context.is_complete else 'No'}\n"
        f"GLM tokens: {glm_stats.get('total_tokens', 0):,}",
        border_style="blue",
    ))

    # Save to memory
    if memory:
        try:
            memory.save(
                key=f"debug-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                value=str(context.to_dict()),
                category="progress",
                priority="high",
            )
        except Exception as e:
            logger.warning(f"Failed to save debug context: {e}")

    # Record in task history
    add_task(
        description=f"Debug: {problem[:80]}",
        success=context.is_complete,
        summary=f"{len(context.iterations)} iterations, solved={context.is_complete}",
    )

    # Ensure all output is flushed before returning
    import sys
    console.file.flush() if hasattr(console, 'file') else None
    sys.stdout.flush()
    sys.stderr.flush()


async def execute_task_with_claude(
    executor: ClaudeExecutor,
    task: dict,
    task_num: int,
    total_tasks: int,
    debug_context: str | None = None,
) -> tuple[TaskResult, dict]:
    """
    Execute a single task with Claude, showing progress.

    Args:
        executor: ClaudeExecutor instance
        task: Task dictionary from GLM
        task_num: Current task number (1-based)
        total_tasks: Total number of tasks
        debug_context: Optional debug session context to inject

    Returns:
        Tuple of (TaskResult, tool_verification_result)
    """
    description = task.get("description", "No description")
    files = task.get("files")
    constraints = task.get("constraints")

    # Detect task type for timeout tier
    # Investigation/research tasks get longer timeout (300s vs 120s)
    from vibe.glm.client import is_investigation_request
    timeout_tier = "research" if is_investigation_request(description) else "code"

    # Progress callback for live updates
    progress_lines: list[str] = []

    def on_progress(text: str) -> None:
        progress_lines.append(text)

    def on_tool_call(tool_call: ToolCall) -> None:
        console.print(f"    [dim]→ {tool_call.name}[/dim]")

    tier_label = f" [dim]({timeout_tier})[/dim]" if timeout_tier == "research" else ""
    console.print(
        Panel(
            f"[bold]Task {task_num}/{total_tasks}:[/bold] {description}{tier_label}",
            border_style="cyan",
        )
    )

    # Execute with spinner
    with console.status("[bold blue]Claude working...[/bold blue]"):
        result = await executor.execute(
            task_description=description,
            files=files,
            constraints=constraints,
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            debug_context=debug_context,
            timeout_tier=timeout_tier,
        )

    # Verify tool usage
    tool_verification = executor.verify_tool_usage(description, result.tool_calls)

    # Show tool verification result if there are issues
    if not tool_verification["passed"]:
        console.print()
        console.print(
            Panel(
                f"[bold yellow]Tool Usage Warning[/bold yellow]\n\n"
                f"{tool_verification['feedback']}",
                border_style="yellow",
            )
        )

    return result, tool_verification


async def review_with_glm(
    glm_client: GLMClient,
    task: dict,
    result: TaskResult,
    project_path: str,
) -> dict:
    """
    Have GLM review Claude's changes.

    Args:
        glm_client: GLM client instance
        task: Original task dictionary
        result: Claude's TaskResult
        project_path: Path to project for git diff

    Returns:
        Review result dict with approved, issues, feedback
    """
    # Get git diff for changed files (ignore truncation flag in CLI mode)
    diff, _ = get_git_diff(project_path, result.file_changes)

    # Get Claude's summary
    claude_summary = result.result or "(no summary provided)"

    # Have GLM review
    with console.status("[bold yellow]GLM reviewing changes...[/bold yellow]"):
        review = await glm_client.review_changes(
            task_description=task.get("description", ""),
            changes_diff=diff,
            claude_summary=claude_summary,
        )

    return review


def show_task_result(result: TaskResult, review: dict) -> None:
    """Display task result and review to user."""
    # Show what Claude did
    if result.file_changes:
        console.print("\n  [bold]Files modified:[/bold]")
        for f in result.file_changes:
            console.print(f"    [green]✓[/green] {f}")

    if result.result:
        console.print(f"\n  [bold]Claude's summary:[/bold]")
        # Show full summary, wrapped for readability
        for line in result.result.split('\n'):
            console.print(f"    {line}")

    # Show review result
    console.print()
    if review.get("approved"):
        console.print(
            Panel(
                "[bold green]APPROVED[/bold green]",
                border_style="green",
            )
        )
    else:
        issues = review.get("issues", [])
        feedback = review.get("feedback", "")
        issue_text = "\n".join(f"  • {issue}" for issue in issues) if issues else ""
        console.print(
            Panel(
                f"[bold red]REJECTED[/bold red]\n\n{issue_text}\n\n{feedback}",
                border_style="red",
            )
        )

    # Show duration (Claude cost excluded - included in Max subscription)
    if result.duration_ms > 0:
        duration_sec = result.duration_ms / 1000
        console.print(f"  [dim]Duration: {duration_sec:.1f}s[/dim]")


async def process_user_request(
    glm_client: GLMClient,
    context: SessionContext,
    project: Project,
    user_request: str,
    memory: VibeMemory | None = None,
) -> None:
    """
    Process a user request through GLM and execute with Claude.

    Args:
        glm_client: GLM client instance
        context: Session context
        project: Current project
        user_request: User's request text
        memory: Optional memory client for persistence
    """
    # Log user request
    session_logger.info(SessionLogEntry(
        timestamp=now_iso(),
        session_id=get_session_id(),
        event_type="request",
        project_name=context.project_name,
        user_request=user_request[:500],  # Truncate for log size
    ).to_json())

    # Check if this is a debug request - use dedicated debug workflow
    from vibe.glm.client import is_investigation_request

    # Keywords that trigger debug workflow - only check FIRST LINE to avoid
    # false positives from pasted findings containing words like "Fixed"
    DEBUG_KEYWORDS = ["debug", "broken", "not working", "bug", "crash", "why is", "what's wrong"]
    first_line = user_request.split('\n')[0].lower()
    is_debug = any(keyword in first_line for keyword in DEBUG_KEYWORDS)

    if is_debug:
        console.print("[dim]Debug request detected - using debug workflow[/dim]")

        # Record request
        add_request(user_request)

        # Create executor for debug workflow
        executor = ClaudeExecutor(
            project_path=project.path,
            timeout_tier="research",
        )

        # Run debug workflow
        await execute_debug_workflow(
            glm_client=glm_client,
            executor=executor,
            project=project,
            problem=user_request,
            memory=memory,
        )
        return

    # Load project context (task history comes from TaskHistory class)
    project_context = load_project_context(project)

    # Record this request in TaskHistory
    add_request(user_request)

    # Add user message to history
    context.add_glm_message("user", user_request)

    # Check if clarification is needed (max 1 question, then delegate)
    console.print()
    with console.status("[bold blue]GLM analyzing request...[/bold blue]"):
        clarification = await glm_client.ask_clarification(
            user_request, project_context, clarification_count=context.clarification_count
        )

    if clarification:
        # GLM needs more info - but only allow 1 clarification
        context.clarification_count += 1
        console.print(
            Panel(
                clarification,
                title="[bold yellow]GLM needs clarification[/bold yellow]",
                border_style="yellow",
            )
        )
        context.add_glm_message("assistant", clarification)
        return

    # Reset clarification count when proceeding to decomposition
    context.clarification_count = 0

    # Decompose into tasks
    console.print()
    with console.status("[bold blue]GLM decomposing into tasks...[/bold blue]"):
        try:
            tasks = await glm_client.decompose_task(user_request, project_context)
        except Exception as e:
            console.print(f"[red]Error decomposing task: {e}[/red]")
            return

    # Persist GLM decomposition response
    if tasks:
        import json as json_module
        decomposition_content = json_module.dumps([
            {"description": t.get("description", ""), "files": t.get("files", []), "constraints": t.get("constraints", [])}
            for t in tasks
        ])
        persist_message(context.repo_session_id, MessageRole.GLM, decomposition_content, MessageType.DECOMPOSITION)

    # Show task plan
    console.print(
        Panel.fit(
            "[bold]Task Plan[/bold]",
            border_style="blue",
        )
    )

    for i, task in enumerate(tasks, 1):
        console.print(f"\n  [bold cyan]Task {i}:[/bold cyan] {task.get('description', 'No description')}")
        if task.get("files"):
            console.print(f"    [dim]Files: {', '.join(task['files'])}[/dim]")
        if task.get("constraints"):
            console.print(f"    [dim]Constraints: {', '.join(task['constraints'])}[/dim]")

    console.print()

    # Ask for confirmation - use simple input() instead of Rich Prompt.ask
    # Rich's Prompt with choices can't handle multi-line paste (each line
    # triggers "Please select one of the available options" spam)
    try:
        console.print("[bold]Execute these tasks?[/bold] [dim][y/n] (y):[/dim] ", end="")
        sys.stdout.flush()
        confirm = input().strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        confirm = "n"

    if confirm != "y":
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Create checkpoint before execution (if memory available)
    if memory:
        try:
            checkpoint_name = f"pre-execution-{datetime.now().strftime('%H%M%S')}"
            memory.create_checkpoint_with_git(
                name=checkpoint_name,
                description=f"Before executing: {user_request[:50]}",
                project_path=project.path,
            )
            console.print(f"  [dim]Checkpoint created: {checkpoint_name}[/dim]")
        except Exception as e:
            console.print(f"  [dim]Warning: Could not create checkpoint: {e}[/dim]")

    # Load global conventions from memory
    global_conventions = []
    if memory:
        try:
            global_conventions = memory.load_conventions()
            if global_conventions:
                console.print(f"  [dim]Loaded {len(global_conventions)} global conventions[/dim]")
        except Exception:
            pass  # Non-critical

    # Initialize Claude executor with conventions
    # Use "research" tier for investigation tasks, "code" for normal tasks
    try:
        executor = ClaudeExecutor(
            project_path=project.path,
            timeout_tier="code",  # Default, will be overridden per-task
            global_conventions=global_conventions,
        )
    except ClaudeError as e:
        console.print(f"[red]Claude error: {e}[/red]")
        return

    # Execute each task
    context.transition_to(SessionState.EXECUTING)
    completed = 0
    failed = 0
    all_file_changes: list[str] = []  # Track all modified files
    all_summaries: list[str] = []  # Track all task summaries
    task_ids: dict[int, str] = {}  # Map task index to persistence task ID

    # Create tasks in persistence before execution
    if _repository and context.repo_session_id:
        for idx, t in enumerate(tasks, 1):
            try:
                repo_task = _repository.create_task(
                    session_id=context.repo_session_id,
                    description=t.get("description", ""),
                    files=t.get("files", []),
                    constraints=t.get("constraints", []),
                    original_request=user_request[:200] if user_request else None,
                )
                task_ids[idx] = repo_task.id
            except Exception as e:
                logger.debug(f"Failed to create task in persistence: {e}")

    MAX_RETRIES = 3  # Maximum retry attempts per task

    for i, task in enumerate(tasks, 1):
        attempt = 0
        task_completed = False
        previous_feedback = ""

        while attempt < MAX_RETRIES and not task_completed:
            attempt += 1
            try:
                # Get debug context if in debug session
                debug_ctx = _debug_session.get_context_for_claude() if _debug_session else None

                # Add previous rejection feedback to constraints for retry
                # Use deepcopy to prevent state bleed from mutable constraint lists
                task_with_feedback = copy.deepcopy(task)
                if previous_feedback:
                    # Limit feedback to prevent context overflow - only include last rejection
                    # Truncate to 500 chars max to prevent prompt bloat
                    truncated_feedback = previous_feedback[:500]
                    if len(previous_feedback) > 500:
                        truncated_feedback += "... [truncated]"

                    existing_constraints = task_with_feedback.get("constraints", []) or []
                    retry_constraints = [
                        f"PREVIOUS ATTEMPT REJECTED: {truncated_feedback}",
                        "Address the feedback above before proceeding.",
                    ]
                    task_with_feedback["constraints"] = existing_constraints + retry_constraints
                    console.print(f"\n  [yellow]Retrying task (attempt {attempt}/{MAX_RETRIES})...[/yellow]")

                # Execute with Claude
                result, tool_verification = await execute_task_with_claude(
                    executor=executor,
                    task=task_with_feedback,
                    task_num=i,
                    total_tasks=len(tasks),
                    debug_context=debug_ctx,
                )

                if not result.success:
                    console.print(f"[red]Task failed: {result.error}[/red]")
                    failed += 1
                    context.add_error(result.error or "Unknown error")
                    # Save failure to TaskHistory (always works) and memory (if available)
                    add_task(task.get("description", ""), success=False, summary=result.error or "Unknown error")
                    if memory:
                        memory.save_task_result(
                            task_description=task.get("description", ""),
                            success=False,
                            summary=result.error or "Unknown error",
                        )
                    break  # Don't retry on execution failure, only on review rejection

                # Check if tool verification failed (missing required tools)
                if not tool_verification["passed"] and tool_verification.get("missing_required"):
                    console.print(
                        f"[yellow]Warning: Task may be incomplete - missing required tools[/yellow]"
                    )

                # Review with GLM - wrapped in try/except to avoid losing work on review crash
                context.transition_to(SessionState.REVIEWING)
                try:
                    review = await review_with_glm(
                        glm_client=glm_client,
                        task=task,
                        result=result,
                        project_path=project.path,
                    )
                except Exception as review_error:
                    # Review crashed - auto-approve to avoid losing Claude's work
                    console.print(f"[yellow]Review failed ({review_error}), auto-approving to preserve work[/yellow]")
                    review = {
                        "approved": True,
                        "issues": [],
                        "feedback": f"Auto-approved due to review error: {str(review_error)[:100]}",
                    }

                # Persist GLM review response to new persistence
                import json as json_module
                review_content = json_module.dumps({
                    "task": task.get("description", "")[:100],
                    "approved": review.get("approved", False),
                    "issues": review.get("issues", []),
                    "feedback": review.get("feedback", "")[:500],
                })
                persist_message(context.repo_session_id, MessageRole.GLM, review_content, MessageType.REVIEW)

                # If tool verification failed, add that to review issues
                if not tool_verification["passed"]:
                    existing_issues = review.get("issues", [])
                    existing_issues.append(f"Tool verification: {tool_verification['feedback']}")
                    review["issues"] = existing_issues

                # Show result
                show_task_result(result, review)

                if review.get("approved"):
                    completed += 1
                    task_completed = True
                    context.add_completed_task(task.get("description", ""))
                    # Track file changes and summaries for project updates
                    all_file_changes.extend(result.file_changes)
                    if result.result:
                        all_summaries.append(result.result)
                    # Save success to TaskHistory (always works) and memory (if available)
                    add_task(
                        task.get("description", ""),
                        success=True,
                        summary=result.result or "",
                        files_changed=result.file_changes,
                    )
                    if memory:
                        memory.save_task_result(
                            task_description=task.get("description", ""),
                            success=True,
                            summary=result.result or "",
                            files_changed=result.file_changes,
                        )
                    # Update task status in new persistence
                    if _repository and i in task_ids:
                        try:
                            _repository.update_task_status(
                                task_ids[i], TaskStatus.COMPLETED, reason="Approved by GLM review"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to update task status: {e}")

                    # Git commit after approval - GLM owns the commit
                    if result.file_changes:
                        try:
                            import subprocess
                            # Add all changed files
                            subprocess.run(
                                ["git", "add", "--"] + result.file_changes,
                                cwd=project.path,
                                capture_output=True,
                                timeout=30,
                            )
                            # Commit with task description
                            task_desc = task.get("description", "Task completed")[:72]
                            commit_msg = f"vibe: {task_desc}\n\nApproved by GLM review gate."
                            subprocess.run(
                                ["git", "commit", "-m", commit_msg],
                                cwd=project.path,
                                capture_output=True,
                                timeout=30,
                            )
                            console.print(f"  [dim]Git: committed {len(result.file_changes)} file(s)[/dim]")
                        except Exception as git_err:
                            console.print(f"  [yellow]Git commit skipped: {git_err}[/yellow]")
                else:
                    # Task rejected - prepare for retry with meaningful default feedback
                    feedback_text = review.get("feedback", "")
                    issues_text = "; ".join(review.get("issues", []))
                    previous_feedback = feedback_text or issues_text or "Task did not meet quality standards. Please review and improve."
                    # Save rejection to memory for learning
                    if memory:
                        memory.save(
                            key=f"rejection-{task.get('id', i)}-{attempt}",
                            value=f"Task: {task.get('description', '')}\nFeedback: {previous_feedback}",
                            category="warning",
                            priority="high",
                        )

                    if attempt >= MAX_RETRIES:
                        # Out of retries
                        failed += 1
                        console.print(f"\n  [red]Task failed after {MAX_RETRIES} attempts[/red]")
                        # Save failure to TaskHistory (always works) and memory (if available)
                        add_task(
                            task.get("description", ""),
                            success=False,
                            summary=f"Rejected after {MAX_RETRIES} attempts: {previous_feedback[:100]}",
                        )
                        if memory:
                            memory.save_task_result(
                                task_description=task.get("description", ""),
                                success=False,
                                summary=f"Rejected after {MAX_RETRIES} attempts: {previous_feedback}",
                            )
                        # Update task status in new persistence
                        if _repository and i in task_ids:
                            try:
                                _repository.update_task_status(
                                    task_ids[i], TaskStatus.FAILED,
                                    reason=f"Rejected after {MAX_RETRIES} attempts"
                                )
                            except Exception as e:
                                logger.debug(f"Failed to update task status: {e}")

                context.transition_to(SessionState.EXECUTING)

            except ClaudeError as e:
                console.print(f"[red]Claude error on task {i}: {e}[/red]")
                failed += 1
                context.add_error(str(e))
                # Record the failed task so "redo the failed task" works
                add_task(
                    task.get("description", ""),
                    success=False,
                    summary=f"Error: {str(e)[:200]}",
                )
                # Update task status in new persistence
                if _repository and i in task_ids:
                    try:
                        _repository.update_task_status(
                            task_ids[i], TaskStatus.FAILED,
                            reason=f"Claude error: {str(e)[:100]}"
                        )
                    except Exception:
                        pass  # Non-critical
                break  # Don't retry on Claude errors

    # Final summary with GLM usage
    context.transition_to(SessionState.IDLE)
    glm_stats = glm_client.get_usage_stats()
    glm_tokens = glm_stats.get("total_tokens", 0)
    glm_cost = glm_tokens * 0.0000006  # Approximate GLM-4.7 cost per token
    console.print()
    console.print(
        Panel(
            f"[bold]Session Complete[/bold]\n\n"
            f"  Completed: [green]{completed}[/green]\n"
            f"  Failed: [red]{failed}[/red]\n"
            f"  Total: {len(tasks)}\n"
            f"  GLM tokens: {glm_tokens:,} (~${glm_cost:.4f})",
            border_style="blue",
        )
    )

    # Record in context and memory
    summary = f"Executed {completed}/{len(tasks)} tasks successfully"
    context.add_glm_message("assistant", summary)

    # Save session progress to memory
    if memory:
        memory.save(
            key=f"request-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            value=f"Request: {user_request}\nResult: {summary}",
            category="progress",
            priority="normal",
        )

    # Update project documentation if files were changed
    if completed > 0 and all_file_changes:
        try:
            updater = ProjectUpdater(project.path, glm_client)
            combined_summary = " ".join(all_summaries) if all_summaries else user_request

            with console.status("[bold blue]Updating project docs...[/bold blue]"):
                update_results = await updater.update_after_task(
                    task_description=user_request,
                    files_changed=list(set(all_file_changes)),  # Deduplicate
                    claude_summary=combined_summary,
                    update_starmap=True,
                    update_changelog=True,
                )

            if update_results.get("starmap"):
                console.print("  [dim]Updated STARMAP.md[/dim]")
            if update_results.get("changelog"):
                console.print("  [dim]Updated CHANGELOG.md[/dim]")
        except Exception as e:
            console.print(f"  [dim]Warning: Project update failed: {e}[/dim]")


def conversation_loop(
    context: SessionContext,
    config: VibeConfig,
    project: Project,
    glm_client: GLMClient,
    memory: VibeMemory | None = None,
) -> None:
    """
    Main conversation loop with GLM.

    This is where the user interacts with GLM, which delegates to Claude.
    Features: command history (up/down arrows), tab completion for /commands.
    """
    console.print("[bold]What do you want to work on?[/bold]")
    console.print("[dim]Type your request, or /help for commands, /quit to exit[/dim]")
    console.print("[dim]Use ↑/↓ arrows for history, Tab for command completion[/dim]")
    console.print()

    # Create prompt session with history and completion
    try:
        from vibe.cli.prompt import create_prompt_session, prompt_input
        prompt_session = create_prompt_session(project_path=project.path)
        use_enhanced_prompt = True
    except ImportError:
        # Fallback to basic input if prompt_toolkit not available
        prompt_session = None
        use_enhanced_prompt = False
        console.print("[dim]Note: Install prompt_toolkit for enhanced input features[/dim]")

    def read_multiline_input() -> str:
        """Read input that may span multiple lines (for paste support).

        Uses prompt_toolkit for history and completion if available.
        - Commands (starting with /) are processed immediately
        - Everything else reads until empty line or EOF
        """
        if use_enhanced_prompt and prompt_session:
            try:
                return prompt_input(prompt_session, "> ", multiline=True)
            except (KeyboardInterrupt, EOFError):
                return ""

        # Fallback to basic input
        lines: list[str] = []
        console.print("[bold cyan]>[/bold cyan] ", end="")
        sys.stdout.flush()

        empty_count = 0

        try:
            while True:
                line = input()

                # Commands are always single-line, process immediately
                if not lines and line.startswith("/"):
                    return line

                # Track consecutive empty lines
                if not line.strip():
                    empty_count += 1
                    # Two empty lines in a row = done
                    if empty_count >= 2 and lines:
                        break
                    # Single empty line after content
                    if empty_count >= 1 and lines:
                        # Add the empty line (preserve formatting)
                        lines.append("")
                        # Brief moment for more paste
                        import select
                        if hasattr(select, 'select'):
                            readable, _, _ = select.select([sys.stdin], [], [], 0.15)
                            if not readable:
                                # Remove trailing empty line if we're done
                                if lines and lines[-1] == "":
                                    lines.pop()
                                break
                        else:
                            lines.pop()  # Remove empty on Windows
                            break
                    continue
                else:
                    empty_count = 0

                lines.append(line)

        except EOFError:
            pass

        return "\n".join(lines)

    while True:
        try:
            user_input = read_multiline_input()

            if not user_input.strip():
                continue

            # Handle exit commands (with or without slash, like Claude Code)
            if user_input.lower().strip() in ("exit", "quit", "q"):
                if memory and memory.session_id:
                    memory.end_session("Session ended by user")
                # End new persistence session with summary
                if _repository and context.repo_session_id:
                    try:
                        stats = context.get_stats()
                        summary = f"Completed {stats['completed_tasks']} tasks, {stats['error_count']} errors, duration {stats['duration_seconds']:.0f}s"
                        _repository.end_session(context.repo_session_id, summary=summary)
                    except Exception:
                        pass
                console.print("[yellow]Goodbye![/yellow]")
                break

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd in ("/quit", "/exit", "/q"):
                    # End memory session before exiting
                    if memory and memory.session_id:
                        memory.end_session("Session ended by user")
                    # End new persistence session with summary
                    if _repository and context.repo_session_id:
                        try:
                            stats = context.get_stats()
                            summary = f"Completed {stats['completed_tasks']} tasks, {stats['error_count']} errors, duration {stats['duration_seconds']:.0f}s"
                            _repository.end_session(context.repo_session_id, summary=summary)
                        except Exception:
                            pass
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif cmd == "/help":
                    console.print("\n[bold]Commands:[/bold]")
                    console.print("  /quit       - Exit Vibe")
                    console.print("  /status     - Show session status")
                    console.print("  /usage      - Show GLM usage stats")
                    console.print("  /memory     - Show memory stats")
                    console.print("  /history    - Show task history (what GLM sees)")
                    console.print("  /redo       - Re-execute the most recent failed task")
                    console.print("  /convention - Manage global conventions")
                    console.print("  /debug      - Debug session tracking")
                    console.print("  /rollback   - Rollback to debug checkpoint")
                    console.print("  /research   - Research a topic via Perplexity")
                    console.print("  /github     - Show GitHub repo info")
                    console.print("  /issues     - List GitHub issues")
                    console.print("  /prs        - List GitHub pull requests")
                    console.print("  /project    - Switch project")
                    console.print("  /help       - Show this help")
                    console.print()
                elif cmd == "/status":
                    stats = context.get_stats()
                    console.print(f"\n[bold]Session Status:[/bold]")
                    console.print(f"  State: {stats['state']}")
                    console.print(f"  Project: {stats['project']}")
                    console.print(f"  Completed tasks: {stats['completed_tasks']}")
                    console.print(f"  Errors: {stats['error_count']}")
                    console.print(f"  Duration: {stats['duration_seconds']:.0f}s")
                    console.print()
                elif cmd == "/usage":
                    usage = glm_client.get_usage_stats()
                    console.print(f"\n[bold]GLM Usage:[/bold]")
                    console.print(f"  Model: {usage['model']}")
                    console.print(f"  Requests: {usage['request_count']}")
                    console.print(f"  Total tokens: {usage['total_tokens']}")
                    console.print()
                elif cmd == "/memory":
                    if memory:
                        stats = memory.get_stats()
                        console.print(f"\n[bold]Memory Stats:[/bold]")
                        console.print(f"  Total items: {stats['total_items']}")
                        console.print(f"  Sessions: {stats['session_count']}")
                        console.print(f"  Checkpoints: {stats['checkpoint_count']}")
                        if stats['by_category']:
                            console.print(f"  By category:")
                            for cat, count in stats['by_category'].items():
                                console.print(f"    {cat}: {count}")
                        console.print()
                    else:
                        console.print("[yellow]Memory not available[/yellow]")
                elif cmd == "/history":
                    # Show task history (from in-memory TaskHistory)
                    stats = TaskHistory.get_stats()
                    console.print(f"\n[bold]Task History:[/bold]")
                    console.print(f"  Total tasks: {stats['total_tasks']}")
                    console.print(f"  Completed: [green]{stats['completed']}[/green]")
                    console.print(f"  Failed: [red]{stats['failed']}[/red]")
                    console.print(f"  Requests tracked: {stats['requests']}")
                    console.print()

                    # Show recent tasks
                    tasks = TaskHistory.get_recent_tasks(10)
                    if tasks:
                        console.print("[bold]Recent Tasks:[/bold]")
                        for task in tasks:
                            icon = "[green]✓[/green]" if task.status == "completed" else "[red]✗[/red]"
                            console.print(f"  {icon} {task.description[:70]}")
                elif cmd == "/redo" or cmd.startswith("/redo "):
                    # Redo failed tasks directly without GLM decomposition
                    failed_tasks = [t for t in TaskHistory.get_recent_tasks(20) if t.status == "failed"]
                    if not failed_tasks:
                        console.print("[yellow]No failed tasks to redo[/yellow]")
                    else:
                        console.print(f"\n[bold]Failed Tasks ({len(failed_tasks)}):[/bold]")
                        for i, task in enumerate(failed_tasks, 1):
                            console.print(f"  {i}. {task.description[:70]}")
                        console.print()
                        console.print("[dim]Re-executing most recent failed task...[/dim]")
                        # Create task dict from the failed task
                        retry_task = {
                            "id": "retry-1",
                            "description": failed_tasks[0].description,
                            "files": ["investigate relevant files"],
                            "constraints": ["This is a retry of a previously failed task"],
                        }
                        # Execute directly
                        asyncio.run(execute_tasks(
                            glm_client=glm_client,
                            executor=ClaudeExecutor(project.path),
                            project=project,
                            tasks=[retry_task],
                            memory=memory,
                            context=context,
                            user_request=f"Retry: {failed_tasks[0].description}",
                        ))
                    continue
                elif cmd.startswith("/convention"):
                    # /convention - Manage global conventions
                    if not memory:
                        console.print("[yellow]Memory not available[/yellow]")
                        continue

                    parts = user_input.split(maxsplit=2)
                    subcommand = parts[1] if len(parts) > 1 else "list"

                    if subcommand == "list":
                        conventions = memory.list_conventions()
                        if conventions:
                            console.print(f"\n[bold]Global Conventions ({len(conventions)}):[/bold]")
                            for conv in conventions:
                                console.print(f"\n  [cyan]{conv['key']}[/cyan] ({conv['applies_to']})")
                                console.print(f"    {conv['convention']}")
                            console.print()
                        else:
                            console.print("[dim]No conventions defined yet.[/dim]")
                            console.print("[dim]Use: /convention add <key> <convention text>[/dim]")
                    elif subcommand == "add":
                        if len(parts) < 3:
                            console.print("[red]Usage: /convention add <key> <convention text>[/red]")
                            continue
                        rest = parts[2]
                        # Split key from convention text
                        key_parts = rest.split(maxsplit=1)
                        if len(key_parts) < 2:
                            console.print("[red]Usage: /convention add <key> <convention text>[/red]")
                            continue
                        key, convention = key_parts
                        memory.save_convention(key, convention)
                        console.print(f"[green]Saved convention: {key}[/green]")
                    elif subcommand == "delete":
                        if len(parts) < 3:
                            console.print("[red]Usage: /convention delete <key>[/red]")
                            continue
                        key = parts[2]
                        if memory.delete_convention(key):
                            console.print(f"[green]Deleted convention: {key}[/green]")
                        else:
                            console.print(f"[yellow]Convention not found: {key}[/yellow]")
                    else:
                        console.print("[dim]Usage:[/dim]")
                        console.print("  /convention list               - List all conventions")
                        console.print("  /convention add <key> <text>   - Add a convention")
                        console.print("  /convention delete <key>       - Delete a convention")
                elif cmd.startswith("/research"):
                    # /research <query> - Research a topic via Perplexity
                    if not _perplexity or not _perplexity.is_available:
                        console.print("[yellow]Perplexity not available (PERPLEXITY_API_KEY not set)[/yellow]")
                        continue
                    query = user_input[9:].strip()  # Remove "/research "
                    if not query:
                        query = Prompt.ask("What do you want to research?")
                    if query:
                        with console.status("[bold blue]Researching...[/bold blue]"):
                            try:
                                result = asyncio.run(_perplexity.research(query, context=load_project_context(project)))
                                console.print(Panel(
                                    Markdown(result.answer),
                                    title=f"[bold]Research: {query[:50]}...[/bold]" if len(query) > 50 else f"[bold]Research: {query}[/bold]",
                                    border_style="green",
                                ))
                                if result.citations:
                                    console.print("[dim]Citations:[/dim]")
                                    for citation in result.citations[:5]:
                                        console.print(f"  [dim]• {citation}[/dim]")
                            except ResearchError as e:
                                console.print(f"[red]Research failed: {e}[/red]")
                elif cmd == "/github":
                    # Show GitHub repo info
                    if not _github:
                        console.print("[yellow]GitHub CLI not configured[/yellow]")
                        continue
                    try:
                        repo_info = _github.get_repo_info()
                        if repo_info:
                            console.print(f"\n[bold]GitHub Repository:[/bold]")
                            console.print(f"  Name: {repo_info.get('name', 'Unknown')}")
                            owner = repo_info.get('owner', {})
                            console.print(f"  Owner: {owner.get('login', 'Unknown') if isinstance(owner, dict) else owner}")
                            console.print(f"  URL: {repo_info.get('url', 'Unknown')}")
                            console.print(f"  Private: {'Yes' if repo_info.get('isPrivate') else 'No'}")
                            default_branch = repo_info.get('defaultBranchRef', {})
                            console.print(f"  Default branch: {default_branch.get('name', 'Unknown') if isinstance(default_branch, dict) else 'main'}")
                            console.print()
                        else:
                            console.print("[yellow]No repo info available[/yellow]")
                    except GitHubError as e:
                        console.print(f"[red]GitHub error: {e}[/red]")
                elif cmd == "/issues":
                    # List GitHub issues
                    if not _github:
                        console.print("[yellow]GitHub CLI not configured[/yellow]")
                        continue
                    try:
                        with console.status("[bold blue]Fetching issues...[/bold blue]"):
                            issues = _github.list_issues(limit=10)
                        if issues:
                            console.print(f"\n[bold]Open Issues ({len(issues)}):[/bold]")
                            for issue in issues:
                                labels_str = f" [{', '.join(issue.labels)}]" if issue.labels else ""
                                console.print(f"  [cyan]#{issue.number}[/cyan] {issue.title}{labels_str}")
                            console.print()
                        else:
                            console.print("[dim]No open issues[/dim]")
                    except GitHubError as e:
                        console.print(f"[red]GitHub error: {e}[/red]")
                elif cmd == "/prs":
                    # List GitHub pull requests
                    if not _github:
                        console.print("[yellow]GitHub CLI not configured[/yellow]")
                        continue
                    try:
                        with console.status("[bold blue]Fetching pull requests...[/bold blue]"):
                            prs = _github.list_prs(limit=10)
                        if prs:
                            console.print(f"\n[bold]Open Pull Requests ({len(prs)}):[/bold]")
                            for pr in prs:
                                draft_str = " [draft]" if pr.draft else ""
                                console.print(f"  [cyan]#{pr.number}[/cyan] {pr.title}{draft_str}")
                                console.print(f"    [dim]{pr.head} → {pr.base}[/dim]")
                            console.print()
                        else:
                            console.print("[dim]No open pull requests[/dim]")
                    except GitHubError as e:
                        console.print(f"[red]GitHub error: {e}[/red]")
                elif cmd.startswith("/debug"):
                    # /debug - Debug session management
                    global _debug_session
                    parts = user_input.split(maxsplit=2)
                    subcommand = parts[1] if len(parts) > 1 else "status"

                    if subcommand == "start":
                        # /debug start <problem description>
                        if _debug_session and _debug_session.is_active:
                            console.print("[yellow]Debug session already active. Use /debug end first.[/yellow]")
                            continue
                        problem = parts[2] if len(parts) > 2 else Prompt.ask("Describe the problem")
                        if problem:
                            _debug_session = DebugSession(
                                project_path=project.path,
                                problem=problem,
                            )
                            console.print(f"\n[green bold]Debug Session Started[/green bold]")
                            console.print(f"  Problem: {problem}")
                            console.print(f"  Initial commit: {_debug_session.initial_commit or 'N/A'}")
                            console.print()
                            console.print("[dim]Use /debug preserve <feature> to add features that must work[/dim]")
                            console.print("[dim]Use /debug hypothesis <text> to set current hypothesis[/dim]")
                            # Save to memory
                            if memory:
                                memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "preserve":
                        # /debug preserve <feature>
                        if not _debug_session:
                            console.print("[yellow]No active debug session. Use /debug start first.[/yellow]")
                            continue
                        feature = parts[2] if len(parts) > 2 else ""
                        if not feature:
                            feature = Prompt.ask("Feature to preserve")
                        if feature:
                            _debug_session.add_must_preserve(feature)
                            console.print(f"[green]Added to preservation list: {feature}[/green]")
                            if memory:
                                memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "hypothesis":
                        # /debug hypothesis <hypothesis text>
                        if not _debug_session:
                            console.print("[yellow]No active debug session. Use /debug start first.[/yellow]")
                            continue
                        hypothesis = parts[2] if len(parts) > 2 else ""
                        if not hypothesis:
                            hypothesis = Prompt.ask("Current hypothesis")
                        if hypothesis:
                            _debug_session.set_hypothesis(hypothesis)
                            console.print(f"[green]Hypothesis set: {hypothesis}[/green]")
                            if memory:
                                memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "attempt":
                        # /debug attempt <description>
                        if not _debug_session:
                            console.print("[yellow]No active debug session. Use /debug start first.[/yellow]")
                            continue
                        description = parts[2] if len(parts) > 2 else ""
                        if not description:
                            description = Prompt.ask("Describe the fix attempt")
                        if description:
                            attempt = _debug_session.start_attempt(description)
                            console.print(f"\n[bold]Attempt #{attempt.id} Started[/bold]")
                            console.print(f"  Description: {description}")
                            console.print(f"  Rollback to: {attempt.rollback_commit or 'N/A'}")
                            console.print()
                            console.print("[dim]Use /debug fail <reason> or /debug success after testing[/dim]")
                            if memory:
                                memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "fail":
                        # /debug fail <reason>
                        if not _debug_session:
                            console.print("[yellow]No active debug session.[/yellow]")
                            continue
                        # Find pending attempt
                        pending = [a for a in _debug_session.attempts if a.result == AttemptResult.PENDING]
                        if not pending:
                            console.print("[yellow]No pending attempt to mark as failed.[/yellow]")
                            continue
                        reason = parts[2] if len(parts) > 2 else ""
                        if not reason:
                            reason = Prompt.ask("Why did it fail?")
                        if reason:
                            attempt = pending[-1]
                            _debug_session.complete_attempt(
                                attempt.id,
                                AttemptResult.FAILED,
                                reason
                            )
                            console.print(f"[red]Attempt #{attempt.id} marked as FAILED[/red]")
                            console.print(f"  Reason: {reason}")
                            if memory:
                                memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "partial":
                        # /debug partial <what helped>
                        if not _debug_session:
                            console.print("[yellow]No active debug session.[/yellow]")
                            continue
                        pending = [a for a in _debug_session.attempts if a.result == AttemptResult.PENDING]
                        if not pending:
                            console.print("[yellow]No pending attempt to mark.[/yellow]")
                            continue
                        reason = parts[2] if len(parts) > 2 else ""
                        if not reason:
                            reason = Prompt.ask("How did it help?")
                        if reason:
                            attempt = pending[-1]
                            _debug_session.complete_attempt(
                                attempt.id,
                                AttemptResult.PARTIAL,
                                reason
                            )
                            console.print(f"[yellow]Attempt #{attempt.id} marked as PARTIAL[/yellow]")
                            console.print(f"  Result: {reason}")
                            if memory:
                                memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "success":
                        # /debug success
                        if not _debug_session:
                            console.print("[yellow]No active debug session.[/yellow]")
                            continue
                        pending = [a for a in _debug_session.attempts if a.result == AttemptResult.PENDING]
                        if not pending:
                            console.print("[yellow]No pending attempt to mark as success.[/yellow]")
                            continue
                        attempt = pending[-1]
                        _debug_session.complete_attempt(
                            attempt.id,
                            AttemptResult.SUCCESS,
                            "Fix worked"
                        )
                        console.print(f"[green bold]Attempt #{attempt.id} marked as SUCCESS![/green bold]")
                        if memory:
                            memory.save_debug_session(_debug_session.to_dict())

                    elif subcommand == "status":
                        # /debug status - Show debug session state
                        if not _debug_session:
                            console.print("[dim]No active debug session.[/dim]")
                            console.print("[dim]Use /debug start <problem> to begin.[/dim]")
                            continue

                        console.print(f"\n[bold]Debug Session Status[/bold]")
                        console.print(f"  Problem: {_debug_session.problem}")
                        console.print(f"  Hypothesis: {_debug_session.current_hypothesis or '(not set)'}")
                        console.print(f"  Features to preserve: {len(_debug_session.must_preserve)}")
                        for feat in _debug_session.must_preserve:
                            console.print(f"    • {feat}")
                        console.print()
                        console.print(f"  [bold]Attempts ({len(_debug_session.attempts)}):[/bold]")
                        for attempt in _debug_session.attempts:
                            if attempt.result == AttemptResult.SUCCESS:
                                icon = "[green]✓[/green]"
                            elif attempt.result == AttemptResult.FAILED:
                                icon = "[red]✗[/red]"
                            elif attempt.result == AttemptResult.PARTIAL:
                                icon = "[yellow]~[/yellow]"
                            else:
                                icon = "[blue]?[/blue]"
                            console.print(f"    {icon} #{attempt.id}: {attempt.description}")
                            if attempt.reason:
                                console.print(f"       [dim]{attempt.reason}[/dim]")
                        console.print()

                    elif subcommand == "context":
                        # /debug context - Show what gets injected into Claude
                        if not _debug_session:
                            console.print("[yellow]No active debug session.[/yellow]")
                            continue
                        context = _debug_session.get_context_for_claude()
                        console.print(Panel(context, title="Debug Context for Claude", border_style="blue"))

                    elif subcommand == "end":
                        # /debug end - End the debug session
                        if not _debug_session:
                            console.print("[yellow]No active debug session.[/yellow]")
                            continue
                        _debug_session.is_active = False
                        if memory:
                            memory.save_debug_session(_debug_session.to_dict())
                        console.print("[green]Debug session ended.[/green]")
                        console.print(_debug_session.get_summary())
                        _debug_session = None

                    else:
                        console.print("[dim]Usage:[/dim]")
                        console.print("  /debug start <problem>      - Start a debug session")
                        console.print("  /debug preserve <feature>   - Add feature to preserve")
                        console.print("  /debug hypothesis <text>    - Set current hypothesis")
                        console.print("  /debug attempt <description>- Start a fix attempt")
                        console.print("  /debug fail <reason>        - Mark attempt as failed")
                        console.print("  /debug partial <result>     - Mark attempt as partially worked")
                        console.print("  /debug success              - Mark attempt as successful")
                        console.print("  /debug status               - Show session status")
                        console.print("  /debug context              - Show Claude context")
                        console.print("  /debug end                  - End debug session")

                elif cmd.startswith("/rollback"):
                    # /rollback [attempt_id] - Rollback to before an attempt
                    if not _debug_session:
                        console.print("[yellow]No active debug session.[/yellow]")
                        continue
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            attempt_id = int(parts[1])
                            if _debug_session.rollback_to_attempt(attempt_id):
                                console.print(f"[green]Rolled back to before attempt #{attempt_id}[/green]")
                            else:
                                console.print(f"[red]Rollback failed - attempt #{attempt_id} not found or no checkpoint[/red]")
                        except ValueError:
                            if parts[1] == "start":
                                if _debug_session.rollback_to_start():
                                    console.print("[green]Rolled back to session start[/green]")
                                else:
                                    console.print("[red]Rollback failed - no initial checkpoint[/red]")
                            else:
                                console.print("[red]Invalid attempt ID[/red]")
                    else:
                        console.print("[dim]Usage:[/dim]")
                        console.print("  /rollback <attempt_id>  - Rollback to before attempt")
                        console.print("  /rollback start         - Rollback to session start")
                        console.print()
                        console.print("[dim]Available rollback points:[/dim]")
                        console.print(f"  start ({_debug_session.initial_commit})")
                        for attempt in _debug_session.attempts:
                            if attempt.rollback_commit:
                                console.print(f"  #{attempt.id} ({attempt.rollback_commit})")
                else:
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                continue

            # Input size validation - warn if very large
            MAX_INPUT_SIZE = 50000  # ~50KB reasonable limit
            if len(user_input) > MAX_INPUT_SIZE:
                console.print(f"[yellow]Warning: Input is very large ({len(user_input):,} chars). Truncating to {MAX_INPUT_SIZE:,}.[/yellow]")
                user_input = user_input[:MAX_INPUT_SIZE] + "\n\n[... truncated due to size ...]"

            # Persist user message to new system
            persist_message(context.repo_session_id, MessageRole.USER, user_input, MessageType.CHAT)

            # Update heartbeat in new persistence
            if _repository and context.repo_session_id:
                try:
                    _repository.update_heartbeat(context.repo_session_id)
                except Exception:
                    pass  # Non-critical

            # Process request through GLM
            asyncio.run(process_user_request(glm_client, context, project, user_input, memory))

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            break


@app.command()
def main(
    skip_ping: bool = typer.Option(False, "--skip-ping", help="Skip GLM API ping (faster startup)"),
    tui: bool = typer.Option(False, "--tui", help="Use Textual TUI with escape-to-cancel"),
) -> None:
    """Start Vibe Orchestrator."""
    global _glm_client, _memory, _repository

    # Step 1: Validate startup
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            description="Validating systems..." + (" (pinging GLM)" if not skip_ping else ""),
            total=None,
        )
        results = validate_startup(ping_glm=not skip_ping)

    all_passed = show_startup_panel(results)

    if not all_passed:
        console.print("[bold red]Startup validation failed![/bold red]")
        console.print("Fix the issues above and try again.")
        raise typer.Exit(1)

    # Step 2: Load configuration
    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    # Step 3: Initialize GLM client
    try:
        api_key = get_openrouter_key()
        _glm_client = GLMClient(api_key)
    except ConfigError as e:
        console.print(f"[bold red]GLM initialization failed:[/bold red] {e}")
        raise typer.Exit(1)

    # Step 4: Initialize optional integrations
    global _perplexity, _github

    # Perplexity for research (optional)
    _perplexity = PerplexityClient()
    if not _perplexity.is_available:
        console.print("  [dim]Perplexity: not configured (PERPLEXITY_API_KEY)[/dim]")

    # GitHub CLI for repo operations (optional)
    _github = GitHubOps()
    if not _github.check_auth():
        _github = None

    # Step 5: Project selection
    selection = show_project_list(config)
    if selection is None:
        raise typer.Exit(0)

    project = config.get_project(selection)

    if not project.exists():
        console.print(f"[bold red]Project directory does not exist:[/bold red] {project.path}")
        raise typer.Exit(1)

    # Step 6: Initialize persistence and start session
    memory_items = 0
    repo_session_id = ""

    # Initialize new unified persistence layer
    try:
        _repository = VibeRepository()
        _repository.initialize()

        # Get or create project in new persistence
        repo_project = _repository.get_or_create_project(
            name=project.name,
            path=str(project.path),
            description=project.description or "",
        )

        # Check for orphaned sessions (crash recovery)
        orphans = _repository.get_orphaned_sessions()
        if orphans:
            console.print(f"[yellow]⚠ Found {len(orphans)} orphaned session(s) from previous crash(es)[/yellow]")
            for orphan in orphans[:3]:  # Show max 3
                console.print(f"  [dim]- Session {orphan.id[:8]}... started {orphan.started_at}[/dim]")

        # Start new session in persistence layer
        global _current_repo_session_id
        repo_session = _repository.start_session(repo_project.id)
        repo_session_id = repo_session.id
        _current_repo_session_id = repo_session_id  # Track for signal handler
        console.print(f"  [dim]Persistence: session {repo_session_id[:8]}...[/dim]")

    except Exception as e:
        logger.error(f"Failed to initialize persistence: {e}")
        console.print(f"[yellow]Warning: New persistence not available: {e}[/yellow]")
        _repository = None

    # Also initialize old memory system (for backward compatibility)
    try:
        _memory = VibeMemory(project.name)
        _memory.start_session(f"Vibe session for {project.name}")

        # Set logging context for session correlation
        set_session_id(_memory.session_id)
        set_project_name(project.name)

        # Log session start
        session_logger.info(SessionLogEntry(
            timestamp=now_iso(),
            session_id=_memory.session_id,
            event_type="start",
            project_name=project.name,
        ).to_json())

        # Load existing context items count
        context_items = _memory.load_project_context(limit=100)
        memory_items = len(context_items)

        # Load task history from database into in-memory TaskHistory
        TaskHistory.load_from_memory(_memory)
        stats = TaskHistory.get_stats()
        if stats["total_tasks"] > 0:
            console.print(f"  [dim]TaskHistory: {stats['total_tasks']} tasks loaded ({stats['completed']} completed, {stats['failed']} failed)[/dim]")
    except MemoryConnectionError as e:
        console.print(f"[yellow]Warning: Old memory not available: {e}[/yellow]")
        _memory = None
        # Still set project name for logging even without memory
        set_project_name(project.name)

    # Step 7: Initialize session context
    # Prefer old memory session ID for backward compatibility, fall back to new repo
    session_id = (_memory.session_id if _memory else "") or repo_session_id
    context = SessionContext(
        state=SessionState.IDLE,
        project_name=project.name,
        project_path=project.path,
        session_id=session_id,
        repo_session_id=repo_session_id,  # New persistence session
    )

    show_project_loaded(project, memory_items=memory_items)

    # Step 8: Enter conversation loop or TUI
    if tui:
        from vibe.tui import run_tui
        console.print("[bold cyan]Starting Textual TUI...[/bold cyan]")
        console.print("[dim]Press Escape to cancel operations, Ctrl+C to quit[/dim]")
        run_tui(
            config=config,
            project=project,
            glm_client=_glm_client,
            context=context,
            memory=_memory,
        )
    else:
        conversation_loop(context, config, project, _glm_client, _memory)


@app.command()
def add(
    name: str = typer.Argument(..., help="Project name"),
    path: str = typer.Argument(..., help="Path to project directory"),
    description: str = typer.Option("", help="Project description"),
) -> None:
    """Add a new project to Vibe."""
    try:
        config = load_config()
        project = Project(name=name, path=path, description=description)

        if not project.exists():
            console.print(f"[yellow]Warning: Directory does not exist: {project.path}[/yellow]")

        config.add_project(project)
        save_config(config)
        console.print(f"[green]Added project '{name}'[/green]")

    except ConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def remove(name: str = typer.Argument(..., help="Project name to remove")) -> None:
    """Remove a project from Vibe."""
    try:
        config = load_config()
        project = config.remove_project(name)
        save_config(config)
        console.print(f"[green]Removed project '{project.name}'[/green]")

    except ConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command(name="list")
def list_projects() -> None:
    """List all registered projects."""
    try:
        config = load_config()
        show_project_list(config)
    except ConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def restore(
    session_id: str = typer.Argument(None, help="Session ID to restore (or 'list' to see all)"),
    show_messages: bool = typer.Option(False, "--messages", "-m", help="Show conversation messages"),
    show_tasks: bool = typer.Option(False, "--tasks", "-t", help="Show task details"),
) -> None:
    """Recover from a crashed session.

    Shows orphaned/crashed sessions and their context for recovery.

    Examples:
        vibe restore              # List all orphaned sessions
        vibe restore list         # Same as above
        vibe restore abc123       # Show details for session abc123
        vibe restore abc123 -m    # Show with conversation messages
        vibe restore abc123 -t    # Show with task details
    """
    from rich.table import Table
    from rich.panel import Panel

    # Initialize repository
    try:
        repo = VibeRepository()
        repo.initialize()
    except Exception as e:
        console.print(f"[red]Failed to initialize persistence:[/red] {e}")
        raise typer.Exit(1)

    # List mode - show all orphaned sessions
    if session_id is None or session_id.lower() == "list":
        orphans = repo.get_orphaned_sessions()

        if not orphans:
            console.print("[green]No crashed/orphaned sessions found.[/green]")
            console.print("[dim]All sessions ended gracefully.[/dim]")
            return

        console.print(f"\n[bold yellow]Found {len(orphans)} crashed session(s):[/bold yellow]\n")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Session ID", style="cyan")
        table.add_column("Project", style="green")
        table.add_column("Started", style="dim")
        table.add_column("Last Heartbeat", style="dim")
        table.add_column("Tasks", style="yellow")

        for orphan in orphans:
            session_short = orphan.get("session_id", "")[:12]
            project = orphan.get("project_name", "unknown")
            started = orphan.get("started_at", "")[:16] if orphan.get("started_at") else "-"
            heartbeat = orphan.get("last_heartbeat", "")[:16] if orphan.get("last_heartbeat") else "-"
            tasks_completed = orphan.get("tasks_completed", 0)
            tasks_failed = orphan.get("tasks_failed", 0)
            task_info = f"{tasks_completed} done, {tasks_failed} failed"

            table.add_row(session_short, project, started, heartbeat, task_info)

        console.print(table)
        console.print("\n[dim]Use 'vibe restore <session_id>' to see full context[/dim]")
        console.print("[dim]Use 'vibe restore <session_id> -m -t' for messages and tasks[/dim]")
        return

    # Detail mode - show specific session context
    # Try to find session by partial ID
    full_session_id = session_id
    if len(session_id) < 36:
        # Search for matching session
        orphans = repo.get_orphaned_sessions()
        matches = [o for o in orphans if o.get("session_id", "").startswith(session_id)]
        if len(matches) == 0:
            # Also check all sessions
            all_sessions = repo.list_sessions(limit=100)
            matches = [{"session_id": s.id} for s in all_sessions if s.id.startswith(session_id)]

        if len(matches) == 0:
            console.print(f"[red]No session found starting with '{session_id}'[/red]")
            raise typer.Exit(1)
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]")
            for m in matches[:5]:
                console.print(f"  {m.get('session_id', '')[:12]}")
            raise typer.Exit(1)
        else:
            full_session_id = matches[0].get("session_id", session_id)

    # Get full recovery context
    context = repo.get_session_recovery_context(full_session_id)

    if "error" in context:
        console.print(f"[red]{context['error']}[/red]")
        raise typer.Exit(1)

    session = context["session"]
    project = context["project"]
    summary = context["summary"]
    tasks_info = context["tasks"]

    # Display session summary
    console.print(Panel(
        f"[bold]Session:[/bold] {session.id[:12]}...\n"
        f"[bold]Project:[/bold] {project.name if project else 'unknown'}\n"
        f"[bold]Status:[/bold] {session.status}\n"
        f"[bold]Started:[/bold] {session.started_at}\n"
        f"[bold]Last Heartbeat:[/bold] {session.last_heartbeat_at or 'never'}\n"
        f"\n"
        f"[bold]Messages:[/bold] {summary['total_messages']} ({summary['user_messages']} from user)\n"
        f"[bold]Tasks:[/bold] {summary['pending_tasks']} pending, {summary['completed_tasks']} completed, {summary['failed_tasks']} failed",
        title="[bold cyan]Recovery Context[/bold cyan]",
        border_style="cyan",
    ))

    # Show last user request
    if context["last_request"]:
        console.print(Panel(
            context["last_request"][:500] + ("..." if len(context["last_request"]) > 500 else ""),
            title="[bold yellow]Last User Request[/bold yellow]",
            border_style="yellow",
        ))

    # Show pending tasks
    if tasks_info["pending"]:
        console.print("\n[bold red]Pending/In-Progress Tasks:[/bold red]")
        for i, task in enumerate(tasks_info["pending"], 1):
            status_color = "yellow" if task.status == TaskStatus.PENDING else "blue"
            console.print(f"  [{status_color}]{i}. [{task.status.value}][/{status_color}] {task.description[:80]}")

    # Show messages if requested
    if show_messages:
        messages = context["messages"]
        if messages:
            console.print("\n[bold]Conversation History:[/bold]")
            for msg in messages[-20:]:  # Last 20 messages
                role_color = {
                    "user": "green",
                    "glm": "cyan",
                    "system": "dim",
                    "assistant": "blue",
                }.get(msg.role.value, "white")
                content_preview = msg.content[:100].replace("\n", " ")
                console.print(f"  [{role_color}][{msg.role.value}][/{role_color}] {content_preview}...")

    # Show task details if requested
    if show_tasks:
        all_tasks = tasks_info["pending"] + tasks_info["completed"] + tasks_info["failed"]
        if all_tasks:
            console.print("\n[bold]All Tasks:[/bold]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("#", style="dim", width=3)
            table.add_column("Status", width=10)
            table.add_column("Description")
            table.add_column("Created", style="dim", width=16)

            for i, task in enumerate(all_tasks, 1):
                status_style = {
                    "completed": "green",
                    "failed": "red",
                    "pending": "yellow",
                    "queued": "yellow",
                    "executing": "blue",
                    "in_progress": "blue",
                }.get(task.status.value, "white")

                table.add_row(
                    str(i),
                    f"[{status_style}]{task.status.value}[/{status_style}]",
                    task.description[:60],
                    task.created_at[:16] if task.created_at else "-",
                )

            console.print(table)

    # Offer recovery action
    console.print("\n[bold]Recovery Options:[/bold]")
    console.print(f"  1. Start vibe with this project: [cyan]vibe {project.name if project else ''}[/cyan]")
    console.print("     (Orphaned session will be auto-marked as recovered)")
    console.print("  2. The pending tasks above can be re-requested in the new session")


@app.command()
def ping() -> None:
    """Test GLM API connectivity."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set[/red]")
        raise typer.Exit(1)

    with console.status("[bold blue]Pinging GLM...[/bold blue]"):
        success, message = ping_glm_sync(api_key, timeout=15.0)

    if success:
        console.print(f"[green]GLM API OK:[/green] {message}")
    else:
        console.print(f"[red]GLM API failed:[/red] {message}")
        raise typer.Exit(1)


@app.command()
def logs(
    log_type: str = typer.Option("all", "--type", "-t", help="Log type: glm, claude, session, all"),
    since: str = typer.Option(None, "--since", "-s", help="Time filter (ISO or relative: 1h, 30m, 2d)"),
    session: str = typer.Option(None, "--session", help="Filter by session ID"),
    tail: int = typer.Option(20, "--tail", "-n", help="Show last N entries"),
    stats: bool = typer.Option(False, "--stats", help="Show statistics instead of entries"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs in real-time (like tail -f)"),
) -> None:
    """View and analyze Vibe logs."""
    from vibe.logging.viewer import follow_logs

    def print_entry(entry: dict) -> None:
        """Print a single log entry with color."""
        line = format_entry_line(entry)
        source = entry.get("_source", "")
        if source == "glm":
            console.print(f"[cyan]{line}[/cyan]")
        elif source == "claude":
            if entry.get("success"):
                console.print(f"[green]{line}[/green]")
            else:
                console.print(f"[red]{line}[/red]")
        else:
            console.print(f"[dim]{line}[/dim]")

    try:
        if follow:
            # Live follow mode
            console.print(f"[dim]Following {log_type} logs... (Ctrl+C to stop)[/dim]")
            try:
                for entry in follow_logs(log_type=log_type):
                    print_entry(entry)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped following logs[/dim]")
            return

        # Query logs with filters
        entries = query_logs(
            log_type=log_type,
            since=since,
            session_id=session,
            limit=tail if not stats else 1000,  # Get more for stats
        )

        if not entries:
            console.print("[dim]No log entries found[/dim]")
            return

        if stats:
            # Calculate and display statistics
            stats_data = calculate_stats(entries)
            console.print(format_stats(stats_data))
        else:
            # Display log entries (most recent last)
            for entry in reversed(entries[:tail]):
                print_entry(entry)

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error reading logs:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
