#!/usr/bin/env python3
"""
Vibe Orchestrator - CLI Entry Point

Main entry point for the `vibe` command.
Shows startup validation, project selection, and conversation interface.
"""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import NoReturn

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
_perplexity: PerplexityClient | None = None
_github: GitHubOps | None = None
_debug_session: DebugSession | None = None


def handle_shutdown(signum: int, frame: types.FrameType | None) -> NoReturn:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    console.print("\n[yellow]Shutting down Vibe...[/yellow]")
    # End session and save state
    if _memory and _memory.session_id:
        try:
            _memory.end_session("Session ended via signal")
        except Exception:
            pass  # Best effort on shutdown
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def validate_startup(ping_glm: bool = True) -> dict[str, tuple[bool, str]]:
    """
    Validate all required systems are available.

    Args:
        ping_glm: Whether to actually ping GLM API (slower but more reliable)

    Returns:
        Dict mapping system name to (success, message) tuple
    """
    results: dict[str, tuple[bool, str]] = {}

    # 1. Check OpenRouter API and optionally ping GLM
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        results["OpenRouter API"] = (False, "OPENROUTER_API_KEY not set")
    elif ping_glm:
        # Actually ping GLM to verify API works
        success, message = ping_glm_sync(api_key, timeout=15.0)
        if success:
            results["OpenRouter API"] = (True, f"GLM-4 ({message})")
        else:
            results["OpenRouter API"] = (False, message)
    else:
        results["OpenRouter API"] = (True, "key configured")

    # 2. Check Claude CLI
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
            results["Claude Code CLI"] = (True, version)
        except Exception as e:
            results["Claude Code CLI"] = (False, str(e))
    else:
        results["Claude Code CLI"] = (False, "not installed")

    # 3. Check memory-keeper database
    if MEMORY_DB_PATH.exists():
        results["Memory-keeper"] = (True, "database found")
    else:
        results["Memory-keeper"] = (False, f"not found")

    # 4. Check GitHub CLI
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
                # Extract username from output
                for line in result.stderr.split("\n"):
                    if "Logged in" in line:
                        results["GitHub CLI"] = (True, "authenticated")
                        break
                else:
                    results["GitHub CLI"] = (True, "authenticated")
            else:
                results["GitHub CLI"] = (False, "not authenticated")
        except Exception as e:
            results["GitHub CLI"] = (False, str(e))
    else:
        results["GitHub CLI"] = (False, "not installed")

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
    Load project context for GLM (starmap, recent changes, etc.)

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
            content = starmap_path.read_text()[:2000]  # Limit to 2000 chars
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

    return "\n".join(context_parts)


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

    # Progress callback for live updates
    progress_lines: list[str] = []

    def on_progress(text: str) -> None:
        progress_lines.append(text)

    def on_tool_call(tool_call: ToolCall) -> None:
        console.print(f"    [dim]→ {tool_call.name}[/dim]")

    console.print(
        Panel(
            f"[bold]Task {task_num}/{total_tasks}:[/bold] {description}",
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
    # Get git diff for changed files
    diff = get_git_diff(project_path, result.file_changes)

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
        console.print(f"    {result.result[:200]}")

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

    # Show cost
    if result.cost_usd > 0:
        console.print(f"  [dim]Cost: ${result.cost_usd:.4f} | Duration: {result.duration_ms}ms[/dim]")


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
    # Load project context
    project_context = load_project_context(project)

    # Add user message to history
    context.add_glm_message("user", user_request)

    # Check if clarification is needed
    console.print()
    with console.status("[bold blue]GLM analyzing request...[/bold blue]"):
        clarification = await glm_client.ask_clarification(user_request, project_context)

    if clarification:
        # GLM needs more info
        console.print(
            Panel(
                clarification,
                title="[bold yellow]GLM needs clarification[/bold yellow]",
                border_style="yellow",
            )
        )
        context.add_glm_message("assistant", clarification)
        return

    # Decompose into tasks
    console.print()
    with console.status("[bold blue]GLM decomposing into tasks...[/bold blue]"):
        try:
            tasks = await glm_client.decompose_task(user_request, project_context)
        except Exception as e:
            console.print(f"[red]Error decomposing task: {e}[/red]")
            return

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

    # Ask for confirmation before executing
    confirm = Prompt.ask(
        "Execute these tasks?",
        choices=["y", "n"],
        default="y",
    )

    if confirm.lower() != "y":
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
    try:
        executor = ClaudeExecutor(
            project_path=project.path,
            timeout_tier="code",
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

    for i, task in enumerate(tasks, 1):
        try:
            # Get debug context if in debug session
            debug_ctx = _debug_session.get_context_for_claude() if _debug_session else None

            # Execute with Claude
            result, tool_verification = await execute_task_with_claude(
                executor=executor,
                task=task,
                task_num=i,
                total_tasks=len(tasks),
                debug_context=debug_ctx,
            )

            if not result.success:
                console.print(f"[red]Task failed: {result.error}[/red]")
                failed += 1
                context.add_error(result.error or "Unknown error")
                # Save failure to memory
                if memory:
                    memory.save_task_result(
                        task_description=task.get("description", ""),
                        success=False,
                        summary=result.error or "Unknown error",
                    )
                continue

            # Check if tool verification failed (missing required tools)
            if not tool_verification["passed"] and tool_verification.get("missing_required"):
                console.print(
                    f"[yellow]Warning: Task may be incomplete - missing required tools[/yellow]"
                )

            # Review with GLM
            context.transition_to(SessionState.REVIEWING)
            review = await review_with_glm(
                glm_client=glm_client,
                task=task,
                result=result,
                project_path=project.path,
            )

            # If tool verification failed, add that to review issues
            if not tool_verification["passed"]:
                existing_issues = review.get("issues", [])
                existing_issues.append(f"Tool verification: {tool_verification['feedback']}")
                review["issues"] = existing_issues

            # Show result
            show_task_result(result, review)

            if review.get("approved"):
                completed += 1
                context.add_completed_task(task.get("description", ""))
                # Track file changes and summaries for project updates
                all_file_changes.extend(result.file_changes)
                if result.result:
                    all_summaries.append(result.result)
                # Save success to memory
                if memory:
                    memory.save_task_result(
                        task_description=task.get("description", ""),
                        success=True,
                        summary=result.result or "",
                        files_changed=result.file_changes,
                    )
            else:
                failed += 1
                # Save rejection to memory
                if memory:
                    memory.save_task_result(
                        task_description=task.get("description", ""),
                        success=False,
                        summary=f"Rejected: {review.get('feedback', '')}",
                    )
                # TODO: Implement retry with feedback

            context.transition_to(SessionState.EXECUTING)

        except ClaudeError as e:
            console.print(f"[red]Claude error on task {i}: {e}[/red]")
            failed += 1
            context.add_error(str(e))

    # Final summary
    context.transition_to(SessionState.IDLE)
    console.print()
    console.print(
        Panel(
            f"[bold]Session Complete[/bold]\n\n"
            f"  Completed: [green]{completed}[/green]\n"
            f"  Failed: [red]{failed}[/red]\n"
            f"  Total: {len(tasks)}",
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
    """
    console.print("[bold]What do you want to work on?[/bold]")
    console.print("[dim]Type your request, or /help for commands, /quit to exit[/dim]")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd in ("/quit", "/exit", "/q"):
                    # End memory session before exiting
                    if memory and memory.session_id:
                        memory.end_session("Session ended by user")
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif cmd == "/help":
                    console.print("\n[bold]Commands:[/bold]")
                    console.print("  /quit       - Exit Vibe")
                    console.print("  /status     - Show session status")
                    console.print("  /usage      - Show GLM usage stats")
                    console.print("  /memory     - Show memory stats")
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

            # Process request through GLM
            asyncio.run(process_user_request(glm_client, context, project, user_input, memory))

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            break


@app.command()
def main(
    skip_ping: bool = typer.Option(False, "--skip-ping", help="Skip GLM API ping (faster startup)"),
) -> None:
    """Start Vibe Orchestrator."""
    global _glm_client, _memory

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

    # Step 6: Initialize memory and start session
    memory_items = 0
    try:
        _memory = VibeMemory(project.name)
        _memory.start_session(f"Vibe session for {project.name}")

        # Load existing context items count
        context_items = _memory.load_project_context(limit=100)
        memory_items = len(context_items)
    except MemoryConnectionError as e:
        console.print(f"[yellow]Warning: Memory not available: {e}[/yellow]")
        _memory = None

    # Step 7: Initialize session context
    context = SessionContext(
        state=SessionState.IDLE,
        project_name=project.name,
        project_path=project.path,
        session_id=_memory.session_id if _memory else "",
    )

    show_project_loaded(project, memory_items=memory_items)

    # Step 8: Enter conversation loop
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


if __name__ == "__main__":
    app()
