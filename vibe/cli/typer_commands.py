"""
Vibe CLI - Typer Commands

CLI commands for managing projects and sessions.
"""

import logging
import os
import signal
import sys
import types
from typing import NoReturn

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from vibe.config import (
    Project,
    VibeConfig,
    get_openrouter_key,
    load_config,
    save_config,
)
from vibe.exceptions import ConfigError
from vibe.glm.client import GLMClient, ping_glm_sync
from vibe.integrations import PerplexityClient, GitHubOps
from vibe.logging import set_session_id, set_project_name, session_logger, SessionLogEntry, now_iso
from vibe.logging.viewer import query_logs, calculate_stats, format_entry_line, format_stats, follow_logs
from vibe.memory.keeper import VibeMemory
from vibe.memory.task_history import TaskHistory
from vibe.persistence.models import SessionStatus, TaskStatus
from vibe.persistence.repository import VibeRepository
from vibe.state import SessionContext, SessionState

from vibe.cli.startup import validate_startup, show_startup_panel
from vibe.cli.project import show_project_list, show_project_loaded
from vibe.cli.interactive import conversation_loop

logger = logging.getLogger(__name__)
console = Console()

# Typer app for CLI
app = typer.Typer(
    name="vibe",
    help="GLM-4.7 as brain, Claude Code as worker - AI pair programming orchestrator",
    add_completion=False,
    invoke_without_command=True,
)

# Global clients
_glm_client: GLMClient | None = None
_memory: VibeMemory | None = None
_repository: VibeRepository | None = None
_perplexity: PerplexityClient | None = None
_github: GitHubOps | None = None
_current_repo_session_id: str | None = None


def handle_shutdown(signum: int, frame: types.FrameType | None) -> NoReturn:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    console.print("\n[yellow]Shutting down Vibe...[/yellow]")
    # End sessions and save state
    if _memory and _memory.session_id:
        try:
            _memory.end_session("Session ended via signal")
        except Exception:
            pass
    # End new persistence session
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


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


@app.callback()
def callback(
    ctx: typer.Context,
    skip_ping: bool = typer.Option(False, "--skip-ping", help="Skip GLM API ping (faster startup)"),
    tui: bool = typer.Option(False, "--tui", help="Use Textual TUI with escape-to-cancel"),
) -> None:
    """
    GLM-4.7 as brain, Claude Code as worker - AI pair programming orchestrator.

    Run without a command to start the interactive orchestrator.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand - run the main orchestrator
        _start_orchestrator(skip_ping=skip_ping, tui=tui)


def _start_orchestrator(
    skip_ping: bool = False,
    tui: bool = False,
) -> None:
    """Start Vibe Orchestrator (internal implementation)."""
    global _glm_client, _memory, _repository, _perplexity, _github, _current_repo_session_id

    # Step 1: Validate startup
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(
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
    _perplexity = PerplexityClient()
    if not _perplexity.is_available:
        console.print("  [dim]Perplexity: not configured (PERPLEXITY_API_KEY)[/dim]")

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

        repo_project = _repository.get_or_create_project(
            name=project.name,
            path=str(project.path),
            description=project.description or "",
        )

        # Check for orphaned sessions
        orphans = _repository.get_orphaned_sessions()
        if orphans:
            console.print(f"[yellow]⚠ Found {len(orphans)} orphaned session(s) from previous crash(es)[/yellow]")
            for orphan in orphans[:3]:
                console.print(f"  [dim]- Session {orphan.id[:8]}... started {orphan.started_at}[/dim]")

        # Start new session
        repo_session = _repository.start_session(repo_project.id)
        repo_session_id = repo_session.id
        _current_repo_session_id = repo_session_id
        console.print(f"  [dim]Persistence: session {repo_session_id[:8]}...[/dim]")

    except Exception as e:
        logger.error(f"Failed to initialize persistence: {e}")
        console.print(f"[yellow]Warning: New persistence not available: {e}[/yellow]")
        _repository = None

    # Also initialize old memory system
    try:
        _memory = VibeMemory(project.name)
        _memory.start_session(f"Vibe session for {project.name}")

        set_session_id(_memory.session_id)
        set_project_name(project.name)

        session_logger.info(SessionLogEntry(
            timestamp=now_iso(),
            session_id=_memory.session_id,
            event_type="start",
            project_name=project.name,
        ).to_json())

        context_items = _memory.load_project_context(limit=100)
        memory_items = len(context_items)

        TaskHistory.load_from_memory(_memory)
        stats = TaskHistory.get_stats()
        if stats["total_tasks"] > 0:
            console.print(f"  [dim]TaskHistory: {stats['total_tasks']} tasks loaded[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Old memory not available: {e}[/yellow]")
        _memory = None
        set_project_name(project.name)

    # Step 7: Initialize session context
    session_id = (_memory.session_id if _memory else "") or repo_session_id
    context = SessionContext(
        state=SessionState.IDLE,
        project_name=project.name,
        project_path=project.path,
        session_id=session_id,
        repo_session_id=repo_session_id,
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
        conversation_loop(
            context=context,
            config=config,
            project=project,
            glm_client=_glm_client,
            memory=_memory,
            repository=_repository,
            perplexity=_perplexity,
            github=_github,
        )


@app.command(hidden=True)
def main(
    skip_ping: bool = typer.Option(False, "--skip-ping", help="Skip GLM API ping"),
    tui: bool = typer.Option(False, "--tui", help="Use Textual TUI"),
) -> None:
    """Start Vibe Orchestrator (use 'vibe' directly instead)."""
    _start_orchestrator(skip_ping=skip_ping, tui=tui)


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
    """Recover from a crashed session."""
    # Initialize repository
    try:
        repo = VibeRepository()
        repo.initialize()
    except Exception as e:
        console.print(f"[red]Failed to initialize persistence:[/red] {e}")
        raise typer.Exit(1)

    # List mode
    if session_id is None or session_id.lower() == "list":
        orphans = repo.get_orphaned_sessions()

        if not orphans:
            console.print("[green]No crashed/orphaned sessions found.[/green]")
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
        return

    # Detail mode - find session
    full_session_id = session_id
    if len(session_id) < 36:
        orphans = repo.get_orphaned_sessions()
        matches = [o for o in orphans if o.get("session_id", "").startswith(session_id)]
        if len(matches) == 0:
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

    # Get recovery context
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
        f"[bold]Tasks:[/bold] {summary['pending_tasks']} pending, {summary['completed_tasks']} completed",
        title="[bold cyan]Recovery Context[/bold cyan]",
        border_style="cyan",
    ))

    if context["last_request"]:
        console.print(Panel(
            context["last_request"][:500] + ("..." if len(context["last_request"]) > 500 else ""),
            title="[bold yellow]Last User Request[/bold yellow]",
            border_style="yellow",
        ))

    if tasks_info["pending"]:
        console.print("\n[bold red]Pending/In-Progress Tasks:[/bold red]")
        for i, task in enumerate(tasks_info["pending"], 1):
            status_color = "yellow" if task.status == TaskStatus.PENDING else "blue"
            console.print(f"  [{status_color}]{i}. [{task.status.value}][/{status_color}] {task.description[:80]}")

    if show_messages:
        messages = context["messages"]
        if messages:
            console.print("\n[bold]Conversation History:[/bold]")
            for msg in messages[-20:]:
                role_color = {"user": "green", "glm": "cyan", "system": "dim"}.get(msg.role.value, "white")
                content_preview = msg.content[:100].replace("\n", " ")
                console.print(f"  [{role_color}][{msg.role.value}][/{role_color}] {content_preview}...")

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
                    "completed": "green", "failed": "red", "pending": "yellow", "executing": "blue"
                }.get(task.status.value, "white")

                table.add_row(
                    str(i),
                    f"[{status_style}]{task.status.value}[/{status_style}]",
                    task.description[:60],
                    task.created_at[:16] if task.created_at else "-",
                )

            console.print(table)

    console.print("\n[bold]Recovery Options:[/bold]")
    console.print(f"  1. Start vibe with this project: [cyan]vibe {project.name if project else ''}[/cyan]")
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
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs in real-time"),
) -> None:
    """View and analyze Vibe logs."""
    def print_entry(entry: dict) -> None:
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
            console.print(f"[dim]Following {log_type} logs... (Ctrl+C to stop)[/dim]")
            try:
                for entry in follow_logs(log_type=log_type):
                    print_entry(entry)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped following logs[/dim]")
            return

        entries = query_logs(
            log_type=log_type,
            since=since,
            session_id=session,
            limit=tail if not stats else 1000,
        )

        if not entries:
            console.print("[dim]No log entries found[/dim]")
            return

        if stats:
            stats_data = calculate_stats(entries)
            console.print(format_stats(stats_data))
        else:
            for entry in reversed(entries[:tail]):
                print_entry(entry)

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error reading logs:[/red] {e}")
        raise typer.Exit(1)


def run() -> None:
    """Entry point wrapper that invokes the Typer app."""
    app()


if __name__ == "__main__":
    run()
