#!/usr/bin/env python3
"""
Vibe Orchestrator - CLI Entry Point

Main entry point for the `vibe` command.
Shows startup validation, project selection, and conversation interface.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
import types
from typing import NoReturn

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from vibe.config import (
    CONFIG_DIR,
    MEMORY_DB_PATH,
    Project,
    VibeConfig,
    load_config,
    save_config,
)
from vibe.exceptions import ConfigError, StartupError
from vibe.state import SessionContext, SessionState

# Rich console for terminal output
console = Console()

# Typer app for CLI
app = typer.Typer(
    name="vibe",
    help="GLM-4.7 as brain, Claude Code as worker - AI pair programming orchestrator",
    add_completion=False,
)


def handle_shutdown(signum: int, frame: types.FrameType | None) -> NoReturn:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    console.print("\n[yellow]Shutting down Vibe...[/yellow]")
    # TODO: Save session state to memory-keeper
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def validate_startup() -> dict[str, tuple[bool, str]]:
    """
    Validate all required systems are available.

    Returns:
        Dict mapping system name to (success, message) tuple
    """
    results: dict[str, tuple[bool, str]] = {}

    # 1. Check OpenRouter API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        results["OpenRouter API"] = (True, "GLM-4.7 ready")
    else:
        results["OpenRouter API"] = (False, "OPENROUTER_API_KEY not set")

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
        results["Memory-keeper"] = (False, f"database not found at {MEMORY_DB_PATH}")

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
                        results["GitHub CLI"] = (True, line.strip())
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


def conversation_loop(context: SessionContext, config: VibeConfig) -> None:
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
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif cmd == "/help":
                    console.print("\n[bold]Commands:[/bold]")
                    console.print("  /quit    - Exit Vibe")
                    console.print("  /status  - Show session status")
                    console.print("  /project - Switch project")
                    console.print("  /help    - Show this help")
                    console.print()
                elif cmd == "/status":
                    stats = context.get_stats()
                    console.print(f"\n[bold]Session Status:[/bold]")
                    console.print(f"  State: {stats['state']}")
                    console.print(f"  Project: {stats['project']}")
                    console.print(f"  Completed tasks: {stats['completed_tasks']}")
                    console.print(f"  Errors: {stats['error_count']}")
                    console.print()
                else:
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                continue

            # TODO: Send to GLM for processing
            console.print()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="GLM thinking...", total=None)
                # Placeholder - will be replaced with actual GLM call
                time.sleep(1)

            console.print(
                Panel(
                    "[dim]GLM integration not yet implemented.\n"
                    "This will decompose your request into tasks for Claude.[/dim]",
                    title="GLM Response",
                    border_style="yellow",
                )
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            break


@app.command()
def main() -> None:
    """Start Vibe Orchestrator."""
    # Step 1: Validate startup
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Validating systems...", total=None)
        results = validate_startup()

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

    # Step 3: Project selection
    selection = show_project_list(config)
    if selection is None:
        raise typer.Exit(0)

    project = config.get_project(selection)

    if not project.exists():
        console.print(f"[bold red]Project directory does not exist:[/bold red] {project.path}")
        raise typer.Exit(1)

    # Step 4: Initialize session context
    context = SessionContext(
        state=SessionState.IDLE,
        project_name=project.name,
        project_path=project.path,
    )

    # TODO: Load memory items for this project
    show_project_loaded(project, memory_items=0)

    # Step 5: Enter conversation loop
    conversation_loop(context, config)


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


if __name__ == "__main__":
    app()
