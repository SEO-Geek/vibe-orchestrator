"""
Vibe CLI - Project Management UI

Project selection, display, and context loading.
"""

import platform
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from vibe.config import Project, VibeConfig
from vibe.memory.task_history import get_context_for_glm

console = Console()


def get_git_log(project_path: Path, num_commits: int = 10) -> str:
    """Get recent git commits for context."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{num_commits}", "--oneline", "--no-decorate"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def get_platform_info() -> str:
    """Get platform/environment info that might affect code."""
    info = []
    info.append(f"OS: {platform.system()} {platform.release()}")
    info.append(f"Python: {platform.python_version()}")

    # Check if running in WSL
    try:
        with open("/proc/version", "r") as f:
            version = f.read().lower()
            if "microsoft" in version or "wsl" in version:
                info.append("Environment: WSL (Windows Subsystem for Linux)")
    except Exception:
        pass

    return "\n".join(info)


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


def load_project_context(project: Project, max_file_size: int = 8000) -> str:
    """
    Load comprehensive project context for Gemini and Claude.

    Includes:
    - Project name and path
    - Platform/environment info (OS, WSL, etc.)
    - STARMAP.md (project architecture)
    - CLAUDE.md (coding conventions)
    - CHANGELOG.md (recent changes)
    - Git log (recent commits)
    - Task history

    Args:
        project: The project to load context for
        max_file_size: Max chars to read from each file (default 8000)

    Returns:
        Comprehensive context string
    """
    context_parts = []

    # Header
    context_parts.append(f"# PROJECT: {project.name}")
    context_parts.append(f"Path: {project.path}")
    context_parts.append("")

    # Platform info - CRITICAL for hardware/OS-specific code
    platform_info = get_platform_info()
    if platform_info:
        context_parts.append("## ENVIRONMENT")
        context_parts.append(platform_info)
        context_parts.append("")

    # Load STARMAP.md - project architecture
    starmap_path = project.starmap_path
    if starmap_path.exists():
        try:
            content = starmap_path.read_text()[:max_file_size]
            context_parts.append("## PROJECT ARCHITECTURE (STARMAP.md)")
            context_parts.append(content)
            context_parts.append("")
        except Exception:
            pass

    # Load CLAUDE.md - coding conventions
    claude_md_path = project.claude_md_path
    if claude_md_path.exists():
        try:
            content = claude_md_path.read_text()[:max_file_size]
            context_parts.append("## CODING CONVENTIONS (CLAUDE.md)")
            context_parts.append(content)
            context_parts.append("")
        except Exception:
            pass

    # Load CHANGELOG.md - recent changes
    changelog_path = Path(project.path) / "CHANGELOG.md"
    if changelog_path.exists():
        try:
            content = changelog_path.read_text()[:4000]  # Less for changelog
            context_parts.append("## RECENT CHANGES (CHANGELOG.md)")
            context_parts.append(content)
            context_parts.append("")
        except Exception:
            pass

    # Git log - recent commits
    git_log = get_git_log(Path(project.path), num_commits=15)
    if git_log:
        context_parts.append("## RECENT COMMITS")
        context_parts.append(git_log)
        context_parts.append("")

    # Task history from current session
    task_context = get_context_for_glm()
    if task_context:
        context_parts.append("## SESSION HISTORY")
        context_parts.append(task_context)
        context_parts.append("")

    return "\n".join(context_parts)


def load_project_context_for_claude(project: Project) -> str:
    """
    Load project context specifically for Claude execution.

    This is a condensed version focused on what Claude needs to execute tasks:
    - Project conventions (CLAUDE.md)
    - Architecture overview (STARMAP.md summary)
    - Platform info (critical for hardware-specific code)

    Args:
        project: The project to load context for

    Returns:
        Context string for Claude's prompt
    """
    context_parts = []

    # Platform info FIRST - most critical for avoiding mistakes
    platform_info = get_platform_info()
    if platform_info:
        context_parts.append("ENVIRONMENT:")
        context_parts.append(platform_info)
        context_parts.append("")

    # CLAUDE.md - coding conventions (full file, very important)
    claude_md_path = project.claude_md_path
    if claude_md_path.exists():
        try:
            content = claude_md_path.read_text()[:10000]
            context_parts.append("PROJECT CONVENTIONS (CLAUDE.md):")
            context_parts.append(content)
            context_parts.append("")
        except Exception:
            pass

    # STARMAP.md - architecture (condensed)
    starmap_path = project.starmap_path
    if starmap_path.exists():
        try:
            content = starmap_path.read_text()[:4000]
            context_parts.append("PROJECT ARCHITECTURE (STARMAP.md):")
            context_parts.append(content)
            context_parts.append("")
        except Exception:
            pass

    return "\n".join(context_parts)
