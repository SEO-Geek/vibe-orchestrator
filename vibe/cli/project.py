"""
Vibe CLI - Project Management UI

Project selection, display, and context loading.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from vibe.config import Project, VibeConfig
from vibe.memory.task_history import get_context_for_glm

console = Console()


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
