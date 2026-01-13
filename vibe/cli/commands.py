"""
Vibe CLI - Slash Command Handlers

All /command handlers for the interactive conversation loop.
"""

import asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from vibe.claude.executor import ClaudeExecutor
from vibe.config import Project
from vibe.exceptions import ResearchError, GitHubError
from vibe.glm.client import GLMClient
from vibe.integrations import PerplexityClient, GitHubOps
from vibe.memory.keeper import VibeMemory
from vibe.memory.debug_session import DebugSession, AttemptResult
from vibe.memory.task_history import TaskHistory
from vibe.persistence.repository import VibeRepository
from vibe.state import SessionContext

from vibe.cli.project import load_project_context

console = Console()


def handle_help() -> None:
    """Handle /help command."""
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


def handle_status(context: SessionContext) -> None:
    """Handle /status command."""
    stats = context.get_stats()
    console.print(f"\n[bold]Session Status:[/bold]")
    console.print(f"  State: {stats['state']}")
    console.print(f"  Project: {stats['project']}")
    console.print(f"  Completed tasks: {stats['completed_tasks']}")
    console.print(f"  Errors: {stats['error_count']}")
    console.print(f"  Duration: {stats['duration_seconds']:.0f}s")
    console.print()


def handle_usage(glm_client: GLMClient) -> None:
    """Handle /usage command."""
    usage = glm_client.get_usage_stats()
    console.print(f"\n[bold]GLM Usage:[/bold]")
    console.print(f"  Model: {usage['model']}")
    console.print(f"  Requests: {usage['request_count']}")
    console.print(f"  Total tokens: {usage['total_tokens']}")
    console.print()


def handle_memory(memory: VibeMemory | None) -> None:
    """Handle /memory command."""
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


def handle_history() -> None:
    """Handle /history command."""
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


def handle_redo(
    project: Project,
    glm_client: GLMClient,
    context: SessionContext,
    memory: VibeMemory | None,
    execute_tasks_func,  # Callable passed in to avoid circular import
) -> None:
    """Handle /redo command."""
    failed_tasks = [t for t in TaskHistory.get_recent_tasks(20) if t.status == "failed"]
    if not failed_tasks:
        console.print("[yellow]No failed tasks to redo[/yellow]")
        return

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
    asyncio.run(execute_tasks_func(
        glm_client=glm_client,
        executor=ClaudeExecutor(project.path),
        project=project,
        tasks=[retry_task],
        memory=memory,
        context=context,
        user_request=f"Retry: {failed_tasks[0].description}",
    ))


def handle_convention(
    user_input: str,
    memory: VibeMemory | None,
) -> None:
    """Handle /convention command."""
    if not memory:
        console.print("[yellow]Memory not available[/yellow]")
        return

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
            return
        rest = parts[2]
        # Split key from convention text
        key_parts = rest.split(maxsplit=1)
        if len(key_parts) < 2:
            console.print("[red]Usage: /convention add <key> <convention text>[/red]")
            return
        key, convention = key_parts
        memory.save_convention(key, convention)
        console.print(f"[green]Saved convention: {key}[/green]")
    elif subcommand == "delete":
        if len(parts) < 3:
            console.print("[red]Usage: /convention delete <key>[/red]")
            return
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


def handle_research(
    user_input: str,
    project: Project,
    perplexity: PerplexityClient | None,
) -> None:
    """Handle /research command."""
    if not perplexity or not perplexity.is_available:
        console.print("[yellow]Perplexity not available (PERPLEXITY_API_KEY not set)[/yellow]")
        return

    query = user_input[9:].strip()  # Remove "/research "
    if not query:
        query = Prompt.ask("What do you want to research?")
    if query:
        with console.status("[bold blue]Researching...[/bold blue]"):
            try:
                result = asyncio.run(perplexity.research(query, context=load_project_context(project)))
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


def handle_github(github: GitHubOps | None) -> None:
    """Handle /github command."""
    if not github:
        console.print("[yellow]GitHub CLI not configured[/yellow]")
        return
    try:
        repo_info = github.get_repo_info()
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


def handle_issues(github: GitHubOps | None) -> None:
    """Handle /issues command."""
    if not github:
        console.print("[yellow]GitHub CLI not configured[/yellow]")
        return
    try:
        with console.status("[bold blue]Fetching issues...[/bold blue]"):
            issues = github.list_issues(limit=10)
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


def handle_prs(github: GitHubOps | None) -> None:
    """Handle /prs command."""
    if not github:
        console.print("[yellow]GitHub CLI not configured[/yellow]")
        return
    try:
        with console.status("[bold blue]Fetching pull requests...[/bold blue]"):
            prs = github.list_prs(limit=10)
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


def handle_debug(
    user_input: str,
    project: Project,
    memory: VibeMemory | None,
    debug_session: DebugSession | None,
) -> DebugSession | None:
    """
    Handle /debug command.

    Returns the updated debug_session (or None if ended).
    """
    parts = user_input.split(maxsplit=2)
    subcommand = parts[1] if len(parts) > 1 else "status"

    if subcommand == "start":
        # /debug start <problem description>
        if debug_session and debug_session.is_active:
            console.print("[yellow]Debug session already active. Use /debug end first.[/yellow]")
            return debug_session
        problem = parts[2] if len(parts) > 2 else Prompt.ask("Describe the problem")
        if problem:
            debug_session = DebugSession(
                project_path=project.path,
                problem=problem,
            )
            console.print(f"\n[green bold]Debug Session Started[/green bold]")
            console.print(f"  Problem: {problem}")
            console.print(f"  Initial commit: {debug_session.initial_commit or 'N/A'}")
            console.print()
            console.print("[dim]Use /debug preserve <feature> to add features that must work[/dim]")
            console.print("[dim]Use /debug hypothesis <text> to set current hypothesis[/dim]")
            # Save to memory
            if memory:
                memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "preserve":
        # /debug preserve <feature>
        if not debug_session:
            console.print("[yellow]No active debug session. Use /debug start first.[/yellow]")
            return debug_session
        feature = parts[2] if len(parts) > 2 else ""
        if not feature:
            feature = Prompt.ask("Feature to preserve")
        if feature:
            debug_session.add_must_preserve(feature)
            console.print(f"[green]Added to preservation list: {feature}[/green]")
            if memory:
                memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "hypothesis":
        # /debug hypothesis <hypothesis text>
        if not debug_session:
            console.print("[yellow]No active debug session. Use /debug start first.[/yellow]")
            return debug_session
        hypothesis = parts[2] if len(parts) > 2 else ""
        if not hypothesis:
            hypothesis = Prompt.ask("Current hypothesis")
        if hypothesis:
            debug_session.set_hypothesis(hypothesis)
            console.print(f"[green]Hypothesis set: {hypothesis}[/green]")
            if memory:
                memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "attempt":
        # /debug attempt <description>
        if not debug_session:
            console.print("[yellow]No active debug session. Use /debug start first.[/yellow]")
            return debug_session
        description = parts[2] if len(parts) > 2 else ""
        if not description:
            description = Prompt.ask("Describe the fix attempt")
        if description:
            attempt = debug_session.start_attempt(description)
            console.print(f"\n[bold]Attempt #{attempt.id} Started[/bold]")
            console.print(f"  Description: {description}")
            console.print(f"  Rollback to: {attempt.rollback_commit or 'N/A'}")
            console.print()
            console.print("[dim]Use /debug fail <reason> or /debug success after testing[/dim]")
            if memory:
                memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "fail":
        # /debug fail <reason>
        if not debug_session:
            console.print("[yellow]No active debug session.[/yellow]")
            return debug_session
        # Find pending attempt
        pending = [a for a in debug_session.attempts if a.result == AttemptResult.PENDING]
        if not pending:
            console.print("[yellow]No pending attempt to mark as failed.[/yellow]")
            return debug_session
        reason = parts[2] if len(parts) > 2 else ""
        if not reason:
            reason = Prompt.ask("Why did it fail?")
        if reason:
            attempt = pending[-1]
            debug_session.complete_attempt(
                attempt.id,
                AttemptResult.FAILED,
                reason
            )
            console.print(f"[red]Attempt #{attempt.id} marked as FAILED[/red]")
            console.print(f"  Reason: {reason}")
            if memory:
                memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "partial":
        # /debug partial <what helped>
        if not debug_session:
            console.print("[yellow]No active debug session.[/yellow]")
            return debug_session
        pending = [a for a in debug_session.attempts if a.result == AttemptResult.PENDING]
        if not pending:
            console.print("[yellow]No pending attempt to mark.[/yellow]")
            return debug_session
        reason = parts[2] if len(parts) > 2 else ""
        if not reason:
            reason = Prompt.ask("How did it help?")
        if reason:
            attempt = pending[-1]
            debug_session.complete_attempt(
                attempt.id,
                AttemptResult.PARTIAL,
                reason
            )
            console.print(f"[yellow]Attempt #{attempt.id} marked as PARTIAL[/yellow]")
            console.print(f"  Result: {reason}")
            if memory:
                memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "success":
        # /debug success
        if not debug_session:
            console.print("[yellow]No active debug session.[/yellow]")
            return debug_session
        pending = [a for a in debug_session.attempts if a.result == AttemptResult.PENDING]
        if not pending:
            console.print("[yellow]No pending attempt to mark as success.[/yellow]")
            return debug_session
        attempt = pending[-1]
        debug_session.complete_attempt(
            attempt.id,
            AttemptResult.SUCCESS,
            "Fix worked"
        )
        console.print(f"[green bold]Attempt #{attempt.id} marked as SUCCESS![/green bold]")
        if memory:
            memory.save_debug_session(debug_session.to_dict())
        return debug_session

    elif subcommand == "status":
        # /debug status - Show debug session state
        if not debug_session:
            console.print("[dim]No active debug session.[/dim]")
            console.print("[dim]Use /debug start <problem> to begin.[/dim]")
            return debug_session

        console.print(f"\n[bold]Debug Session Status[/bold]")
        console.print(f"  Problem: {debug_session.problem}")
        console.print(f"  Hypothesis: {debug_session.current_hypothesis or '(not set)'}")
        console.print(f"  Features to preserve: {len(debug_session.must_preserve)}")
        for feat in debug_session.must_preserve:
            console.print(f"    • {feat}")
        console.print()
        console.print(f"  [bold]Attempts ({len(debug_session.attempts)}):[/bold]")
        for attempt in debug_session.attempts:
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
        return debug_session

    elif subcommand == "context":
        # /debug context - Show what gets injected into Claude
        if not debug_session:
            console.print("[yellow]No active debug session.[/yellow]")
            return debug_session
        context = debug_session.get_context_for_claude()
        console.print(Panel(context, title="Debug Context for Claude", border_style="blue"))
        return debug_session

    elif subcommand == "end":
        # /debug end - End the debug session
        if not debug_session:
            console.print("[yellow]No active debug session.[/yellow]")
            return None
        debug_session.is_active = False
        if memory:
            memory.save_debug_session(debug_session.to_dict())
        console.print("[green]Debug session ended.[/green]")
        console.print(debug_session.get_summary())
        return None

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
        return debug_session


def handle_rollback(
    user_input: str,
    debug_session: DebugSession | None,
) -> None:
    """Handle /rollback command."""
    if not debug_session:
        console.print("[yellow]No active debug session.[/yellow]")
        return

    parts = user_input.split()
    if len(parts) > 1:
        try:
            attempt_id = int(parts[1])
            if debug_session.rollback_to_attempt(attempt_id):
                console.print(f"[green]Rolled back to before attempt #{attempt_id}[/green]")
            else:
                console.print(f"[red]Rollback failed - attempt #{attempt_id} not found or no checkpoint[/red]")
        except ValueError:
            if parts[1] == "start":
                if debug_session.rollback_to_start():
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
        console.print(f"  start ({debug_session.initial_commit})")
        for attempt in debug_session.attempts:
            if attempt.rollback_commit:
                console.print(f"  #{attempt.id} ({attempt.rollback_commit})")
