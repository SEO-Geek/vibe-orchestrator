"""
Vibe CLI - Task Execution

Execute tasks with Claude and review with GLM.
"""

from rich.console import Console
from rich.panel import Panel

from vibe.claude.executor import ClaudeExecutor, TaskResult, ToolCall, get_git_diff
from vibe.glm.client import GLMClient, is_investigation_request

console = Console()


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
