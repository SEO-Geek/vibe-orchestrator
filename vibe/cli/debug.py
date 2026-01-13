"""
Vibe CLI - Debug Workflow

GLM-driven debugging workflow with Claude as hands.
"""

import logging
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from vibe.claude.executor import ClaudeExecutor, TaskResult
from vibe.config import Project
from vibe.glm.client import GLMClient
from vibe.glm.debug_state import DebugContext
from vibe.glm.prompts import DEBUG_CLAUDE_PROMPT
from vibe.memory.keeper import VibeMemory
from vibe.memory.task_history import add_task

logger = logging.getLogger(__name__)
console = Console()


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
    console.file.flush() if hasattr(console, 'file') else None
    sys.stdout.flush()
    sys.stderr.flush()
